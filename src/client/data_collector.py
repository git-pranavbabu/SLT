import cv2
import numpy as np
import os
import sys
import json
import mediapipe as mp
import logging
import urllib.parse
from concurrent.futures import ProcessPoolExecutor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Add root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config

# --- CONFIGURATION ---
JSON_PATH = os.path.join(config.ROOT_DIR, 'WLASL_v0.3.json')
NUM_WORKERS = 4

# --- LOGGING ---
logging.basicConfig(
    filename='processor.log', 
    level=logging.WARNING, # Captures Warnings and Errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- GLOBAL WORKER STATE ---
worker_models = {}

# --- INITIALIZER ---
def init_worker():
    """Initializes models once per worker process."""
    global worker_models
    try:
        # Note: These paths are relative to CWD when running the script.
        # Assuming script run from root SLT/ folder.
        # Assets should be in 'models/' folder.
        
        base_pose = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
        worker_models['pose'] = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_pose))
        
        base_hand = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        worker_models['hand'] = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2))
        
        base_face = python.BaseOptions(model_asset_path='models/face_landmarker.task')
        worker_models['face'] = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_face))
    except Exception as e:
        logging.critical(f"Worker failed to initialize models: {e}")
        raise e

# --- ROBUST ID EXTRACTION ---
def get_yt_identifier(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return urllib.parse.parse_qs(parsed.query).get('v', [None])[0]
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    return None

# --- WORKER FUNCTION ---
def process_single_video(task_data):
    video_path, start_frame, end_frame, gloss, video_id, output_path = task_data
    global worker_models

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open {video_id} at {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame <= 0 or end_frame >= total_frames:
            end_frame = total_frames - 1
            
        if start_frame >= end_frame:
            logging.warning(f"Invalid bounds for {video_id}: {start_frame}-{end_frame} (Total: {total_frames})")
            cap.release()
            return None
        
        target_indices = np.linspace(start_frame, end_frame, config.SEQUENCE_LENGTH, dtype=int)
        target_set = set(target_indices)
        
        final_sequence = []
        
        current_frame = 0
        while current_frame < start_frame:
            if not cap.grab():
                break
            current_frame += 1
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame in target_set:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Inference
                pose_res = worker_models['pose'].detect(mp_image)
                hand_res = worker_models['hand'].detect(mp_image)
                face_res = worker_models['face'].detect(mp_image)
                
                # --- FLATTEN ---
                # Pose
                if pose_res.pose_landmarks:
                    pose_flat = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]]).flatten()
                else:
                    pose_flat = np.zeros(config.TARGET_POSE_DIMS * 4)
                
                # Face
                if face_res.face_landmarks:
                    landmarks = face_res.face_landmarks[0]
                    count = len(landmarks)
                    flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    if count == config.TARGET_FACE_DIMS:
                        face_flat = flat
                    elif count < config.TARGET_FACE_DIMS:
                        padding = np.zeros((config.TARGET_FACE_DIMS - count) * 3)
                        face_flat = np.concatenate([flat, padding])
                    else:
                        face_flat = flat[:config.TARGET_FACE_DIMS*3]
                else:
                    face_flat = np.zeros(config.TARGET_FACE_DIMS * 3)
                    
                # Hands
                lh_flat = np.zeros(config.TARGET_HAND_DIMS * 3)
                rh_flat = np.zeros(config.TARGET_HAND_DIMS * 3)
                if hand_res.hand_landmarks:
                    for i, handedness in enumerate(hand_res.handedness):
                        if not handedness: continue
                        label = handedness[0].category_name
                        landmarks = hand_res.hand_landmarks[i]
                        flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                        if label == 'Left': lh_flat = flat
                        else: rh_flat = flat
                
                final_sequence.append(np.concatenate([pose_flat, face_flat, lh_flat, rh_flat]))
                
                if len(final_sequence) >= config.SEQUENCE_LENGTH:
                    break
            
            current_frame += 1
            
        cap.release()

        # Padding
        if len(final_sequence) < config.SEQUENCE_LENGTH:
            if len(final_sequence) >= config.SEQUENCE_LENGTH - 5 and len(final_sequence) > 0:
                logging.info(f"Padding {video_id} from {len(final_sequence)} to {config.SEQUENCE_LENGTH}")
                while len(final_sequence) < config.SEQUENCE_LENGTH:
                    final_sequence.append(final_sequence[-1])
            else:
                logging.warning(f"Short sequence for {video_id}: {len(final_sequence)}/{config.SEQUENCE_LENGTH}")
                return None

        np_seq = np.array(final_sequence)
        np.save(output_path, np_seq)
        
        meta = {
            'gloss': gloss,
            'id': video_id,
            'shape': str(np_seq.shape)
        }
        with open(output_path.replace('.npy', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        return True

    except Exception as e:
        logging.error(f"CRASH {video_id}: {e}")
        return None

# --- MAIN ---
def main():
    print("--- SLT DATA COLLECTOR ---")
    
    # Check JSON
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    print("Parsing JSON...")
    with open(JSON_PATH, 'r') as f:
        content = json.load(f)

    tasks = []
    
    for entry in content:
        gloss = entry['gloss']
        
        # Filter based on config actions (if they overlap with glosses)
        # Note: Config actions are ['drink', 'hello'], but WLASL might have them lower cased.
        if gloss not in config.ACTIONS: continue
        
        gloss_dir = os.path.join(config.DATA_PROCESSED_DIR, gloss)
        os.makedirs(gloss_dir, exist_ok=True)
        
        for inst in entry['instances']:
            url = inst['url']
            wlasl_id = inst['video_id']
            yt_id = get_yt_identifier(url)
            if not yt_id: continue
            
            # Locate Source
            vid_path = os.path.join(config.DATA_RAW_DIR, f"{yt_id}.mp4")
            if not os.path.exists(vid_path):
                vid_path = os.path.join(config.DATA_RAW_DIR, f"{yt_id}.mkv")
                if not os.path.exists(vid_path): continue
            
            # Check Destination
            out_path = os.path.join(gloss_dir, f"{wlasl_id}.npy")
            if os.path.exists(out_path): continue
            
            f_start = inst['frame_start'] - 1
            f_end = inst['frame_end']
            if f_end > 0: f_end -= 1
            
            tasks.append((vid_path, f_start, f_end, gloss, wlasl_id, out_path))

    print(f"Processing {len(tasks)} clips with {NUM_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        results = list(executor.map(process_single_video, tasks))
        
    success_count = len([r for r in results if r])
    print(f"Done. Processed: {success_count}/{len(tasks)}")
    print("Check processor.log for details.")

if __name__ == '__main__':
    main()