import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import asyncio
import websockets
import json
import time

# --- CONFIGURATION ---
URI = "ws://localhost:8000/ws"
TARGET_FACE_DIMS = 478
TARGET_POSE_DIMS = 33
TARGET_HAND_DIMS = 21

# --- MANUAL HAND CONNECTIONS (The "Hardcoded" Map) ---
# This replaces mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index
    (0, 9), (9, 10), (10, 11), (11, 12),# Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# --- 1. INITIALIZATION ---
print("Initializing Vision (Tasks API)...")
MODEL_ASSETS = {
    'pose': '../models/pose_landmarker_lite.task',
    'hand': '../models/hand_landmarker.task',
    'face': '../models/face_landmarker.task'
}

try:
    base_pose = python.BaseOptions(model_asset_path=MODEL_ASSETS['pose'])
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=base_pose))

    base_hand = python.BaseOptions(model_asset_path=MODEL_ASSETS['hand'])
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2))

    base_face = python.BaseOptions(model_asset_path=MODEL_ASSETS['face'])
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_face))
    print("Vision Ready.")
except Exception as e:
    print(f"CRITICAL: Model assets missing. {e}")
    exit()

# --- 2. EXTRACTION ENGINE ---
def extract_keypoints(pose_res, hand_res, face_res):
    # Pose
    if pose_res.pose_landmarks:
        pose_flat = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]]).flatten()
    else:
        pose_flat = np.zeros(TARGET_POSE_DIMS * 4)
    
    # Face (Enforce 478)
    if face_res.face_landmarks:
        landmarks = face_res.face_landmarks[0]
        count = len(landmarks)
        flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        if count == TARGET_FACE_DIMS:
            face_flat = flat
        elif count < TARGET_FACE_DIMS:
            padding = np.zeros((TARGET_FACE_DIMS - count) * 3)
            face_flat = np.concatenate([flat, padding])
        else:
            face_flat = flat[:TARGET_FACE_DIMS*3]
    else:
        face_flat = np.zeros(TARGET_FACE_DIMS * 3)
        
    # Hands
    lh_flat = np.zeros(TARGET_HAND_DIMS * 3)
    rh_flat = np.zeros(TARGET_HAND_DIMS * 3)
    if hand_res.hand_landmarks:
        for i, handedness in enumerate(hand_res.handedness):
            if not handedness: continue
            label = handedness[0].category_name
            landmarks = hand_res.hand_landmarks[i]
            flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            if label == 'Left': lh_flat = flat
            else: rh_flat = flat
            
    return np.concatenate([pose_flat, face_flat, lh_flat, rh_flat])

# --- 3. ROBUST DRAWING ENGINE (No mp.solutions dependency) ---
def draw_debug(image, pose_res, hand_res):
    h, w, c = image.shape
    
    # Draw Pose (Blue Circles)
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)

    # Draw Hands (Green Lines, Pink Circles)
    if hand_res.hand_landmarks:
        for hand_lms in hand_res.hand_landmarks:
            # 1. Draw Landmarks
            points = []
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                cv2.circle(image, (cx, cy), 3, (147, 20, 255), -1) # Pink
            
            # 2. Draw Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2) # Green

# --- 4. THE ASYNC CLIENT ---
async def main():
    cap = cv2.VideoCapture(0)
    
    sentence = []
    server_status = "Connecting..."
    current_action = "..."
    conf = 0.0

    print(f"🔌 Connecting to Nervous System at {URI}...")
    try:
        async with websockets.connect(URI) as websocket:
            print("✅ Connected! Starting Stream...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 1. VISION
                image = cv2.flip(frame, 1) # Flip for UI
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Raw for AI
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
                
                pose_res = pose_landmarker.detect(mp_image)
                hand_res = hand_landmarker.detect(mp_image)
                face_res = face_landmarker.detect(mp_image)

                # 2. EXTRACT
                keypoints = extract_keypoints(pose_res, hand_res, face_res)
                payload = json.dumps(keypoints.tolist())

                # 3. NETWORK IO
                await websocket.send(payload)
                response = await websocket.recv()
                data = json.loads(response)

                # 4. HANDLE RESPONSE
                if data['status'] == 'prediction':
                    server_status = "Online"
                    current_action = data['action']
                    conf = data['confidence']
                    
                    if conf > 0.8: # Threshold
                        if len(sentence) == 0 or current_action != sentence[-1]:
                            sentence.append(current_action)
                        if len(sentence) > 5: sentence = sentence[-5:]
                
                elif data['status'] == 'buffering':
                    server_status = f"Buffering {data['frames']}/30"

                # 5. DRAW DEBUG (Safe Mode)
                draw_debug(image, pose_res, hand_res)

                # Draw UI
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f"Server: {server_status} | Action: {current_action} ({conf:.2f})", (10,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Remote Interpreter', image)
                await asyncio.sleep(0.001)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"❌ Connection Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())