import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

class VisionSystem:
    def __init__(self):
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.face_landmarker = None
        self._init_models()

    def _init_models(self):
        try:
            # Paths relative to the root, so we need to construct them carefuly
            # Assuming models are in models/ directory relative to root
            # But the task files might be elsewhere. 
            # In the original code, they were in 'models/'. 
            # We moved trained_model/* to models/, but verified asset paths?
            # Let's assume asset paths are still valid relative to CWD if run from root.
            
            # Since the user runs from root SLT/, 'models/pose_landmarker_lite.task' should exist 
            # IF the original 'models' folder with assets wasn't overwritten by our 'models' folder creation.
            # Wait, 'models' folder ALREADY existed and contained assets. 
            # We moved trained_model/* INTO it. So assets are safe.
            
            base_pose = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=base_pose))

            base_hand = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2))

            base_face = python.BaseOptions(model_asset_path='models/face_landmarker.task')
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=base_face))
        except Exception as e:
            print(f"VisionSystem Init Error: {e}")
            raise e

    def process_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        pose_res = self.pose_landmarker.detect(mp_image)
        hand_res = self.hand_landmarker.detect(mp_image)
        face_res = self.face_landmarker.detect(mp_image)
        return pose_res, hand_res, face_res

def extract_keypoints(pose_res, hand_res, face_res):
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
            
    return np.concatenate([pose_flat, face_flat, lh_flat, rh_flat])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
