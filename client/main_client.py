import cv2
import numpy as np
import os
import time
import sys
from tensorflow.keras.models import load_model

# Add root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from client.vision_utils import VisionSystem, extract_keypoints, prob_viz

# --- CONFIGURATION ---
THRESHOLD = 0.5         
SKIP_FRAME_PREDICTION = 5 
SENTENCE_TIMEOUT = 2.0  

# --- INITIALIZATION ---
print("Loading LSTM Brain...")
try:
    model = load_model(config.MODEL_PATH)
    print(f"Brain Loaded from {config.MODEL_PATH}.")
except Exception as e:
    print(f"CRITICAL: Could not load model from {config.MODEL_PATH}. {e}")
    exit()

print("Initializing Vision...")
vision_sys = VisionSystem()
print("Vision Ready.")

colors = [(245,117,16), (117,245,16), (16,117,245)] 

# --- REAL-TIME LOOP ---
sequence = []
sentence = []
predictions = []

last_action_time = time.time()
prev_frame_time = 0

current_prediction = 0
res = [0] * len(config.ACTIONS)
frame_counter = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    print("Starting Optimized Feed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # 1. & 2. DETECT & EXTRACT
        pose_res, hand_res, face_res = vision_sys.process_frame(frame)
        keypoints = extract_keypoints(pose_res, hand_res, face_res)
        
        sequence.append(keypoints)
        sequence = sequence[-config.SEQUENCE_LENGTH:]

        # 3. PREDICT
        if len(sequence) == config.SEQUENCE_LENGTH:
            if frame_counter % SKIP_FRAME_PREDICTION == 0:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                current_prediction = np.argmax(res)
                predictions.append(current_prediction)
                
                # Voting Logic
                if len(predictions) >= 10:
                    last_10 = predictions[-10:]
                    counts = np.bincount(last_10)
                    most_common = np.argmax(counts)
                    
                    if most_common == current_prediction and res[current_prediction] > THRESHOLD:
                        if len(sentence) > 0: 
                            if config.ACTIONS[current_prediction] != sentence[-1]:
                                sentence.append(config.ACTIONS[current_prediction])
                                last_action_time = time.time()
                        else:
                            sentence.append(config.ACTIONS[current_prediction])
                            last_action_time = time.time()

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
            frame = prob_viz(res, config.ACTIONS, frame, colors)

        # Timeout
        if time.time() - last_action_time > SENTENCE_TIMEOUT:
            sentence = []

        # UI
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS: {int(fps)}', (500, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)
        frame_counter += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()