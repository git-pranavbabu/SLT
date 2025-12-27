import os
import numpy as np

# --- PATHS ---
# Root directory is the parent of 'config.py'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# --- MODEL CONSTANTS ---
ACTIONS = np.array(['drink', 'hello']) 
SEQUENCE_LENGTH = 30
MODEL_NAME = 'sign_model.h5'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# --- MEDIAPIPE CONSTANTS ---
TARGET_FACE_DIMS = 478
TARGET_POSE_DIMS = 33
TARGET_HAND_DIMS = 21

# --- TRAINING CONSTANTS ---
LOG_DIR = os.path.join(ROOT_DIR, 'training', 'logs')
