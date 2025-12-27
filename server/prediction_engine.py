import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

# Add root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class PredictionEngine:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = load_model(config.MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, sequence):
        if self.model is None:
            return None
        
        # Expect sequence shape: (1, 30, 1662)
        res = self.model.predict(sequence, verbose=0)
        return config.ACTIONS[np.argmax(res[0])]
