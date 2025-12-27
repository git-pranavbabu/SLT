import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the saved brain
model = load_model('../action.h5') # Load the model with the correct path
print("Model loaded successfully.")

# 2. Create a "Mock" Input
# Shape must match: (1 video, 30 frames, 1692 features)
mock_input = np.random.rand(1, 30, 1692)

# 3. Make a Prediction
prediction = model.predict(mock_input)

# 4. Decode
# prediction looks like [[0.1, 0.9]] (Probabilities)
actions = np.array(['drink', 'hello']) 
predicted_class = np.argmax(prediction) # Returns 0 or 1

print(f"Raw Probabilities: {prediction}")
print(f"Predicted Action: {actions[predicted_class]}")