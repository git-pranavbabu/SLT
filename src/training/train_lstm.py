import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config

# --- 1. DATA LOADING ---
print("Loading Data...")
label_map = {label:num for num, label in enumerate(config.ACTIONS)}
sequences, labels = [], []

for action in config.ACTIONS:
    action_path = os.path.join(config.DATA_PROCESSED_DIR, action)
    if not os.path.exists(action_path):
        print(f"Skipping missing folder: {action_path}")
        continue
        
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    for file_name in files:
        window = np.load(os.path.join(action_path, file_name))
        # Ensure shape matches config
        target_shape = (config.SEQUENCE_LENGTH, window.shape[1]) 
        if window.shape == target_shape:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Check for data
if len(X) == 0:
    print("CRITICAL ERROR: No data loaded.")
    exit()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. ADVANCED MODEL ARCHITECTURE ---
input_shape = (config.SEQUENCE_LENGTH, X.shape[2]) # (30, 1692)
num_classes = len(config.ACTIONS)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

# Create log directory
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- 3. CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(config.MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.00001),
    TensorBoard(log_dir=config.LOG_DIR)
]

# --- 4. TRAINING ---
print("\nStarting Professional Training...")
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=callbacks)

# --- 5. EVALUATION ---
print("\nEvaluating on Test Set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

model.save(config.MODEL_PATH)
print(f"Final Model saved to {config.MODEL_PATH}")