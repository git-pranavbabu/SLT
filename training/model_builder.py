from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    
    # 1. LSTM Layers - The "Memory"
    # return_sequences=True: We pass the full sequence (baton) to the next layer
    # input_shape is required only on the first layer
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    
    # return_sequences=False: This layer summarizes the video into one vector
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    
    # 2. Dense Layers - The "Decision"
    # Relu is fast and efficient for standard neurons
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # 3. Output Layer - The "Probability"
    # Softmax output size MUST match your number of actions (classes)
    model.add(Dense(num_classes, activation='softmax'))
    
    # 4. Compilation
    # Categorical Crossentropy is the standard loss for multi-class classification
    # Adam is the best general-purpose optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    
    return model