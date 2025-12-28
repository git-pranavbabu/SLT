import os
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model
from config import config

# --- IMPORT YOUR COGNITIVE AGENT ---
from agent.agent_graph import app as agent_app

# --- GLOBAL SETTINGS ---
MODEL_PATH = config.MODEL_PATH
# UPDATE THIS LIST based on what you actually trained!
ACTIONS = config.ACTIONS 
SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
FEATURE_COUNT = config.FEATURE_COUNT
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD

# Global Brain Variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"🧠 LOADING VISION BRAIN FROM {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            # Warmup
            dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT))
            model.predict(dummy, verbose=0)
            print("✅ VISION BRAIN READY.")
        except Exception as e:
            print(f"❌ LOAD ERROR: {e}")
    else:
        print("❌ MODEL NOT FOUND.")
    yield
    print("💤 SHUTTING DOWN.")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"🔌 Client connected: {websocket.client}")
    
    # --- CONNECTION STATE ---
    frame_sequence = [] 
    gloss_sequence = []
    
    last_prediction = None
    last_prediction_time = 0
    
    try:
        while True:
            # --- A. RECEIVE FRAME ---
            data = await websocket.receive_json()
            
            frame_sequence.append(data)
            frame_sequence = frame_sequence[-SEQUENCE_LENGTH:]
            
            # Default response (Buffering)
            response = {"status": "buffering", "frames": len(frame_sequence)}

            # --- B. VISION INFERENCE ---
            if len(frame_sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(frame_sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                pred_idx = np.argmax(prediction)
                conf = float(prediction[pred_idx])
                
                current_word = ACTIONS[pred_idx]
                
                # --- C. WORD FILTERING (Debounce) ---
                current_time = time.time()
                
                if conf > CONFIDENCE_THRESHOLD and current_word != "nothing":
                    # Check if it's a new word OR enough time has passed
                    if current_word != last_prediction or (current_time - last_prediction_time > 2.0):
                        
                        # --- STRICT TRIGGER LOGIC ---
                        # If word is 'thanks', we treat it as the 'SEND' command.
                        # Otherwise, we just add the word to the list.
                        if current_word == "drink":
                             print("🚀 TRIGGER RECEIVED: User signed DRINK")
                             gloss_sequence.append("DRINK")
                        else:
                             print(f"📝 Added to buffer: {current_word}")
                             gloss_sequence.append(current_word)
                        
                        last_prediction = current_word
                        last_prediction_time = current_time
                        
                        # --- D. COGNITIVE INFERENCE (LangGraph) ---
                        # We invoke the graph every time a word is added.
                        # The Graph's "Conditional Edge" decides if it should Translate or Wait.
                        agent_result = agent_app.invoke({"gloss_sequence": gloss_sequence})
                        
                        # Check result
                        translation = agent_result.get("final_translation")
                        
                        if translation:
                            # GRAPH DECIDED TO TRANSLATE (Because it saw "SEND")
                            print(f"🗣️ AGENT SAYS: {translation}")
                            response = {
                                "status": "translated",
                                "gloss": " ".join(gloss_sequence[:-1]), # Show gloss without 'SEND'
                                "translation": translation
                            }
                            # Clear buffer explicitly
                            gloss_sequence = []
                        else:
                            # GRAPH DECIDED TO WAIT
                            response = {
                                "status": "collecting",
                                "gloss": " ".join(gloss_sequence),
                                "current_word": current_word
                            }

            # --- E. SEND RESULT ---
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("❌ Client disconnected.")
    except Exception as e:
        print(f"❌ Runtime Error: {e}")