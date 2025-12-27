# Evocode Sign

Real-time sign language translation system using LSTM and MediaPipe.

## Structure
- **client/**: Runs on user device, captures video, extracts landmarks.
- **server/**: Backend API for inference and LLM refinement (future).
- **training/**: Scripts for model training.
- **data/**: Dataset storage.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run client: `python -m client.main_client`
3. Run training: `python -m training.train_lstm`
