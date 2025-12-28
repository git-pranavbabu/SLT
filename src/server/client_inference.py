import asyncio
import websockets
import json
import random
import time

async def test_inference():
    uri = "ws://localhost:8000/ws"
    
    # 1662 zeros
    dummy_frame = [0.0] * 1692
    
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to Inference Engine.")
        
        # We send 40 frames.
        # Frames 1-29: Server should say "buffering"
        # Frames 30-40: Server should return "prediction"
        
        for i in range(40):
            start = time.perf_counter()
            
            await websocket.send(json.dumps(dummy_frame))
            response = await websocket.recv()
            data = json.loads(response)
            
            dt = (time.perf_counter() - start) * 1000
            
            if data['status'] == 'buffering':
                print(f"Frame {i+1}: ⏳ Buffering ({data['frames']}/30)")
            else:
                print(f"Frame {i+1}: 🧠 PREDICTION: {data['action']} ({data['confidence']:.2f}) | Latency: {dt:.1f}ms")
            
            # Simulate 30 FPS
            await asyncio.sleep(0.033)

if __name__ == "__main__":
    asyncio.run(test_inference())