import asyncio
import websockets
import json
import time

async def send_dummy_data():
    uri = "ws://localhost:8000/ws"
    
    # Simulate a single frame of landmarks (1662 floats)
    # 33 pose + 478 face + 21 left + 21 right = ~553 points * 3 dims = ~1662
    dummy_frame = [0.0] * 1662
    
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to Nervous System.")
        
        # Send 10 frames to simulate a stream
        for i in range(10):
            start_time = time.perf_counter()
            
            # 1. SERIALIZE (List -> JSON String)
            payload = json.dumps(dummy_frame)
            
            # 2. SEND
            await websocket.send(payload)
            
            # 3. RECEIVE
            response = await websocket.recv()
            data = json.loads(response)
            
            latency = (time.perf_counter() - start_time) * 1000
            print(f"Frame {i+1}: Server said {data['status']} (Latency: {latency:.2f}ms)")
            
            # Simulate 30 FPS (approx 33ms wait)
            await asyncio.sleep(0.033)

if __name__ == "__main__":
    asyncio.run(send_dummy_data())