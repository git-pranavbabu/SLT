import asyncio
import websockets

async def ping_server():
    uri = "ws://localhost:8000/ws"
    
    print(f"🔌 Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")
        
        # Send a message
        message = "PING"
        print(f"📤 Sending: {message}")
        await websocket.send(message)
        
        # Wait for reply
        response = await websocket.recv()
        print(f"mn Received: {response}")
        
        # Keep connection open for a second just to show stability
        await asyncio.sleep(1)
        print("🔌 Closing connection.")

if __name__ == "__main__":
    asyncio.run(ping_server())