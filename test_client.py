#!/usr/bin/env python3
"""
Simple test client for CSM server.

Usage:
    python test_client.py [--server ws://localhost:8765/ws]
"""

import asyncio
import base64
import json
import argparse
import time

import websockets


async def test_basic_connection(server_url: str):
    """Test basic connection and session initialization."""
    print(f"Connecting to {server_url}...")
    
    async with websockets.connect(server_url) as ws:
        # Initialize session
        session_id = f"test_{int(time.time())}"
        init_msg = {
            "type": "init",
            "session_id": session_id,
            "speaker_id": 0
        }
        await ws.send(json.dumps(init_msg))
        print(f"Sent init message for session {session_id}")
        
        # Wait for ready
        response = await ws.recv()
        data = json.loads(response)
        print(f"Received: {data}")
        
        if data.get("type") == "ready":
            print("Session initialized successfully!")
        else:
            print(f"Unexpected response: {data}")
            return False
        
        # Test generate (without context audio)
        print("\nTesting generation...")
        gen_msg = {
            "type": "generate",
            "text": "Hello, this is a test of the CSM streaming server."
        }
        await ws.send(json.dumps(gen_msg))
        print("Sent generate request")
        
        # Receive audio chunks
        chunk_count = 0
        total_bytes = 0
        start_time = time.time()
        first_chunk_time = None
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "audio":
                chunk_count += 1
                audio_bytes = base64.b64decode(data["data"])
                total_bytes += len(audio_bytes)
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = (first_chunk_time - start_time) * 1000
                    print(f"First chunk received! Latency: {latency:.1f}ms")
                
                if chunk_count % 50 == 0:
                    print(f"Received {chunk_count} chunks, {total_bytes} bytes")
                    
            elif data["type"] == "done":
                elapsed = time.time() - start_time
                audio_duration = total_bytes / 8000  # ulaw 8kHz
                print(f"\nGeneration complete!")
                print(f"  Chunks: {chunk_count}")
                print(f"  Total bytes: {total_bytes}")
                print(f"  Audio duration: {audio_duration:.2f}s")
                print(f"  Wall time: {elapsed:.2f}s")
                print(f"  RTF: {elapsed/audio_duration:.2f}x" if audio_duration > 0 else "")
                break
                
            elif data["type"] == "error":
                print(f"Error: {data.get('message')}")
                return False
        
        # Close session
        close_msg = {"type": "close"}
        await ws.send(json.dumps(close_msg))
        print("\nSession closed.")
        
    return True


async def test_with_audio(server_url: str, audio_file: str = None):
    """Test with simulated audio input."""
    print(f"Connecting to {server_url}...")
    
    async with websockets.connect(server_url) as ws:
        # Initialize session
        session_id = f"test_audio_{int(time.time())}"
        init_msg = {
            "type": "init",
            "session_id": session_id,
            "speaker_id": 0
        }
        await ws.send(json.dumps(init_msg))
        
        response = await ws.recv()
        data = json.loads(response)
        if data.get("type") != "ready":
            print(f"Failed to initialize: {data}")
            return False
        print("Session initialized")
        
        # Simulate sending audio chunks (silence)
        print("Sending simulated audio...")
        silence_chunk = b'\xff' * 160  # 20ms of ulaw silence
        
        for i in range(50):  # 1 second of audio
            audio_msg = {
                "type": "audio",
                "data": base64.b64encode(silence_chunk).decode()
            }
            await ws.send(json.dumps(audio_msg))
            await asyncio.sleep(0.02)  # 20ms
        
        print("Sent 1 second of audio")
        
        # End user turn
        turn_end_msg = {
            "type": "user_turn_end",
            "text": "Hello, how are you?"
        }
        await ws.send(json.dumps(turn_end_msg))
        print("Sent user_turn_end")
        
        # Wait for acknowledgment
        response = await ws.recv()
        data = json.loads(response)
        print(f"Turn end response: {data}")
        
        # Generate response
        gen_msg = {
            "type": "generate",
            "text": "I'm doing well, thank you for asking! How can I help you today?"
        }
        await ws.send(json.dumps(gen_msg))
        print("Sent generate request")
        
        # Receive audio
        chunk_count = 0
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "audio":
                chunk_count += 1
            elif data["type"] == "done":
                print(f"Generation complete! {chunk_count} chunks")
                break
            elif data["type"] == "error":
                print(f"Error: {data.get('message')}")
                break
        
        # Close
        await ws.send(json.dumps({"type": "close"}))
        print("Session closed")
        
    return True


async def main():
    parser = argparse.ArgumentParser(description="Test CSM server")
    parser.add_argument("--server", default="ws://localhost:8765/ws", help="Server URL")
    parser.add_argument("--with-audio", action="store_true", help="Test with simulated audio")
    args = parser.parse_args()
    
    try:
        if args.with_audio:
            success = await test_with_audio(args.server)
        else:
            success = await test_basic_connection(args.server)
        
        if success:
            print("\n✓ Test passed!")
        else:
            print("\n✗ Test failed!")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
