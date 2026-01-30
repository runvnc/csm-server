"""
CSM (Conversational Speech Model) WebSocket Server

Streaming TTS server that maintains conversation context with user audio.
Designed for real-time phone conversations with zero upload latency.

Protocol:
- Client streams audio continuously (parallel to STT)
- On user turn end, client sends text transcription
- On generate request, server runs CSM with full audio context
- Server streams generated audio back as ulaw 8kHz chunks (optionally batched)
"""

import asyncio
import base64
import json
import logging
import socket
import os
import time
from typing import Dict, Optional, List

import torch
import torchaudio
import audioop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

# Add csm-streaming to path
import sys
CSM_PATH = os.environ.get('CSM_PATH', '/files/csm-streaming')
sys.path.insert(0, CSM_PATH)

from generator import Generator, Segment, load_csm_1b_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CSM_MODEL_PATH = os.environ.get('CSM_MODEL_PATH', '/files/csm-streaming')
CSM_DEVICE = os.environ.get('CSM_DEVICE', 'cuda')
CSM_PORT = int(os.environ.get('CSM_PORT', '8765'))
CSM_CODEBOOKS = int(os.environ.get('CSM_CODEBOOKS', '32'))

# WebSocket audio chunking / batching
# A "frame" is 160 bytes = 20ms at 8kHz ulaw.
CSM_WS_FRAME_BYTES = int(os.environ.get('CSM_WS_FRAME_BYTES', '160'))
CSM_WS_FRAME_MS = int(os.environ.get('CSM_WS_FRAME_MS', '20'))

# Batch up to ~100ms by default (i.e., 5 frames at 20ms/frame).
CSM_WS_BATCH_MS = int(os.environ.get('CSM_WS_BATCH_MS', '100'))
CSM_WS_BATCH_FRAMES_DEFAULT = max(1, (CSM_WS_BATCH_MS + CSM_WS_FRAME_MS - 1) // CSM_WS_FRAME_MS)
CSM_WS_BATCH_FRAMES = max(1, int(os.environ.get('CSM_WS_BATCH_FRAMES', str(CSM_WS_BATCH_FRAMES_DEFAULT))))

# Global generator (loaded once on startup)
generator: Optional[Generator] = None


class Session:
    """Manages state for a single conversation session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_buffer: List[bytes] = []  # ulaw chunks from user
        self.context_segments: List[Segment] = []  # conversation history
        self.reference_segments: List[Segment] = []  # voice cloning reference
        self.speaker_id = 0
        self.user_speaker_id = 1
        self.generating = False
        self.current_generation_id: Optional[str] = None
        self._generation_task: Optional[asyncio.Task] = None
        self.created_at = time.time()
        
    def add_audio_chunk(self, ulaw_bytes: bytes):
        """Buffer incoming audio chunk from user."""
        self.audio_buffer.append(ulaw_bytes)
        
    def finalize_user_turn(self, text: str):
        """Convert buffered audio to segment, add to context."""
        if not self.audio_buffer:
            logger.warning(f"Session {self.session_id}: No audio buffered for user turn")
            return
            
        # Combine all buffered audio
        all_audio = b''.join(self.audio_buffer)
        self.audio_buffer = []
        
        logger.info(f"Session {self.session_id}: Finalizing user turn with {len(all_audio)} bytes, text: {text[:50]}...")
        
        # Convert ulaw 8kHz → float 24kHz
        audio_tensor = ulaw_8k_to_float_24k(all_audio)
        
        # Create segment and add to context
        segment = Segment(
            speaker=self.user_speaker_id,
            text=text,
            audio=audio_tensor
        )
        self.context_segments.append(segment)
        
        # Trim context if too long
        self._trim_context()
        
    async def generate(self, text: str, websocket: WebSocket, generation_id: Optional[str] = None):
        """Generate AI response and stream audio back."""
        global generator
        
        if generator is None:
            await websocket.send_json({"type": "error", "message": "Generator not loaded"})
            return
            
        self.generating = True
        self.current_generation_id = generation_id
        logger.info(f"Session {self.session_id}: Generating response for: {text[:50]}...")
        
        try:
            context = self.reference_segments + self.context_segments
            
            # Collect audio for adding to context after generation
            all_audio_chunks = []
            
            gen_start = time.time()
            first_chunk_time = None
            frame_count = 0
            message_count = 0
            max_batch_bytes = CSM_WS_BATCH_FRAMES * CSM_WS_FRAME_BYTES

            batch_buf = bytearray()
            batch_started_at: Optional[float] = None

            async def flush_batch():
                nonlocal batch_buf, batch_started_at, message_count
                if not batch_buf:
                    return
                msg = {
                    "type": "audio",
                    "data": base64.b64encode(bytes(batch_buf)).decode(),
                    "frame_bytes": CSM_WS_FRAME_BYTES
                }
                if generation_id:
                    msg["generation_id"] = generation_id
                await websocket.send_json(msg)
                message_count += 1
                batch_buf = bytearray()
                batch_started_at = None
                # Yield to event loop (avoid starving)
                await asyncio.sleep(0)

            
            for chunk in generator.generate_stream(
                text=text,
                speaker=self.speaker_id,
                context=context,
                max_audio_length_ms=30000,
                temperature=0.7,
                topk=50
            ):
                if not self.generating:  # Interrupted
                    logger.info(f"Session {self.session_id}: Generation interrupted")
                    break
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info(f"Session {self.session_id}: First chunk latency: {(first_chunk_time - gen_start)*1000:.1f}ms")
                    
                all_audio_chunks.append(chunk)
                
                # Convert 24kHz float → ulaw 8kHz
                ulaw_bytes = float_24k_to_ulaw_8k(chunk)
                
                # Frame into CSM_WS_FRAME_BYTES and batch up to ~CSM_WS_BATCH_MS (default 100ms).
                for i in range(0, len(ulaw_bytes), CSM_WS_FRAME_BYTES):
                    frame = ulaw_bytes[i:i+CSM_WS_FRAME_BYTES]
                    if len(frame) < CSM_WS_FRAME_BYTES:
                        # Pad last frame if needed
                        frame = frame + b'xff' * (CSM_WS_FRAME_BYTES - len(frame))

                    if batch_started_at is None:
                        batch_started_at = time.time()

                    batch_buf.extend(frame)
                    frame_count += 1

                    now = time.time()
                    # Flush when we've reached target batch size OR we've held a batch for ~CSM_WS_BATCH_MS.
                    if len(batch_buf) >= max_batch_bytes:
                        await flush_batch()
                    elif batch_started_at is not None and (now - batch_started_at) * 1000.0 >= CSM_WS_BATCH_MS:
                        await flush_batch()

            # Flush any remainder (even if it's less than a full batch)
            await flush_batch()

                    
                    
            # Add AI response to context
            if all_audio_chunks and self.generating:
                full_audio = torch.cat(all_audio_chunks)
                segment = Segment(
                    speaker=self.speaker_id,
                    text=text,
                    audio=full_audio
                )
                self.context_segments.append(segment)
                self._trim_context()
                
            gen_time = time.time() - gen_start
            logger.info(f"Session {self.session_id}: Generation complete in {gen_time:.2f}s, {frame_count} frames sent in {message_count} websocket messages (batch_frames={CSM_WS_BATCH_FRAMES})")
            
            await websocket.send_json({"type": "done"})
            
        except Exception as e:
            logger.error(f"Session {self.session_id}: Generation error: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})
            
        finally:
            self.generating = False
            
    def interrupt(self):
        """Stop generation and clear audio buffer."""
        logger.info(f"Session {self.session_id}: Interrupted")
        self.generating = False
        self.audio_buffer = []
        
    def _trim_context(self, max_segments: int = 8):
        """Keep context within limits."""
        if len(self.context_segments) > max_segments:
            removed = len(self.context_segments) - max_segments
            self.context_segments = self.context_segments[-max_segments:]
            logger.info(f"Session {self.session_id}: Trimmed {removed} old segments from context")


# Session storage
sessions: Dict[str, Session] = {}


def ulaw_8k_to_float_24k(ulaw_bytes: bytes) -> torch.Tensor:
    """Convert ulaw 8kHz to float32 24kHz tensor."""
    # ulaw → PCM 16-bit
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    
    # bytes → tensor
    pcm_tensor = torch.frombuffer(bytearray(pcm_bytes), dtype=torch.int16).float() / 32768.0
    
    # Resample 8kHz → 24kHz
    pcm_tensor = pcm_tensor.unsqueeze(0)
    resampled = torchaudio.functional.resample(pcm_tensor, 8000, 24000)
    
    return resampled.squeeze(0)


def float_24k_to_ulaw_8k(audio_tensor: torch.Tensor) -> bytes:
    """Convert float32 24kHz tensor to ulaw 8kHz bytes."""
    # Ensure on CPU
    if audio_tensor.device.type != 'cpu':
        audio_tensor = audio_tensor.cpu()
    
    # Resample 24kHz → 8kHz
    audio_tensor = audio_tensor.unsqueeze(0)
    resampled = torchaudio.functional.resample(audio_tensor, 24000, 8000)
    audio_tensor = resampled.squeeze(0)
    
    # float → int16
    pcm_int16 = (audio_tensor * 32767.0).clamp(-32768, 32767).to(torch.int16)
    pcm_bytes = pcm_int16.numpy().tobytes()
    
    # PCM → ulaw
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
    
    return ulaw_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global generator
    
    logger.info(f"Loading CSM model from {CSM_MODEL_PATH}...")
    logger.info(f"Device: {CSM_DEVICE}, Codebooks: {CSM_CODEBOOKS}")
    
    try:
        generator = load_csm_1b_local(
            model_path=CSM_MODEL_PATH,
            device=CSM_DEVICE,
            audio_num_codebooks=CSM_CODEBOOKS
        )
        logger.info("CSM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CSM model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down CSM server")
    sessions.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "active_sessions": len(sessions)
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for TTS sessions."""
    await websocket.accept()
    # Note: TCP_NODELAY would need to be set at uvicorn level
    # For now, relying on small message sizes and immediate sends
    session: Optional[Session] = None
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get('type')
            
            if msg_type == 'init':
                session_id = data.get('session_id', f"session_{time.time()}")
                session = Session(session_id)
                sessions[session_id] = session
                
                # Load reference audio if provided
                if 'ref_audio_base64' in data:
                    ref_audio = base64.b64decode(data['ref_audio_base64'])
                    ref_tensor = ulaw_8k_to_float_24k(ref_audio)
                    session.reference_segments.append(Segment(
                        speaker=data.get('speaker_id', 0),
                        text=data.get('ref_text', ''),
                        audio=ref_tensor
                    ))
                    logger.info(f"Session {session_id}: Loaded reference audio ({len(ref_audio)} bytes)")
                    
                session.speaker_id = data.get('speaker_id', 0)
                logger.info(f"Session {session_id}: Initialized")
                
                await websocket.send_json({"type": "ready", "session_id": session_id})
                
            elif msg_type == 'audio':
                if session:
                    audio_bytes = base64.b64decode(data['data'])
                    session.add_audio_chunk(audio_bytes)
                    
            elif msg_type == 'user_turn_end':
                if session:
                    session.finalize_user_turn(data.get('text', ''))
                    await websocket.send_json({"type": "user_turn_processed"})
                    
            elif msg_type == 'generate':
                if session:
                    generation_id = data.get('generation_id')
                    await session.generate(data['text'], websocket, generation_id=generation_id)
                    
            elif msg_type == 'interrupt':
                if session:
                    session.interrupt()
                    await websocket.send_json({"type": "interrupted"})
                    
            elif msg_type == 'close':
                if session:
                    del sessions[session.session_id]
                    logger.info(f"Session {session.session_id}: Closed")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session and session.session_id in sessions:
            del sessions[session.session_id]
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CSM_PORT)
