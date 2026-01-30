# CSM Streaming Server

WebSocket server for CSM (Conversational Speech Model) TTS with audio context.

## Features

- Continuous audio streaming from client (zero upload latency)
- Conversation context management
- Voice cloning via reference audio
- Streaming audio output (ulaw 8kHz for SIP)

## Protocol

### Client → Server

```json
// Initialize session
{"type": "init", "session_id": "xxx", "ref_audio_base64": "...", "ref_text": "...", "speaker_id": 0}

// Audio chunk (sent continuously)
{"type": "audio", "data": "<base64 ulaw 8kHz>"}

// User finished speaking
{"type": "user_turn_end", "text": "transcribed text"}

// Generate AI response
{"type": "generate", "text": "AI response text"}

// Interrupt generation
{"type": "interrupt"}

// Close session
{"type": "close"}
```

### Server → Client

```json
// Session ready
{"type": "ready", "session_id": "xxx"}

// Audio chunk (base64 ulaw 8kHz; payload size is a multiple of 160 bytes; default batching is up to ~100ms)
{"type": "audio", "data": "<base64 ulaw 8kHz>", "frame_bytes": 160}

// Generation complete
{"type": "done"}

// Error
{"type": "error", "message": "..."}
```

## Environment Variables

- `CSM_MODEL_PATH`: Path to CSM model (default: /files/csm-streaming)
- `CSM_DEVICE`: cuda or cpu (default: cuda)
- `CSM_PORT`: Server port (default: 8765)
- `CSM_CODEBOOKS`: Number of audio codebooks (default: 32)
- `CSM_PATH`: Path to csm-streaming code (default: /files/csm-streaming)
- `CSM_WS_BATCH_MS`: Max time to hold an outgoing audio batch before sending (default: 100)
- `CSM_WS_BATCH_FRAMES`: Max number of 20ms frames to batch into one websocket message (default derived from CSM_WS_BATCH_MS; typically 5)
- `CSM_WS_FRAME_BYTES`: Frame size in bytes (default: 160)

## Running

```bash
# Local
python server.py

# Docker
docker build -t csm-server .
docker run -p 8765:8765 --gpus all -v /path/to/model:/models/csm csm-server
```
