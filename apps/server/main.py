import asyncio
import json
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import (
    AutoModelForImageTextToText,
    TextIteratorStreamer,
    GenerationConfig,
)
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
import re
from typing import Optional, Dict, Any
import uvicorn
import time

from dotenv import load_dotenv
from openai import AzureOpenAI

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Import Kokoro TTS library
from kokoro import KPipeline

load_dotenv()

AZURE_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add compatibility for Python < 3.10 where anext is not available
try:
    anext
except NameError:

    async def anext(iterator):
        """Get the next item from an async iterator, or raise StopAsyncIteration."""
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            raise


class ImageManager:
    """Manages image saving and verification"""

    def __init__(self, save_directory="received_images"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        logger.info(f"Image save directory: {self.save_directory.absolute()}")

    def save_image(self, image_data: bytes, client_id: str, prefix: str = "img") -> str:
        """Save image data and return the filename"""
        try:
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"{prefix}_{client_id}_{timestamp}.jpg"
            filepath = self.save_directory / filename

            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_data)

            # Log file info
            file_size = len(image_data)
            logger.info(f"Saved image: {filename} ({file_size:,} bytes)")

            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None

    def verify_image(self, filepath: str) -> dict:
        """Verify saved image and return info"""
        try:
            if not os.path.exists(filepath):
                return {"error": "File not found"}

            # Get file stats
            stat = os.stat(filepath)
            file_size = stat.st_size

            # Try to open with PIL to verify it's a valid image
            with Image.open(filepath) as img:
                info = {
                    "filepath": filepath,
                    "file_size": file_size,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "valid": True,
                }

            logger.info(f"Image verified: {info}")
            return info

        except Exception as e:
            logger.error(f"Error verifying image {filepath}: {e}")
            return {"error": str(e), "valid": False}


class WhisperProcessor:
    """Handles speech-to-text using Whisper model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Using device for Whisper: {self.device}")

        # Load Whisper model
        model_id = "openai/whisper-tiny"
        logger.info(f"Loading {model_id}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes to text (safe length)"""
        try:
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    audio_array,
                    return_timestamps=True,  # 🔥 REQUIRED
                    chunk_length_s=30,
                ),
            )

            text = result["text"].strip()
            self.transcription_count += 1

            logger.info(f"Transcription #{self.transcription_count}: '{text}'")

            if not text or len(text) < 3:
                return "NO_SPEECH"

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


AUDIO_BUFFERS = {}
LAST_AUDIO_TIME = {}
SILENCE_THRESHOLD = 1.2  # seconds


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Kokoro TTS processor...")
        try:
            # Initialize Kokoro TTS pipeline
            self.pipeline = KPipeline(lang_code="a")

            # Set voice
            self.default_voice = "af_sarah"

            logger.info("Kokoro TTS processor initialized successfully")
            # Counter
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None

    async def synthesize_initial_speech_with_timing(self, text):
        """Convert initial text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(f"Synthesizing initial speech for text: '{text}'")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Use the executor to run the TTS pipeline with minimal splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=None,  # No splitting for initial text to process faster
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components as shown in your screenshot
                gs = result.graphemes  # str - the text graphemes
                ps = result.phonemes  # str - the phonemes
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"Initial speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Initial speech synthesis with timing error: {e}")
            return None, []

    async def synthesize_remaining_speech_with_timing(self, text):
        """Convert remaining text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(
                f"Synthesizing chunk speech for text: '{text[:50]}...' if len(text) > 50 else text"
            )

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Determine appropriate split pattern based on text length
            if len(text) < 100:
                split_pattern = None  # No splitting for very short chunks
            else:
                split_pattern = r"[.!?。！？]+"

            # Use the executor to run the TTS pipeline with optimized splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text, voice=self.default_voice, speed=1, split_pattern=split_pattern
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components with NATIVE timing
                gs = result.graphemes  # str
                ps = result.phonemes  # str
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Chunk segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Chunk word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Chunk word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"Chunk speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Chunk speech synthesis with timing error: {e}")
            return None, []


async def collect_remaining_text(streamer, chunk_size=80):
    """Collect remaining text from the streamer in smaller chunks

    Args:
        streamer: The text streamer object
        chunk_size: Maximum characters per chunk before yielding

    Yields:
        Text chunks as they become available
    """
    current_chunk = ""

    if streamer:
        try:
            for chunk in streamer:
                current_chunk += chunk
                logger.info(f"Collecting remaining text chunk: '{chunk}'")

                # Check if we've reached a good breaking point (sentence end)
                if len(current_chunk) >= chunk_size and (
                    current_chunk.endswith(".")
                    or current_chunk.endswith("!")
                    or current_chunk.endswith("?")
                    or "." in current_chunk[-15:]
                ):
                    logger.info(f"Yielding text chunk of length {len(current_chunk)}")
                    yield current_chunk
                    current_chunk = ""

            # Yield any remaining text
            if current_chunk:
                logger.info(f"Yielding final text chunk of length {len(current_chunk)}")
                yield current_chunk

        except asyncio.CancelledError:
            # If there's text collected before cancellation, yield it
            if current_chunk:
                logger.info(
                    f"Yielding partial text chunk before cancellation: {len(current_chunk)} chars"
                )
                yield current_chunk
            raise


INTERVIEW_SYSTEM_PROMPT = """
You are a professional human interviewer conducting a real job interview.

Your responsibilities:
- Lead the interview from start to finish.
- Speak first by greeting the candidate politely.
- Follow a structured interview flow:
  1. Greeting and warm-up
  2. Candidate introduction
  3. Role confirmation
  4. Experience discussion
  5. Technical questions
  6. Behavioral questions
  7. Candidate questions
  8. Professional wrap-up

Behavior rules:
- Always acknowledge the candidate’s response before asking the next question.
- Ask only one clear question at a time.
- Be calm, respectful, and professional.
- Adapt naturally if the candidate hesitates, skips a question, or wants to stop.
- Never pressure or argue with the candidate.
- If the candidate wants to exit, end the interview gracefully.
- If the candidate resumes, continue naturally.

Tone:
- Formal but warm
- Conversational, not robotic
- Encouraging and respectful

Important constraints:
- Do not mention that you are an AI.
- Do not provide personal information about yourself.
- Do not rush the interview.
- Do not ask multiple questions at once.

You are conducting a real interview, not a casual conversation.


"""


class InterviewSession:
    def __init__(self):
        self.history = []


INTERVIEW_SESSIONS = {}


# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # Track current processing tasks for each client
        self.current_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        # Add image manager
        self.image_manager = ImageManager()
        # Track statistics
        self.stats = {
            "audio_segments_received": 0,
            "images_received": 0,
            "audio_with_image_received": 0,
            "last_reset": datetime.now(),
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.current_tasks[client_id] = {"processing": None, "tts": None}
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.current_tasks:
            del self.current_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def cancel_current_tasks(self, client_id: str):
        """Cancel any ongoing processing tasks for a client"""
        if client_id in self.current_tasks:
            tasks = self.current_tasks[client_id]

            # Cancel processing task
            if tasks["processing"] and not tasks["processing"].done():
                logger.info(f"Cancelling processing task for client {client_id}")
                tasks["processing"].cancel()
                try:
                    await tasks["processing"]
                except asyncio.CancelledError:
                    pass

            # Cancel TTS task
            if tasks["tts"] and not tasks["tts"].done():
                logger.info(f"Cancelling TTS task for client {client_id}")
                tasks["tts"].cancel()
                try:
                    await tasks["tts"]
                except asyncio.CancelledError:
                    pass

            # Reset tasks
            self.current_tasks[client_id] = {"processing": None, "tts": None}

    def set_task(self, client_id: str, task_type: str, task: asyncio.Task):
        """Set a task for a client"""
        if client_id in self.current_tasks:
            self.current_tasks[client_id][task_type] = task

    def update_stats(self, event_type: str):
        """Update statistics"""
        if event_type in self.stats:
            self.stats[event_type] += 1

    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = datetime.now() - self.stats["last_reset"]
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "active_connections": len(self.active_connections),
        }


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing models on startup...")
    try:
        # Initialize processors to load models
        whisper_processor = WhisperProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Whisper + SmolVLM2 Voice Assistant",
    description="Real-time voice assistant with speech recognition, image processing, and text-to-speech",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return manager.get_stats()


@app.get("/images")
async def list_saved_images():
    """List all saved images"""
    try:
        images_dir = manager.image_manager.save_directory
        if not images_dir.exists():
            return {"images": [], "message": "No images directory found"}

        images = []
        for image_file in images_dir.glob("*.jpg"):
            stat = image_file.stat()
            images.append(
                {
                    "filename": image_file.name,
                    "path": str(image_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        images.sort(key=lambda x: x["created"], reverse=True)  # Most recent first
        return {"images": images, "count": len(images)}

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return {"error": str(e)}


async def call_interviewer_llm(client_id: str, user_text: str) -> str:
    session = INTERVIEW_SESSIONS.setdefault(client_id, InterviewSession())

    messages = [{"role": "system", "content": INTERVIEW_SYSTEM_PROMPT}]

    for msg in session.history[-4:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_text})

    response = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=0.2,  # professionalism
        max_tokens=180,
    )

    reply = response.choices[0].message.content.strip()

    session.history.append({"role": "user", "content": user_text})
    session.history.append({"role": "assistant", "content": reply})

    return reply


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    whisper_processor = WhisperProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    async def speak_text_immediately(text: str):
        logger.info(f"Speaking immediately: {text}")

        audio, timings = await tts_processor.synthesize_initial_speech_with_timing(text)

        if audio is None or len(audio) == 0:
            logger.warning("TTS produced no audio")
            return

        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        await websocket.send_text(
            json.dumps(
                {
                    "audio": base64_audio,
                    "word_timings": timings,
                    "sample_rate": 24000,
                    "method": "native_kokoro_timing",
                    "modality": "greeting",
                }
            )
        )

        await websocket.send_text(json.dumps({"audio_complete": True}))

    try:
        # 🔹 Confirm connection
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        # 🔹 PROFESSIONAL GREETING (FIRST THING USER HEARS)
        GREETING = (
            "Good day. Thank you for joining me today. "
            "I will be conducting your interview. "
            "To begin, could you please tell me a bit about yourself and your background?"
        )

        await speak_text_immediately(GREETING)

        async def process_audio_segment(audio_data: bytes):
            try:
                await manager.cancel_current_tasks(client_id)

                # Interrupt avatar speech immediately
                await websocket.send_text(json.dumps({"interrupt": True}))

                # 1️⃣ Whisper
                transcribed_text = await whisper_processor.transcribe_audio(audio_data)
                logger.info(f"Transcription result: {transcribed_text}")

                if transcribed_text in ["NO_SPEECH", None]:
                    return

                # 2️⃣ LLM Interviewer
                reply = await call_interviewer_llm(client_id, transcribed_text)
                logger.info(f"Interviewer reply: {reply[:60]}...")

                # 3️⃣ TTS
                audio, timings = (
                    await tts_processor.synthesize_initial_speech_with_timing(reply)
                )

                if audio is None or len(audio) == 0:
                    return

                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

                await websocket.send_text(
                    json.dumps(
                        {
                            "audio": base64_audio,
                            "word_timings": timings,
                            "sample_rate": 24000,
                            "method": "native_kokoro_timing",
                            "modality": "audio_only",
                        }
                    )
                )

                await websocket.send_text(json.dumps({"audio_complete": True}))

            except asyncio.CancelledError:
                logger.info("Audio task cancelled")
                raise
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

        while True:
            message = json.loads(await websocket.receive_text())

            if "audio_segment" in message:
                audio_data = base64.b64decode(message["audio_segment"])

                now = time.time()
                LAST_AUDIO_TIME[client_id] = now

                if client_id not in AUDIO_BUFFERS:
                    AUDIO_BUFFERS[client_id] = bytearray()

                AUDIO_BUFFERS[client_id].extend(audio_data)

                # Schedule silence check
                async def check_silence():
                    await asyncio.sleep(SILENCE_THRESHOLD)
                    last_time = LAST_AUDIO_TIME.get(client_id)

                    if last_time and time.time() - last_time >= SILENCE_THRESHOLD:
                        full_audio = bytes(AUDIO_BUFFERS.pop(client_id))
                        logger.info("User finished speaking — processing full answer")

                        await process_audio_segment(full_audio)

                asyncio.create_task(check_silence())

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)


def main():
    """Main function to start the FastAPI server"""
    logger.info("Starting FastAPI Whisper + SmolVLM2 Voice Assistant server...")

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=60,
        timeout_keep_alive=30,
    )

    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
