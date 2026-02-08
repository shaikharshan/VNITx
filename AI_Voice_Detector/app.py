"""
Voice Detection API - Flask Application (HuggingFace Spaces Version)
Accepts Base64-encoded MP3 audio and returns AI vs Human classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from functools import wraps
import base64
import json
import os
import logging
import shutil
import tempfile
import uuid
import wave
from datetime import datetime
from urllib.parse import parse_qs

# Import the detector
from detector import HybridEnsembleDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Load API key from environment variable (HuggingFace Secrets)
API_KEY = os.environ.get('API_KEY', 'sk_test_123456789')
logger.info(f"API initialized with key: {API_KEY[:10]}...")

def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["1", "true", "yes", "y", "on"]

# Streaming configuration
STREAMING_ENABLED = parse_bool(os.environ.get("ENABLE_STREAMING", "true"))
STREAMING_MAX_BUFFER_SECONDS = int(os.environ.get("STREAMING_MAX_BUFFER_SECONDS", 30))
STREAMING_PARTIAL_INTERVAL_SECONDS = float(os.environ.get("STREAMING_PARTIAL_INTERVAL_SECONDS", 10))
STREAMING_PARTIAL_MODE = os.environ.get("STREAMING_PARTIAL_MODE", "physics").lower()
STREAMING_MAX_CHUNK_BYTES = int(os.environ.get("STREAMING_MAX_CHUNK_BYTES", 2 * 1024 * 1024))
STREAMING_SUPPORTED_FORMATS = {"pcm16", "wav", "mp3"}

# Self-learning / feedback configuration
ENABLE_FEEDBACK_STORAGE = parse_bool(os.environ.get("ENABLE_FEEDBACK_STORAGE", "true"))
FEEDBACK_STORAGE_DIR = os.environ.get("FEEDBACK_STORAGE_DIR", "data/feedback")
FEEDBACK_MAX_BYTES = int(os.environ.get("FEEDBACK_MAX_BYTES", 15 * 1024 * 1024))
CALIBRATION_PATH = os.environ.get("CALIBRATION_PATH", "data/calibration.json")
CALIBRATION_HISTORY_DIR = os.environ.get("CALIBRATION_HISTORY_DIR", "data/calibration_history")
CALIBRATION_HISTORY_MAX = int(os.environ.get("CALIBRATION_HISTORY_MAX", 50))

# Initialize the detector globally (load models once at startup)
logger.info("Loading AI detection models...")
detector = None
SKIP_MODEL_LOAD = parse_bool(os.environ.get("SKIP_MODEL_LOAD", "false"))

def init_detector():
    """Initialize the detector with models"""
    global detector
    try:
        detector = HybridEnsembleDetector(
            deepfake_model_path=r"D:\hackathons\GUVI_HCL\AI_Voice_Detector\wav2vec2-deepfake-voice-detector",
            whisper_model_path="openai/whisper-base",
            physics_weight=0.4,
            dl_weight=0.6,
            use_local_deepfake_model=True,
            use_local_whisper_model=False,
            calibration_path=CALIBRATION_PATH,
            max_audio_duration=30
        )
        logger.info("✅ Detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {str(e)}")
        return False

# Initialize detector at startup
if SKIP_MODEL_LOAD:
    logger.info("⚠️ Skipping detector initialization (SKIP_MODEL_LOAD=true)")
elif not init_detector():
    logger.warning("⚠️ API starting without detector - models will be loaded on first request")


# ==========================================================
# AUTHENTICATION DECORATOR
# ==========================================================
def require_api_key(f):
    """Decorator to validate API key from request headers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers
        provided_key = request.headers.get('x-api-key')
        
        if not provided_key:
            logger.warning(f"Request without API key from {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Missing API key. Please provide 'x-api-key' in request headers."
            }), 401
        
        if provided_key != API_KEY:
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Invalid API key"
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


def get_ws_api_key(environ):
    if not environ:
        return None

    key = environ.get("HTTP_X_API_KEY")
    if key:
        return key

    auth = environ.get("HTTP_AUTHORIZATION")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]

    query_params = parse_qs(environ.get("QUERY_STRING", ""))
    if "api_key" in query_params:
        return query_params["api_key"][0]

    return None


def normalize_label(label):
    if label is None:
        return None
    label_value = str(label).strip().upper()
    if label_value in ["AI_GENERATED", "AI", "FAKE", "SYNTHETIC"]:
        return "AI_GENERATED"
    if label_value in ["HUMAN", "REAL"]:
        return "HUMAN"
    return None


def decode_audio_base64(audio_base64):
    detected_format = None
    if isinstance(audio_base64, str) and audio_base64.startswith("data:"):
        header, audio_base64 = audio_base64.split(",", 1)
        header_lower = header.lower()
        if "audio/wav" in header_lower or "audio/x-wav" in header_lower:
            detected_format = "wav"
        elif "audio/mpeg" in header_lower or "audio/mp3" in header_lower:
            detected_format = "mp3"
    audio_bytes = base64.b64decode(audio_base64)
    return audio_bytes, detected_format


def write_bytes_to_temp_file(data, suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data)
    temp_file.close()
    return temp_file.name


def write_pcm16_to_wav_file(pcm_bytes, sample_rate, channels):
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:len(pcm_bytes) - 1]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return temp_path


def format_detection_payload(result, requested_language=None):
    if result.get("status") != "success":
        return {
            "status": "error",
            "message": result.get("error") or result.get("message") or "Unknown error"
        }

    payload = {
        "status": "success",
        "classification": result.get("classification"),
        "confidenceScore": result.get("confidenceScore"),
        "explanation": result.get("explanation"),
        "detectedLanguage": result.get("language", "Unknown"),
        "analysisMode": result.get("analysisMode", "full")
    }

    if requested_language:
        payload["requestedLanguage"] = requested_language

    return payload


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def build_calibration_version_id():
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def calibration_history_files():
    if not os.path.isdir(CALIBRATION_HISTORY_DIR):
        return []

    files = []
    for name in os.listdir(CALIBRATION_HISTORY_DIR):
        if name.startswith("calibration_") and name.endswith(".json"):
            if name.endswith(".meta.json"):
                continue
            files.append(os.path.join(CALIBRATION_HISTORY_DIR, name))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def archive_calibration(reason=None):
    if not os.path.exists(CALIBRATION_PATH):
        return None

    ensure_dir(CALIBRATION_HISTORY_DIR)
    version_id = build_calibration_version_id()
    filename = f"calibration_{version_id}.json"
    dest_path = os.path.join(CALIBRATION_HISTORY_DIR, filename)
    shutil.copy2(CALIBRATION_PATH, dest_path)

    meta = {
        "versionId": version_id,
        "source": CALIBRATION_PATH,
        "archivedAt": datetime.utcnow().isoformat() + "Z",
        "reason": reason or "manual"
    }
    meta_path = os.path.join(CALIBRATION_HISTORY_DIR, f"calibration_{version_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if CALIBRATION_HISTORY_MAX > 0:
        history = calibration_history_files()
        for path in history[CALIBRATION_HISTORY_MAX:]:
            try:
                os.unlink(path)
            except Exception:
                pass
            meta_path = path.replace(".json", ".meta.json")
            if os.path.exists(meta_path):
                try:
                    os.unlink(meta_path)
                except Exception:
                    pass

    return {
        "versionId": version_id,
        "path": dest_path
    }


def list_calibration_history():
    entries = []
    for path in calibration_history_files():
        name = os.path.basename(path)
        version_id = name.replace("calibration_", "").replace(".json", "")
        meta_path = path.replace(".json", ".meta.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except Exception:
                meta = {}
        entries.append({
            "versionId": version_id,
            "path": path,
            "archivedAt": meta.get("archivedAt"),
            "reason": meta.get("reason")
        })
    return entries


def resolve_history_path(version_id):
    if not version_id:
        return None
    filename = f"calibration_{version_id}.json"
    return os.path.join(CALIBRATION_HISTORY_DIR, filename)


class StreamSession:
    def __init__(
        self,
        audio_format,
        sample_rate,
        channels,
        max_seconds,
        enable_partial,
        partial_interval_seconds,
        partial_mode
    ):
        self.session_id = str(uuid.uuid4())
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_seconds = max_seconds
        self.enable_partial = enable_partial
        self.partial_interval_seconds = partial_interval_seconds
        self.partial_mode = partial_mode
        self.buffer = bytearray()
        self.total_bytes_received = 0
        self.total_seconds_received = 0.0
        self.last_partial_seconds = 0.0

    def add_chunk(self, chunk_bytes):
        self.total_bytes_received += len(chunk_bytes)
        self.buffer.extend(chunk_bytes)

        if self.audio_format == "pcm16":
            bytes_per_second = self.sample_rate * self.channels * 2
            if bytes_per_second > 0:
                self.total_seconds_received = self.total_bytes_received / bytes_per_second
                max_bytes = int(self.max_seconds * bytes_per_second)
                if max_bytes > 0 and len(self.buffer) > max_bytes:
                    overflow = len(self.buffer) - max_bytes
                    del self.buffer[:overflow]

        return self.current_buffer_seconds()

    def current_buffer_seconds(self):
        if self.audio_format != "pcm16":
            return None
        bytes_per_second = self.sample_rate * self.channels * 2
        if bytes_per_second <= 0:
            return None
        return len(self.buffer) / bytes_per_second

    def should_run_partial(self):
        if not self.enable_partial:
            return False
        if self.audio_format != "pcm16":
            return False
        if self.partial_interval_seconds <= 0:
            return False
        if (self.total_seconds_received - self.last_partial_seconds) >= self.partial_interval_seconds:
            self.last_partial_seconds = self.total_seconds_received
            return True
        return False

    def write_temp_audio_file(self):
        if self.audio_format == "pcm16":
            return write_pcm16_to_wav_file(self.buffer, self.sample_rate, self.channels), "wav"

        suffix = ".mp3" if self.audio_format == "mp3" else ".wav"
        return write_bytes_to_temp_file(self.buffer, suffix), self.audio_format


# ==========================================================
# ROOT ENDPOINT (HuggingFace Spaces Homepage)
# ==========================================================
@app.route('/', methods=['GET'])
def home():
    """Root endpoint - API information"""
    return jsonify({
        "name": "Voice Detection API",
        "version": "1.0.0",
        "description": "AI-powered voice detection system for identifying AI-generated vs human voices",
        "endpoints": {
            "health": "/health",
            "detection": "/api/voice-detection",
            "streaming": "/ws/voice-stream",
            "feedback": "/api/feedback",
            "reload_calibration": "/api/reload-calibration",
            "backup_calibration": "/api/backup-calibration",
            "rollback_calibration": "/api/rollback-calibration",
            "calibration_history": "/api/calibration-history"
        },
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "authentication": "Required - use 'x-api-key' header",
        "documentation": "See README for full API documentation"
    }), 200


# ==========================================================
# HEALTH CHECK ENDPOINT
# ==========================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (no authentication required)"""
    return jsonify({
        "status": "healthy",
        "service": "Voice Detection API",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": detector is not None,
        "calibration_loaded": bool(detector and detector.calibrator and detector.calibrator.ready),
        "streaming_enabled": STREAMING_ENABLED,
        "platform": "HuggingFace Spaces"
    }), 200


# ==========================================================
# MAIN VOICE DETECTION ENDPOINT
# ==========================================================
@app.route('/api/voice-detection', methods=['POST'])
@require_api_key
def voice_detection():
    """
    Main voice detection endpoint
    
    Expected JSON Body:
    {
        "language": "Tamil" | "English" | "Hindi" | "Malayalam" | "Telugu",
        "audioFormat": "mp3",
        "audioBase64": "base64_encoded_audio_string"
    }
    
    Returns:
    {
        "status": "success",
        "language": "Tamil",
        "classification": "AI_GENERATED" | "HUMAN",
        "confidenceScore": 0.0-1.0,
        "explanation": "..."
    }
    """
    global detector
    
    try:
        # Validate Content-Type
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['language', 'audioFormat', 'audioBase64']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Validate language
        supported_languages = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
        if data['language'] not in supported_languages:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(supported_languages)}"
            }), 400
        
        # Validate audio format
        if data['audioFormat'].lower() != 'mp3':
            return jsonify({
                "status": "error",
                "message": "Only MP3 audio format is supported"
            }), 400
        
        # Validate base64 string
        audio_base64 = data['audioBase64']
        if not audio_base64 or len(audio_base64) < 100:
            return jsonify({
                "status": "error",
                "message": "Invalid or empty audio data"
            }), 400
        
        # Initialize detector if not already loaded
        if detector is None:
            logger.info("Lazy loading detector on first request...")
            if not init_detector():
                return jsonify({
                    "status": "error",
                    "message": "Failed to load AI detection models. Please try again later."
                }), 503
        
        # Log request
        logger.info(f"Processing voice detection request for language: {data['language']}")
        
        # Analyze audio
        result = detector.analyze(
            audio_base64,
            input_type="base64",
            audio_format=data['audioFormat']
        )
        
        # Check if analysis was successful
        if result['status'] != 'success':
            error_msg = result.get('error', 'Unknown error during analysis')
            logger.error(f"Analysis failed: {error_msg}")
            return jsonify({
                "status": "error",
                "message": f"Audio analysis failed: {error_msg}"
            }), 500
        
        # Prepare response (API compliant format - NO DEBUG INFO in production)
        response = {
            "status": "success",
            "language": data['language'],  # Use requested language from input
            "classification": result['classification'],
            "confidenceScore": result['confidenceScore'],
            "explanation": result['explanation']
        }
        
        logger.info(f"✅ Analysis complete: {result['classification']} (confidence: {result['confidenceScore']})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in voice_detection: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error occurred during processing"
        }), 500


# ==========================================================
# FEEDBACK / SELF-LEARNING ENDPOINT
# ==========================================================
@app.route('/api/feedback', methods=['POST'])
@require_api_key
def feedback():
    """
    Collect labeled audio samples for periodic self-learning.

    Expected JSON Body:
    {
        "label": "AI_GENERATED" | "HUMAN",
        "audioFormat": "mp3" | "wav",
        "audioBase64": "base64_encoded_audio_string",
        "runDetection": false,
        "metadata": { ... }
    }
    """
    if not ENABLE_FEEDBACK_STORAGE:
        return jsonify({
            "status": "error",
            "message": "Feedback storage is disabled"
        }), 403

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type must be application/json"
        }), 400

    data = request.get_json()
    label = normalize_label(data.get("label"))
    if not label:
        return jsonify({
            "status": "error",
            "message": "Invalid label. Use AI_GENERATED or HUMAN."
        }), 400

    audio_format = str(data.get("audioFormat", "mp3")).lower()
    if audio_format not in ["mp3", "wav"]:
        return jsonify({
            "status": "error",
            "message": "audioFormat must be 'mp3' or 'wav'"
        }), 400

    audio_base64 = data.get("audioBase64")
    if not audio_base64 or len(audio_base64) < 100:
        return jsonify({
            "status": "error",
            "message": "Invalid or empty audio data"
        }), 400

    try:
        audio_bytes, detected_format = decode_audio_base64(audio_base64)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to decode audio: {str(e)}"
        }), 400

    if detected_format:
        audio_format = detected_format

    if len(audio_bytes) > FEEDBACK_MAX_BYTES:
        return jsonify({
            "status": "error",
            "message": "Audio payload exceeds maximum size"
        }), 413

    now = datetime.utcnow()
    date_dir = now.strftime("%Y%m%d")
    label_dir = os.path.join(FEEDBACK_STORAGE_DIR, label, date_dir)
    os.makedirs(label_dir, exist_ok=True)

    sample_id = str(uuid.uuid4())
    extension = ".mp3" if audio_format == "mp3" else ".wav"
    file_path = os.path.join(label_dir, f"{sample_id}{extension}")

    with open(file_path, "wb") as handle:
        handle.write(audio_bytes)

    metadata = {
        "id": sample_id,
        "label": label,
        "audio_format": audio_format,
        "created_at": now.isoformat() + "Z",
        "bytes": len(audio_bytes),
        "path": file_path,
        "client_metadata": data.get("metadata", {})
    }

    if parse_bool(data.get("runDetection", False)):
        global detector
        if detector is None:
            logger.info("Lazy loading detector for feedback scoring...")
            if not init_detector():
                return jsonify({
                    "status": "error",
                    "message": "Failed to load AI detection models for scoring"
                }), 503

        scores = detector.extract_scores(file_path, input_type="file")
        if scores.get("status") == "success":
            metadata["physics_score"] = scores.get("physics_score")
            metadata["dl_score"] = scores.get("dl_score")
            metadata["dl_label"] = scores.get("dl_label")
            metadata["audio_duration"] = scores.get("audio_duration")
            metadata["was_truncated"] = scores.get("was_truncated")

    meta_path = os.path.join(label_dir, f"{sample_id}.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    index_path = os.path.join(FEEDBACK_STORAGE_DIR, "index.jsonl")
    with open(index_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata) + "\n")

    return jsonify({
        "status": "success",
        "id": sample_id,
        "label": label,
        "audioFormat": audio_format,
        "stored": True
    }), 200


# ==========================================================
# CALIBRATION RELOAD ENDPOINT
# ==========================================================
@app.route('/api/reload-calibration', methods=['POST'])
@require_api_key
def reload_calibration():
    global detector

    if detector is None:
        logger.info("Lazy loading detector for calibration reload...")
        if not init_detector():
            return jsonify({
                "status": "error",
                "message": "Failed to load AI detection models"
            }), 503

    loaded = detector.reload_calibration(CALIBRATION_PATH)
    if not loaded:
        return jsonify({
            "status": "error",
            "message": "Calibration file not found or invalid"
        }), 404

    return jsonify({
        "status": "success",
        "calibrationPath": detector.calibrator.calibration_path
    }), 200


@app.route('/api/backup-calibration', methods=['POST'])
@require_api_key
def backup_calibration():
    payload = request.get_json(silent=True) or {}
    reason = payload.get("reason")

    if not os.path.exists(CALIBRATION_PATH):
        return jsonify({
            "status": "error",
            "message": "Calibration file not found"
        }), 404

    backup = archive_calibration(reason=reason or "api_backup")
    if not backup:
        return jsonify({
            "status": "error",
            "message": "Failed to archive calibration"
        }), 500

    return jsonify({
        "status": "success",
        "versionId": backup["versionId"],
        "path": backup["path"]
    }), 200


@app.route('/api/calibration-history', methods=['GET'])
@require_api_key
def calibration_history():
    history = list_calibration_history()
    return jsonify({
        "status": "success",
        "history": history
    }), 200


@app.route('/api/rollback-calibration', methods=['POST'])
@require_api_key
def rollback_calibration():
    payload = request.get_json(silent=True) or {}
    version_id = payload.get("versionId")

    if not version_id:
        return jsonify({
            "status": "error",
            "message": "Missing versionId"
        }), 400

    source_path = resolve_history_path(version_id)
    if not source_path or not os.path.exists(source_path):
        return jsonify({
            "status": "error",
            "message": "Calibration version not found"
        }), 404

    ensure_dir(os.path.dirname(CALIBRATION_PATH))
    shutil.copy2(source_path, CALIBRATION_PATH)

    global detector
    if detector is None:
        logger.info("Lazy loading detector for rollback...")
        if not init_detector():
            return jsonify({
                "status": "error",
                "message": "Failed to load AI detection models"
            }), 503

    loaded = detector.reload_calibration(CALIBRATION_PATH)
    if not loaded:
        return jsonify({
            "status": "error",
            "message": "Failed to load calibration after rollback"
        }), 500

    return jsonify({
        "status": "success",
        "versionId": version_id,
        "calibrationPath": CALIBRATION_PATH
    }), 200


# ==========================================================
# REALTIME STREAMING ENDPOINT (WEBSOCKET)
# ==========================================================
@sock.route('/ws/voice-stream')
def voice_stream(ws):
    if not STREAMING_ENABLED:
        ws.send(json.dumps({
            "type": "error",
            "message": "Streaming is disabled"
        }))
        return

    api_key = get_ws_api_key(ws.environ)
    if api_key != API_KEY:
        ws.send(json.dumps({
            "type": "error",
            "message": "Invalid API key"
        }))
        return

    session = None
    requested_language = None

    while True:
        message = ws.receive()
        if message is None:
            break

        try:
            payload = json.loads(message)
        except Exception:
            ws.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
            continue

        msg_type = payload.get("type")

        if msg_type == "start":
            if session is not None:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Stream already started"
                }))
                continue

            audio_format = str(payload.get("audioFormat", "pcm16")).lower()
            if audio_format in ["pcm_s16le", "s16le", "pcm16le"]:
                audio_format = "pcm16"
            if audio_format not in STREAMING_SUPPORTED_FORMATS:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Unsupported audioFormat for streaming"
                }))
                continue

            sample_rate = int(payload.get("sampleRate", 16000))
            channels = int(payload.get("channels", 1))
            if audio_format == "pcm16":
                if sample_rate <= 0 or channels <= 0:
                    ws.send(json.dumps({
                        "type": "error",
                        "message": "sampleRate and channels must be positive for pcm16"
                    }))
                    continue
                if channels not in [1, 2]:
                    ws.send(json.dumps({
                        "type": "error",
                        "message": "channels must be 1 or 2 for pcm16"
                    }))
                    continue
            requested_language = payload.get("language")
            enable_partial = parse_bool(payload.get("enablePartial", True))
            partial_interval = float(payload.get("partialIntervalSec", STREAMING_PARTIAL_INTERVAL_SECONDS))
            max_seconds = int(payload.get("maxSeconds", STREAMING_MAX_BUFFER_SECONDS))
            partial_mode = str(payload.get("partialMode", STREAMING_PARTIAL_MODE)).lower()
            if partial_mode not in ["full", "physics", "dl"]:
                partial_mode = "physics"

            session = StreamSession(
                audio_format=audio_format,
                sample_rate=sample_rate,
                channels=channels,
                max_seconds=max_seconds,
                enable_partial=enable_partial,
                partial_interval_seconds=partial_interval,
                partial_mode=partial_mode
            )

            ws.send(json.dumps({
                "type": "ack",
                "status": "ready",
                "sessionId": session.session_id,
                "streaming": {
                    "audioFormat": audio_format,
                    "sampleRate": sample_rate,
                    "channels": channels,
                    "maxSeconds": max_seconds,
                    "partialIntervalSec": partial_interval,
                    "partialMode": partial_mode,
                    "enablePartial": enable_partial
                }
            }))
            continue

        if msg_type == "ping":
            ws.send(json.dumps({"type": "pong"}))
            continue

        if msg_type not in ["audio_chunk", "stop"]:
            ws.send(json.dumps({
                "type": "error",
                "message": "Unsupported message type"
            }))
            continue

        if session is None:
            ws.send(json.dumps({
                "type": "error",
                "message": "Stream not started"
            }))
            continue

        finalize_only = False
        if msg_type == "stop":
            payload["final"] = True
            finalize_only = True

        chunk_b64 = payload.get("audioChunkBase64")
        chunk_bytes = None
        if not chunk_b64:
            if not finalize_only:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Missing audioChunkBase64"
                }))
                continue
        else:
            try:
                chunk_bytes = base64.b64decode(chunk_b64)
            except Exception:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Invalid base64 audio chunk"
                }))
                continue

            if len(chunk_bytes) > STREAMING_MAX_CHUNK_BYTES:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Audio chunk exceeds maximum size"
                }))
                continue

            buffer_seconds = session.add_chunk(chunk_bytes)
            ws.send(json.dumps({
                "type": "progress",
                "receivedBytes": session.total_bytes_received,
                "bufferBytes": len(session.buffer),
                "bufferSeconds": buffer_seconds
            }))

        if session.should_run_partial():
            if detector is None:
                logger.info("Lazy loading detector for streaming...")
                if not init_detector():
                    ws.send(json.dumps({
                        "type": "error",
                        "message": "Failed to load AI detection models"
                    }))
                    break

            temp_path = None
            try:
                temp_path, file_format = session.write_temp_audio_file()
                result = detector.analyze(
                    temp_path,
                    input_type="file",
                    audio_format=file_format,
                    analysis_mode=session.partial_mode
                )
                ws.send(json.dumps({
                    "type": "partial_result",
                    "result": format_detection_payload(result, requested_language=requested_language)
                }))
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

        if parse_bool(payload.get("final", False)):
            if not session.buffer:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "No audio received"
                }))
                break

            if detector is None:
                logger.info("Lazy loading detector for streaming...")
                if not init_detector():
                    ws.send(json.dumps({
                        "type": "error",
                        "message": "Failed to load AI detection models"
                    }))
                    break

            temp_path = None
            try:
                temp_path, file_format = session.write_temp_audio_file()
                result = detector.analyze(
                    temp_path,
                    input_type="file",
                    audio_format=file_format,
                    analysis_mode="full"
                )
                ws.send(json.dumps({
                    "type": "final_result",
                    "result": format_detection_payload(result, requested_language=requested_language)
                }))
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            break


# ==========================================================
# ERROR HANDLERS
# ==========================================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        "status": "error",
        "message": "Method not allowed for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


# ==========================================================
# RUN APPLICATION
# ==========================================================
if __name__ == '__main__':
    # HuggingFace Spaces uses port 7860
    port = int(os.environ.get('PORT', 7860))
    
    # Run the app
    logger.info(f"🚀 Starting Voice Detection API on port {port}")
    logger.info(f"📍 Endpoint: http://0.0.0.0:{port}/api/voice-detection")
    logger.info(f"🔑 API Key: {API_KEY}")
    logger.info(f"🌐 Platform: HuggingFace Spaces")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Always False in production
    )