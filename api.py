import io
import logging
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

# Silence noisy dependency logs before imports touch them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dataset import SAMPLE_RATE
from inference import PronunciationScorer


# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("pronunciation_api")

for _lib in ("httpx", "httpcore", "huggingface_hub", "transformers"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# Config
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))
DEVICE = os.getenv("DEVICE") or None
MAX_FILE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MIN_AUDIO_SEC = float(os.getenv("MIN_AUDIO_SEC", "0.3"))
THRESHOLD = 60.0

_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [
    origin.strip() for origin in _cors_origins.split(",") if origin.strip()
]

# Model loaded once at startup.
_scorer: Optional[PronunciationScorer] = None
_score_lock = threading.Lock()


def _check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _scorer

    if not _check_ffmpeg():
        logger.warning(
            "ffmpeg not found in PATH. MP3, WebM, and other compressed formats may fail. "
            "Install ffmpeg and ensure it is on PATH: https://ffmpeg.org/download.html"
        )
    else:
        logger.info("ffmpeg found: %s", shutil.which("ffmpeg"))

    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.is_file():
        logger.warning(
            "Checkpoint not found: %s. The pretrained encoder will use cosine fallback scoring.",
            checkpoint,
        )

    logger.info("Loading model from %s ...", checkpoint)
    try:
        _scorer = PronunciationScorer(
            checkpoint_path=str(checkpoint),
            embed_dim=EMBED_DIM,
            device=DEVICE,
        )
        _scorer.model.eval()
        logger.info("Model ready  device=%s", _scorer.device)
    except Exception:
        logger.exception("FATAL: model failed to load; /score will return 503")
        _scorer = None

    try:
        yield
    finally:
        _scorer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# App
app = FastAPI(
    title="Pronunciation Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=not (CORS_ORIGINS == ["*"]),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# Response schema
class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Pronunciation score 0-100")
    label: str = Field(..., description="Good | Try Again")
    similarity: float = Field(..., description="Cosine similarity of embeddings")
    ref_duration: float = Field(..., description="Reference audio length in seconds")
    child_duration: float = Field(..., description="Child audio length in seconds")
    
    
# Audio decoding
def _safe_filename(upload: UploadFile) -> str:
    return upload.filename or "uploaded_audio"


def _file_suffix(name: str) -> str:
    _, ext = os.path.splitext(name.lower())
    return ext if ext else ".audio"


def _read_upload(upload: UploadFile) -> bytes:
    name = _safe_filename(upload)
    try:
        upload.file.seek(0)
        raw = upload.file.read(MAX_FILE_BYTES + 1)
    except Exception as exc:
        raise HTTPException(400, f"Could not read '{name}': {exc}") from exc

    if not raw:
        raise HTTPException(400, f"'{name}' is empty.")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            413,
            f"'{name}' exceeds {MAX_FILE_BYTES // 1_048_576} MB limit.",
        )
    return raw


def _decode_bytes_to_numpy(raw: bytes, name: str) -> Tuple[np.ndarray, int]:
    """
    Decode raw audio bytes to a mono float32 numpy array using a fallback chain:
      1. soundfile  — fast; handles wav/flac/ogg/aiff (NOT mp3/webm)
      2. librosa    — handles mp3/wav/flac via audioread+ffmpeg
      3. pydub      — handles mp3/wav/webm/ogg via ffmpeg
      4. torchaudio — handles wav/flac/mp3 (with torchaudio-ffmpeg backend)

    Returns (waveform_1d_float32, sample_rate_int).
    """
    suffix = _file_suffix(name)
    errors: list = []

    # --- Strategy 1: soundfile (no ffmpeg needed, but no mp3/webm) ---
    try:
        audio_np, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio_np.ndim == 2:
            # soundfile shape: [samples, channels] → average to mono
            audio_np = audio_np.mean(axis=1)
        logger.debug("soundfile decoded '%s'  sr=%d  samples=%d", name, sr, audio_np.shape[0])
        return audio_np.astype(np.float32), int(sr)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")
        logger.debug("soundfile failed for '%s': %s", name, exc)

    # Remaining strategies write to a temp file so libraries can open by path.
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # --- Strategy 2: librosa (mp3 / wav / flac via audioread + ffmpeg) ---
        try:
            import librosa  # type: ignore
            audio_np, sr = librosa.load(tmp_path, sr=None, mono=True)
            logger.debug("librosa decoded '%s'  sr=%d  samples=%d", name, sr, audio_np.shape[0])
            return audio_np.astype(np.float32), int(sr)
        except ImportError:
            errors.append("librosa: not installed")
        except Exception as exc:
            errors.append(f"librosa: {exc}")
            logger.debug("librosa failed for '%s': %s", name, exc)

        # --- Strategy 3: pydub + ffmpeg (mp3 / wav / webm / ogg / …) ---
        try:
            from pydub import AudioSegment  # type: ignore
            seg = AudioSegment.from_file(tmp_path)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)
            max_val = float(2 ** (seg.sample_width * 8 - 1))
            audio_np = samples / max_val
            logger.debug("pydub decoded '%s'  sr=%d  samples=%d", name, sr, audio_np.shape[0])
            return audio_np.astype(np.float32), int(sr)
        except ImportError:
            errors.append("pydub: not installed")
        except Exception as exc:
            errors.append(f"pydub: {exc}")
            logger.debug("pydub failed for '%s': %s", name, exc)

        # --- Strategy 4: torchaudio (wav / flac / mp3 with torchaudio-ffmpeg) ---
        try:
            import torchaudio  # type: ignore
            waveform, sr = torchaudio.load(tmp_path)
            # torchaudio shape: [channels, samples] → average to mono
            audio_np = waveform.mean(dim=0).numpy().astype(np.float32)
            logger.debug("torchaudio decoded '%s'  sr=%d  samples=%d", name, sr, audio_np.shape[0])
            return audio_np, int(sr)
        except ImportError:
            errors.append("torchaudio: not installed")
        except Exception as exc:
            errors.append(f"torchaudio: {exc}")
            logger.debug("torchaudio failed for '%s': %s", name, exc)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    raise ValueError("; ".join(errors))


def _decode_upload(upload: UploadFile) -> Tuple[torch.Tensor, float]:
    """
    Decode an uploaded audio file and return (waveform, duration_seconds).

    The returned waveform is produced by PronunciationScorer._preprocess(), so
    API preprocessing stays consistent with the inference pipeline.
    """
    if _scorer is None:
        raise HTTPException(503, "Model not loaded.")

    name = _safe_filename(upload)
    raw = _read_upload(upload)
    content_type = getattr(upload, "content_type", "") or ""

    logger.info(
        "Decoding '%s'  content_type=%s  size=%d bytes",
        name, content_type or "unknown", len(raw),
    )

    try:
        audio_np, sr = _decode_bytes_to_numpy(raw, name)
    except Exception as exc:
        logger.error(
            "Cannot decode '%s'  content_type=%s  size=%d bytes  error=%s",
            name, content_type or "unknown", len(raw), exc,
        )
        if not _check_ffmpeg():
            logger.error(
                "ffmpeg is not installed or not on PATH. "
                "MP3 and WebM files require ffmpeg. "
                "Install from https://ffmpeg.org/download.html"
            )
        raise HTTPException(
            400,
            f"Cannot decode audio from '{name}' "
            f"(content_type={content_type or 'unknown'}, size={len(raw)} bytes). "
            f"Tried soundfile, librosa, pydub, torchaudio. "
            f"Details: {exc}",
        ) from exc

    if not isinstance(sr, int) or sr <= 0:
        raise HTTPException(422, f"'{name}' has an invalid sample rate ({sr}).")
    if audio_np.size == 0:
        raise HTTPException(422, f"'{name}' contains no audio samples.")
    if audio_np.ndim != 1:
        raise HTTPException(422, f"'{name}' has unsupported audio shape {audio_np.shape}.")
    if not np.isfinite(audio_np).all():
        raise HTTPException(422, f"'{name}' contains invalid (NaN/Inf) audio samples.")

    original_peak = float(np.max(np.abs(audio_np)))
    if original_peak < 1e-4:
        raise HTTPException(422, f"'{name}' is silent or near-silent; cannot score.")

    try:
        waveform = _scorer._preprocess(audio_np, source_sr=sr)
    except Exception as exc:
        logger.exception("Preprocessing failed for %s", name)
        raise HTTPException(500, f"Preprocessing failed for '{name}'.") from exc

    if waveform.ndim != 1:
        raise HTTPException(500, f"Preprocessing produced invalid shape for '{name}'.")
    if not torch.isfinite(waveform).all():
        raise HTTPException(422, f"'{name}' produced invalid audio samples after preprocessing.")

    duration = waveform.shape[0] / SAMPLE_RATE
    if duration < MIN_AUDIO_SEC:
        raise HTTPException(
            422,
            f"'{name}' is shorter than {MIN_AUDIO_SEC:.1f}s after resampling; too short to score.",
        )
    if waveform.abs().max().item() < 1e-4:
        raise HTTPException(422, f"'{name}' is silent or near-silent after preprocessing; cannot score.")

    return waveform, duration


def _validate_embedding(name: str, emb: torch.Tensor) -> None:
    if emb.ndim != 2 or emb.shape[0] != 1 or emb.shape[1] != EMBED_DIM:
        raise HTTPException(500, f"{name} embedding has invalid shape.")
    if not torch.isfinite(emb).all():
        raise HTTPException(500, f"{name} embedding contains invalid values.")


# Routes
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok" if _scorer is not None else "unavailable",
        "model_loaded": _scorer is not None,
        "checkpoint": CHECKPOINT_PATH,
        "device": str(_scorer.device) if _scorer else "none",
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_pronunciation(
    reference: UploadFile = File(..., description="Reference pronunciation audio"),
    child: UploadFile = File(..., description="Child pronunciation audio to score"),
):
    if _scorer is None:
        raise HTTPException(503, "Model not loaded.")

    ref_wav, ref_dur = _decode_upload(reference)
    child_wav, child_dur = _decode_upload(child)

    try:
        with _score_lock, torch.inference_mode():
            _scorer.model.eval()

            ref_emb = _scorer._encode(ref_wav)
            child_emb = _scorer._encode(child_wav)
            _validate_embedding("Reference", ref_emb)
            _validate_embedding("Child", child_emb)
            ref_norm = float(ref_emb.norm(dim=-1).item())
            child_norm = float(child_emb.norm(dim=-1).item())

            prob = _scorer.model.score_from_embeddings(ref_emb, child_emb)
            if prob.numel() != 1 or not torch.isfinite(prob).all():
                raise RuntimeError("model returned an invalid score")
            prob = prob.clamp(0.0, 1.0)

            score = prob.mul(100.0).clamp(0.0, 100.0)
            calibrated_score_value = float(score.item())

            similarity = F.cosine_similarity(ref_emb, child_emb)
            similarity_value = float(similarity.clamp(-1.0, 1.0).item())
    except HTTPException as e:
        raise
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(500, "Scoring failed.") from exc

    # Simple binary scoring
    confidence = min(1.0, max(0.25, (similarity_value - 0.35) / 0.37))
    final_score = max(0.0, min(100.0, calibrated_score_value * confidence))

    if final_score >= THRESHOLD:
        label = "Good"
    else:
        label = "Try Again"

    logger.info(
        "scored score=%.2f label=%s raw_cosine=%.4f calibrated_score=%.2f confidence=%.3f ref_norm=%.4f child_norm=%.4f ref=%.3fs child=%.3fs",
        final_score,
        label,
        similarity_value,
        calibrated_score_value,
        confidence,
        ref_norm,
        child_norm,
        ref_dur,
        child_dur,
    )

    return ScoreResponse(
        score=round(final_score, 2),
        label=label,
        similarity=round(similarity_value, 4),
        ref_duration=round(ref_dur, 3),
        child_duration=round(child_dur, 3)
    )


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )