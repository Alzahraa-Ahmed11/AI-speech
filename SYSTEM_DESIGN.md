# System Design — AI Pronunciation Scoring

## Architecture

The system is divided into three layers:

### 1. Training Pipeline
- SpeechCommands dataset
- TORGO dataset (used as negative samples only)
- Siamese Wav2Vec2 encoder
- Scoring head (MLP)

### 2. Inference Pipeline
- Audio preprocessing (16kHz mono normalization)
- Shared encoder
- Embedding comparison
- Calibrated scoring function

### 3. API Layer
- FastAPI server
- /score endpoint
- Multi-format audio decoding
- Thread-safe inference

## Flow

Reference Audio + Child Audio
        ↓
Preprocessing (16kHz, normalize)
        ↓
Wav2Vec2 Encoder (shared weights)
        ↓
Embedding vectors (256-d)
        ↓
Scoring Head (MLP comparison)
        ↓
Final score (0–100)