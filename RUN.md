# Pronunciation API — Run Guide

## 1. Install

```bash
pip install -r requirements.txt
```

> First run will also download the facebook/wav2vec2-base weights (~360 MB). This happens once and is cached automatically.

---

## 2. Run the API

```bash
python api.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

You should see:

```
09:12:01  INFO      pronunciation_api  Loading model from checkpoints/best_model.pt ...
09:12:06  INFO      pronunciation_api  Model ready  (device=cpu)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Interactive docs: http://localhost:8000/docs

---

## 3. Test the /score endpoint

### Windows (PowerShell)
```powershell
curl.exe -X POST http://localhost:8000/score `
  -F "reference=@reference.wav" `
  -F "child=@child.wav"
```

### Windows (CMD)
```cmd
curl -X POST http://localhost:8000/score -F "reference=@reference.wav" -F "child=@child.wav"
```

### Linux / macOS / Git Bash
```bash
curl -X POST http://localhost:8000/score \
  -F "reference=@reference.wav" \
  -F "child=@child.wav"
```

### Expected JSON response
```json
{
  "score": 78.43,
  "label": "Good",
  "similarity": 0.8312,
  "ref_duration": 0.812,
  "child_duration": 0.953
}
```

Label mapping:
- score >= 65 → "Good"
- score < 65 → "Try Again"

---

## 4. Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "checkpoint": "checkpoints/best_model.pt",
  "device": "cpu"
}
```

---

## 5. Public access with ngrok

### Install ngrok
Download from https://ngrok.com/download  
Or with pip:
```bash
pip install pyngrok
```

### Expose the API
```bash
ngrok http 8000
```

You will see output like:
```
Forwarding  https://a1b2c3d4.ngrok-free.app -> http://localhost:8000
```

Your public URL is: `https://a1b2c3d4.ngrok-free.app`

Send this URL to your Laravel backend. All endpoints work:
- POST  https://a1b2c3d4.ngrok-free.app/score
- GET   https://a1b2c3d4.ngrok-free.app/health

> ngrok URL changes every restart on the free plan. Use `ngrok http 8000 --subdomain=mypronunciationapi` with a paid plan for a fixed URL.

### Alternative: localtunnel (no account needed)
```bash
npm install -g localtunnel
lt --port 8000
```
Returns a URL like: `https://heavy-frog-12.loca.lt`

### Alternative: cloudflared (Cloudflare Tunnel, free + stable)
```bash
# Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
cloudflared tunnel --url http://localhost:8000
```
Returns a URL like: `https://random-name.trycloudflare.com`

---

## 6. Laravel integration

In Laravel, use Http facade or Guzzle:

```php
use Illuminate\Support\Facades\Http;

$response = Http::attach(
    'reference', file_get_contents($referenceAudioPath), 'reference.wav'
)->attach(
    'child', file_get_contents($childAudioPath), 'child.wav'
)->post('https://YOUR_NGROK_URL/score');

$result = $response->json();
// $result['score']      → float  (0-100)
// $result['label']      → string ("Good" | "Try Again")
// $result['similarity'] → float
```

---

## 7. Environment variable overrides

Set before running `python api.py`:

| Variable | Default | Description |
|---|---|---|
| CHECKPOINT_PATH | checkpoints/best_model.pt | Path to trained model |
| EMBED_DIM | 256 | Must match training embed_dim |
| NEW_DATA_DIR | data/raw/new_children | Where submitted recordings are saved |
| MAX_FILE_BYTES | 10485760 (10 MB) | Max upload size |
| MIN_AUDIO_SEC | 0.3 | Reject audio shorter than this |

Windows example:
```cmd
set CHECKPOINT_PATH=checkpoints/best_model.pt
python api.py
```

Linux/macOS example:
```bash
CHECKPOINT_PATH=checkpoints/best_model.pt python api.py
```
