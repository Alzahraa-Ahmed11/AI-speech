import json
from pathlib import Path
from collections import defaultdict
import librosa
import numpy as np


AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def analyze_speechcommands(root):
    report = {}

    root = Path(root)

    for word_dir in root.iterdir():
        if not word_dir.is_dir():
            continue

        word = word_dir.name
        files = [
            f for f in word_dir.rglob("*")
            if f.suffix.lower() in AUDIO_EXTS
        ]

        durations = []
        speakers = set()
        bad_files = 0

        for f in files[:200]:  # sample للتحليل فقط
            try:
                y, sr = librosa.load(f, sr=16000, mono=True)

                if len(y) == 0:
                    bad_files += 1
                    continue

                durations.append(len(y) / sr)

                # SpeechCommands speaker id
                if "_nohash_" in f.stem:
                    speakers.add(f.stem.split("_nohash_")[0])

            except Exception:
                bad_files += 1

        if durations:
            report[word] = {
                "files": len(files),
                "avg_duration_sec": round(float(np.mean(durations)), 3),
                "min_duration_sec": round(float(np.min(durations)), 3),
                "max_duration_sec": round(float(np.max(durations)), 3),
                "estimated_speakers": len(speakers),
                "sample_bad_files": bad_files,
            }

    return report


def analyze_torgo(root):
    root = Path(root)

    files = [
        f for f in root.rglob("*")
        if f.suffix.lower() in AUDIO_EXTS
    ]

    durations = []
    speakers = set()
    bad_files = 0

    for f in files[:1000]:  # sample أكبر لأن TORGO غير structured
        try:
            y, sr = librosa.load(f, sr=16000, mono=True)

            if len(y) == 0:
                bad_files += 1
                continue

            durations.append(len(y) / sr)

            # speaker estimate from folder name
            parts = f.parts
            for p in parts:
                if p.startswith(("FC", "MC", "F", "M")):
                    speakers.add(p)
                    break

        except Exception:
            bad_files += 1

    return {
        "total_files": len(files),
        "avg_duration_sec": round(float(np.mean(durations)), 3) if durations else 0,
        "min_duration_sec": round(float(np.min(durations)), 3) if durations else 0,
        "max_duration_sec": round(float(np.max(durations)), 3) if durations else 0,
        "estimated_speakers": len(speakers),
        "sample_bad_files": bad_files,
    }


if __name__ == "__main__":

    report = {
        "SpeechCommands": analyze_speechcommands(
            "DATA_SET/SpeechCommands"
        ),
        "TORGO": analyze_torgo(
            "DATA_SET/torgo"
        ),
    }

    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved report.json")