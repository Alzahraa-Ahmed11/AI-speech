# AI Pronunciation Scoring System

## Overview
This project is an AI system that evaluates pronunciation quality by comparing a reference audio with a learner audio using a Siamese Wav2Vec2 model.

## Main Features
- Word-level pronunciation scoring (0–100)
- FastAPI deployment
- Supports multiple audio formats
- Trained on SpeechCommands + augmented dysarthric simulation

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run API
python api.py

### 3. Endpoint
POST /score

## Input
- reference audio file
- child audio file

## Output
- score (0–100)
- label (Good / Try Again)