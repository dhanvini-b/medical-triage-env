---
title: Medical Triage Env
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# Medical Triage Environment

An OpenEnv-compatible environment where an AI agent must assess patient symptoms,
vitals, and medical history to assign the correct triage urgency level.

## Motivation

Medical triage is a high-stakes real-world task performed by healthcare workers
every day — especially critical in countries like India where doctor availability
is limited. This environment tests whether AI agents can reason clinically under
uncertainty and prioritize patients correctly.

## Action Space

| Field | Type | Values |
|-------|------|--------|
| triage_level | string | "emergency", "urgent", "routine" |
| reasoning | string | Agent's clinical explanation |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| patient_id | string | Unique patient identifier |
| age | integer | Patient age in years |
| symptoms | list[string] | Reported symptoms |
| vitals | dict | heart_rate, bp_systolic, bp_diastolic, spo2, temperature |
| history | list[string] | Relevant medical history |
| step_number | integer | Current step in episode |

## Tasks

| Task | Difficulty | Description | Success Threshold |
|------|-----------|-------------|-------------------|
| easy | Easy | Classic emergency — chest pain, low BP, low SpO2 | 0.7 |
| medium | Medium | Ambiguous fever and fatigue — flu, dengue, or typhoid? | 0.5 |
| hard | Hard | Atypical MI in elderly diabetic — easy to miss | 0.7 |

## Reward Function

- **0.7** for correct triage level
- **up to 0.3** for reasoning quality (based on symptoms and history referenced)
- **0.0** for sending a routine patient to emergency (over-triage penalized less than under-triage)

Total reward is between 0.0 and 1.0.

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.82 |
| medium | 0.88 |
| hard | 0.83 |

Model: Qwen/Qwen2.5-72B-Instruct via HuggingFace Router

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

### Run Baseline

```bash
export HF_TOKEN=your_token_here
python inference.py
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start new episode. Body: {"task": "easy"} |
| /step | POST | Take action. Body: {"triage_level": "...", "reasoning": "..."} |
| /state | GET | Get current environment state |
| /health | GET | Health check |