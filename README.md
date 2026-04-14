---
title: Traffic Corridor Pro
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
license: apache-2.0
---

# Traffic Corridor Pro

Real-world OpenEnv benchmark for adaptive traffic signal control with emergency-priority behavior and corridor coordination.

[![Hugging Face Space](https://img.shields.io/badge/dynamic/json?label=HF%20Space&query=%24.runtime.stage&url=https://huggingface.co/api/spaces/Sanjeev-Kumar78/traffic_corridor_RL&logo=huggingface&style=for-the-badge&color=brightgreen&labelColor=151515)](https://huggingface.co/spaces/Sanjeev-Kumar78/traffic_corridor_RL)
[![HF Hardware](https://img.shields.io/badge/dynamic/json?label=Hardware&query=%24.runtime.hardware.current&url=https://huggingface.co/api/spaces/Sanjeev-Kumar78/traffic_corridor_RL&style=for-the-badge&color=007ec6&labelColor=151515)](https://huggingface.co/spaces/Sanjeev-Kumar78/traffic_corridor_RL)
[![GitHub](https://img.shields.io/badge/GitHub-traffic__corridor__RL-181717?logo=github&style=for-the-badge)](https://github.com/Sanjeev-Kumar78/traffic_corridor_RL)

## Project Links

- GitHub: https://github.com/Sanjeev-Kumar78/traffic_corridor_RL
- Hugging Face Space: https://huggingface.co/spaces/Sanjeev-Kumar78/traffic_corridor_RL

## Quick Start

Run locally in two commands:

```bash
uv sync
uv run server
```

Then in a second terminal:

```bash
uv run python inference.py
```

## Tasks

- `easy_4_phase`: one balanced intersection.
- `medium_asymmetric`: one intersection with directional imbalance.
- `hard_corridor_emergency`: three-intersection corridor with emergency pressure and downstream coupling.

## API Summary

- `POST /reset` with `{ "task_id": "...", "seed": 77, "session_id": "optional" }`
- `GET /state?session_id=...`
- `POST /step?session_id=...` with `{ "actions": [{ "intersection_id": 0, "phase": 0 }] }`
- `GET /history?session_id=...`

### Endpoint Compatibility Matrix

| Route      | Method | Expected      | Notes                          |
| ---------- | ------ | ------------- | ------------------------------ |
| `/`        | GET    | 200 (HTML UI) | Root interactive UI            |
| `/state`   | GET    | 200 (JSON)    | Current session state          |
| `/history` | GET    | 200 (JSON)    | Session trajectory history     |
| `/reset`   | POST   | 200 (JSON)    | Accepts `task_id`, `seed`, `session_id` |
| `/step`    | POST   | 200 (JSON)    | Returns reward plus next observation |

Phase map:

- `0`: `N_S_Straight`
- `1`: `N_S_Left`
- `2`: `E_W_Straight`
- `3`: `E_W_Left`

## Hugging Face Space UI

The root route `/` serves a built-in browser UI.

- Interactive mode: call reset/state/history/step directly.
- Direct-route mode: set base URL to your HF Space URL and open endpoint links.

Optional env for prefilled Space URL in the UI:

```bash
HF_SPACE_URL=https://your-space-name.hf.space
```

## Inference

`inference.py` uses OpenAI-compatible HF routing, prints strict evaluator logs, and scores trajectories with deterministic graders.

Required env vars:

- `API_BASE_URL`
- `MODEL_NAME`

Optional for LLM-tuned policy hints:

- `HF_TOKEN`

Common optional env vars:

- `ENV_BASE_URL` (default `http://localhost:8000`)
- `TRAFFIC_CORRIDOR_TASKS`
- `TRAFFIC_CORRIDOR_EVAL_SEEDS`
- `SUCCESS_SCORE_THRESHOLD`

## Local Run

### Recommended (uv)

```bash
uv sync
uv run server
```

Second terminal:

```bash
uv run python inference.py
```

Run tests:

```bash
python -B -m unittest discover -s tests -v
```

### Fallback (venv + pip)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn environment:app --host 0.0.0.0 --port 8000
```

Second terminal:

```bash
python inference.py
```

## Docker

```bash
docker build -t traffic-corridor-pro:latest .
docker run --rm -p 8000:8000 traffic-corridor-pro:latest
```

## Final Verified Scores

Latest validated 3-seed evaluation run:

- `easy_4_phase`: `0.823`
- `medium_asymmetric`: `0.747`
- `hard_corridor_emergency`: `0.685`

Trivial baselines for comparison:

- `always_0`: `0.107 / 0.142 / 0.114`
- `round_robin`: `0.181 / 0.194 / 0.015`

## Submission Files

- `Dockerfile`
- `environment.py`
- `graders.py`
- `inference.py`
- `openenv.yaml`
- `requirements.txt`
