---
title: Traffic Signal Env
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚦 TrafficSignalEnv

**Smart-city adaptive traffic signal control benchmark — OpenEnv compliant**

An RL environment that simulates traffic congestion at one or more intersections using camera-like image observations plus structured metadata. Agents learn adaptive traffic light scheduling, including emergency vehicle prioritisation (ambulance > fire > police > normal).

---

## Table of Contents

- [Overview](#overview)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Logic](#reward-logic)
- [Tasks](#tasks)
- [Grader Design](#grader-design)
- [Emergency Priority System](#emergency-priority-system)
- [Weather & Partial Observability](#weather--partial-observability)
- [Setup & Installation](#setup--installation)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Validation](#validation)
- [Baseline Scores](#baseline-scores)
- [API Reference](#api-reference)
- [Deployment (Hugging Face Spaces)](#deployment-hugging-face-spaces)
- [Project Structure](#project-structure)

---

## Overview

TrafficSignalEnv simulates realistic traffic dynamics at one or more intersections. Unlike toy grid-world environments, this benchmark:

- Uses **image-based observations** (top-down camera view, stacked over time) plus **compact metadata** vectors.
- Models **stochastic vehicle arrivals** (Poisson process), **queue buildup**, and **phase transitions** with yellow intermediates.
- Supports **emergency vehicle priority** (ambulance > fire > police).
- Includes **weather noise** (rain, fog, night, cloudy), **camera occlusion**, and **congestion propagation** between neighbouring intersections.
- Provides **three graded tasks** with deterministic, exploit-resistant graders.

---

## Observation Space

At each step the agent receives a `TrafficObservation` with two components:

### Image frames
```
Shape: (frame_stack=4, H=84, W=84, 3)  dtype: uint8
```
Each frame is a top-down rendering of all intersections. Visual elements:
- **Lane occupancy bars** — colour-coded green→yellow→red by queue fill ratio
- **Phase indicator** — intersection box colour (green=NS, blue=EW, red=ALL_RED, yellow=transition)
- **Phase timer bar** — thin white bar showing time in current phase
- **Emergency vehicle markers** — coloured square + letter (A=Ambulance, F=Fire, P=Police)
- **Weather overlays** — rain streaks, fog veil, night darkening applied to whole frame
- **Camera occlusion** — affected lanes rendered as dark grey

### Metadata
```
Shape: (n_intersections, 11)  dtype: float32
```
Features per intersection (all normalised to [0, 1]):

| Index | Feature |
|-------|---------|
| 0–3   | Lane queue fractions (N, S, E, W) |
| 4     | Current phase (0=NS_GREEN, 1=EW_GREEN, 2=ALL_RED) |
| 5     | Phase timer (fraction of max) |
| 6     | Yellow transition remaining |
| 7     | Emergency vehicle type (0=none, 1=police, 2=fire, 3=ambulance) |
| 8     | Emergency lane index |
| 9     | Weather condition (0=clear … 4=night) |
| 10    | Spillback flag (1.0 if any lane at capacity) |

---

## Action Space

```
Discrete(N_PHASES ^ n_intersections)
```
Phase encoding: `0=NS_GREEN`, `1=EW_GREEN`, `2=ALL_RED`

Accepted input formats:
- `List[int]` — one phase per intersection (recommended)
- `int` — flat integer decoded via mixed-radix encoding
- `TrafficAction` — typed action object with optional `emergency_override`

**Safety guarantees:**
- Invalid actions silently become no-ops (hold current phase)
- Phase changes below `phase_duration_min` are blocked (anti-oscillation)
- Yellow transitions are internal — the agent never needs to choose yellow

---

## Reward Logic

Dense, multi-objective reward at every step:

| Component | Weight | Description |
|-----------|--------|-------------|
| `throughput_bonus` | +0.40 | Vehicles released per step (normalised) |
| `queue_penalty`    | −0.25 | Mean queue fill ratio across all lanes |
| `wait_penalty`     | −0.20 | Cumulative wait time (normalised) |
| `switch_penalty`   | −0.08 | Phase switches that violate min duration |
| `emergency_bonus`  | +0.60 | Emergency lane gets green (×priority multiplier) |
| `spillback_penalty`| −0.30 | Lane at/above capacity threshold |
| `starvation_penalty`|−0.20 | Lane waiting excessively long |
| `fairness_bonus`   | +0.15 | Jain's fairness index across intersections |

**Emergency multipliers:** Ambulance ×4, Fire ×2, Police ×1

---

## Tasks

### Task 1 — Easy: Single Intersection
- **Config:** 1 intersection, 4 lanes, normal traffic only
- **Steps:** 500
- **Arrival rate:** λ=0.30 vehicles/step/lane
- **Objective:** Minimise average waiting time and queue length
- **No** emergency vehicles, weather, or partial observability

### Task 2 — Medium: 2×2 Grid
- **Config:** 4 intersections (2×2 grid), congestion propagation enabled
- **Steps:** 1 000
- **Arrival rate:** λ=0.40 (higher load)
- **Objective:** Maximise throughput, prevent cross-intersection spillback
- **No** emergency vehicles or weather

### Task 3 — Hard: Emergency + Partial Observability + Weather
- **Config:** 4 intersections (2×2), all features enabled
- **Steps:** 1 500
- **Arrival rate:** λ=0.45 + random spikes
- **Emergency probability:** 1.5% per step per intersection
- **Camera occlusion probability:** 8% per lane per step
- **Weather change probability:** 2% per step
- **Objective:** Emergency prioritisation + system stability + fairness

---

## Grader Design

All graders are **deterministic**, **trajectory-consuming**, return `float` ∈ [0, 1], and include degenerate-strategy penalties.

### Easy Grader
```
score = 0.40 × throughput_score
      + 0.35 × wait_score
      + 0.15 × queue_score
      − 0.10 × switch_penalty
```
Starvation guard: if one phase dominates >95% of steps → 50% score reduction.

### Medium Grader
```
score = 0.35 × throughput_score
      + 0.25 × spillback_score
      + 0.20 × wait_score
      + 0.12 × fairness_score
      − 0.08 × switch_penalty
```
Each starved intersection reduces score by 0.25.

### Hard Grader
```
score = 0.35 × emergency_response_score
      + 0.25 × throughput_score
      + 0.15 × wait_score
      + 0.12 × spillback_score
      + 0.08 × fairness_score
      − 0.05 × switch_penalty
```
Emergency response uses exponential decay:
- ≤5 steps response → 1.0 credit
- ≥30 steps response → 0.0 credit
- Ambulance gets 2× strictness multiplier vs. police

---

## Emergency Priority System

```
Ambulance (priority 3) > Fire (priority 2) > Police (priority 1) > Normal (0)
```

When an emergency vehicle appears:
1. It is assigned to a random lane with no active emergency
2. The vehicle has `emergency_clear_steps` steps to pass through
3. The reward strongly incentivises serving the emergency lane with green
4. If the vehicle waits too long, a penalty scales with its priority
5. The grader records response delay per-emergency and applies priority-weighted scoring

Emergency vehicles **do not** cause permanent green or infinite oscillation — the vehicle clears after a fixed number of steps regardless.

---

## Weather & Partial Observability

Weather conditions (hard task only):
| Condition | Visual Effect |
|-----------|--------------|
| Clear | No modification |
| Cloudy | Slight fog overlay (alpha=60) |
| Rain | Random rain streaks + mild blur |
| Fog | Heavy fog overlay (alpha=140) |
| Night | Frame darkened to 35% brightness |

Camera occlusion: individual lanes rendered as dark grey. The metadata `is_occluded` flag is `True` for those lanes. The agent must rely on frame history and metadata from neighbouring lanes.

---

## Setup & Installation

```bash
# Clone / enter repo
cd OpenENV

# Create virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Minimum requirements:** Python 3.10+, 2 vCPU, 8 GB RAM

---

## Running Locally

### Baseline inference (all 3 tasks)
```bash
python inference.py
```

### Specific task
```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### LLM-powered agent (requires API access)

Create a `.env` file in the root of the project and add your Hugging Face token:
```
HF_TOKEN="your_hugging_face_token"
```

Then run the inference script:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --llm --task hard
```

### Start the API server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Visit: http://localhost:7860
```

---

## Running with Docker

```bash
# Build
docker build -t traffic-signal-env .

# Run
docker run -p 7860:7860 traffic-signal-env

# With LLM env vars
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  --env-file .env \
  traffic-signal-env
```

---

## Validation

```bash
# Quick import check
python -c "from env.traffic_env import TrafficSignalEnv; print('OK')"

# Full baseline run (validates all 3 tasks end-to-end)
python inference.py

# Check grader outputs are in [0,1]
python -c "
from tasks.task_easy import make_env
from graders.easy_grader import EasyGrader
env = make_env()
env.reset()
for _ in range(50):
    obs, r, done, info = env.step([0])
    if done: break
score = EasyGrader().grade(env.trajectory)
assert 0.0 <= score <= 1.0, f'Invalid score: {score}'
print(f'Easy grader score: {score:.4f}')
"
```

---

## Baseline Scores

Rule-based heuristic agent, seed=42, 3 independent runs average:

| Task   | Score  | Notes |
|--------|--------|-------|
| Easy   | ~0.72  | Pressure-based switching, no emergencies |
| Medium | ~0.64  | Grid coordination, some spillback |
| Hard   | ~0.55  | Emergency handling under noise + occlusion |

*LLM agent (GPT-4o-mini) scores approximately 5–10% higher on hard task due to context-aware emergency reasoning.*

---

## API Reference

After starting the server (`uvicorn app.main:app --port 7860`):

| Method | Path | Body | Description |
|--------|------|------|-------------|
| GET  | `/`         | — | HTML overview page |
| POST | `/reset`    | `{"task_id":"easy","seed":42}` | Reset env, returns initial obs |
| POST | `/step`     | `{"action":[0,1,0,2]}` | Step env, returns obs/reward/done/info |
| GET  | `/state`    | — | Full current state JSON |
| GET  | `/render`   | — | Current frame as PNG |
| GET  | `/analytics`| — | Episode analytics (heatmap, timeline, violations) |
| POST | `/grade`    | — | Grade current episode, returns score ∈ [0,1] |
| GET  | `/docs`     | — | Interactive Swagger UI |

---

## Deployment (Hugging Face Spaces)

1. Create a new Space with **Docker** SDK
2. Push this repository
3. Create a `.env` file in the root of the project and add your Hugging Face token:
   ```
   HF_TOKEN="your_hugging_face_token"
   ```
4. Set environment variables in Space settings:
   - `API_BASE_URL` (optional, for LLM agent)
   - `MODEL_NAME` (default: `gpt-4o-mini`)
5. The app starts automatically on port 7860

---

## Project Structure

```
OpenENV/
├── openenv.yaml              # OpenEnv manifest
├── inference.py              # Baseline inference script
├── Dockerfile                # Container build
├── requirements.txt
├── README.md
├── config/
│   ├── env_config.py         # EnvConfig, SimConfig dataclasses
│   └── task_configs.py       # easy/medium/hard presets
├── env/
│   ├── schemas.py            # Typed Observation/Action/Reward/State
│   ├── simulator.py          # Core traffic simulator
│   ├── observation.py        # Image + metadata builder
│   ├── reward.py             # Dense multi-objective reward
│   └── traffic_env.py        # OpenEnv-compliant env class
├── tasks/
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
├── graders/
│   ├── base_grader.py
│   ├── easy_grader.py
│   ├── medium_grader.py
│   └── hard_grader.py
├── baseline/
│   └── rule_based_agent.py   # Heuristic baseline policy
├── utils/
│   └── replay.py             # Episode analytics & replay
└── app/
    └── main.py               # FastAPI server (HF Spaces)
```

---

## License

MIT
