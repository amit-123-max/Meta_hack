"""FastAPI application for TrafficSignalEnv — HF Spaces compatible.

Endpoints:
  GET  /                → health check + environment description
  POST /reset           → reset environment, return initial observation
  POST /step            → take a step, return obs/reward/done/info
  GET  /state           → full current state
  GET  /render          → current frame as PNG
  GET  /analytics       → episode analytics (if episode done)
  POST /grade           → grade current episode trajectory
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import easy_config, hard_config, medium_config
from env.traffic_env import TrafficSignalEnv
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from tasks.task_easy import make_env as make_easy
from tasks.task_hard import make_env as make_hard
from tasks.task_medium import make_env as make_medium
from utils.replay import build_analytics

# ---------------------------------------------------------------------------
app = FastAPI(
    title="TrafficSignalEnv",
    description=(
        "Smart-city adaptive traffic signal control benchmark. "
        "Agents observe camera-like image frames + metadata and control traffic lights "
        "to minimise congestion and prioritise emergency vehicles."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment state
# ---------------------------------------------------------------------------
_env: Optional[TrafficSignalEnv] = None
_task_id: str = "easy"
_graders = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def _get_env() -> TrafficSignalEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    action: List[int]          # one phase index per intersection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_dict(obs) -> Dict:
    """Convert TrafficObservation to JSON-serialisable dict."""
    # Return last frame as base64 PNG + metadata
    from PIL import Image
    last_frame = obs.frames[-1]  # (H, W, 3)
    img = Image.fromarray(last_frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "frame_b64_png": b64,
        "frame_shape":   list(obs.frames.shape),
        "metadata":      obs.metadata.tolist(),
        "step":          obs.step,
    }


def _state_to_dict(state) -> Dict:
    return {
        "step":               state.step,
        "global_throughput":  state.global_throughput,
        "global_avg_wait":    state.global_avg_wait,
        "phase_switches":     state.phase_switches,
        "done":               state.done,
        "episode_emergency_delays": state.episode_emergency_delays,
        "intersections": [
            {
                "id":           i.intersection_id,
                "phase":        i.current_phase.value,
                "phase_name":   i.current_phase.name,
                "phase_timer":  i.phase_timer,
                "yellow_remaining": i.yellow_remaining,
                "emergency":    i.emergency_active.value,
                "emergency_name": i.emergency_active.name,
                "emergency_lane": i.emergency_lane,
                "weather":      i.weather.value,
                "weather_name": i.weather.name,
                "spillback_count": i.spillback_count,
                "total_throughput": i.total_throughput,
                "lanes": [
                    {
                        "lane_id":    l.lane_id,
                        "direction":  l.direction,
                        "queue":      l.queue_length,
                        "throughput": l.throughput,
                        "is_green":   l.is_green,
                        "is_occluded":l.is_occluded,
                        "emergency":  l.emergency.value,
                    }
                    for l in i.lanes
                ],
            }
            for i in state.intersections
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    env_desc = TrafficSignalEnv(easy_config())
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TrafficSignalEnv — Smart City Traffic Control</title>
<style>
  :root {{color-scheme: dark;}}
  * {{box-sizing: border-box; margin: 0; padding: 0;}}
  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
    color: #e2e8f0;
    padding: 2rem;
  }}
  .card {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
  }}
  h1 {{font-size: 2.2rem; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;}}
  h2 {{font-size: 1.3rem; color: #a78bfa; margin-bottom: 1rem;}}
  p  {{line-height: 1.7; color: #94a3b8; margin-bottom: 0.75rem;}}
  code {{font-family: monospace; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; color: #60a5fa;}}
  .badge {{display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-right: 8px; margin-bottom: 8px;}}
  .badge-easy   {{background: #065f46; color: #6ee7b7;}}
  .badge-medium {{background: #78350f; color: #fcd34d;}}
  .badge-hard   {{background: #7f1d1d; color: #fca5a5;}}
  .endpoint {{display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem;}}
  .method {{padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; min-width: 48px; text-align: center;}}
  .get  {{background:#065f46; color:#6ee7b7;}}
  .post {{background:#1e3a5f; color:#93c5fd;}}
  .path {{color:#e2e8f0; font-family: monospace;}}
  .desc {{color:#94a3b8; font-size: 0.9rem;}}
</style>
</head>
<body>
<div class="card">
  <h1>🚦 TrafficSignalEnv</h1>
  <p>Smart-city adaptive traffic signal control benchmark — OpenEnv compliant.</p>
  <span class="badge badge-easy">Easy</span>
  <span class="badge badge-medium">Medium</span>
  <span class="badge badge-hard">Hard</span>
</div>

<div class="card">
  <h2>Observation Space</h2>
  <p><code>{env_desc.observation_space_description()}</code></p>
  <h2 style="margin-top:1.2rem">Action Space</h2>
  <p><code>{env_desc.action_space_description()}</code></p>
</div>

<div class="card">
  <h2>API Endpoints</h2>
  <div class="endpoint"><span class="method get">GET</span><div><div class="path">/</div><div class="desc">Health check and environment overview</div></div></div>
  <div class="endpoint"><span class="method post">POST</span><div><div class="path">/reset</div><div class="desc">Reset environment — body: {{"task_id": "easy|medium|hard", "seed": 42}}</div></div></div>
  <div class="endpoint"><span class="method post">POST</span><div><div class="path">/step</div><div class="desc">Take a step — body: {{"action": [0,1,...]}}</div></div></div>
  <div class="endpoint"><span class="method get">GET</span><div><div class="path">/state</div><div class="desc">Full current environment state</div></div></div>
  <div class="endpoint"><span class="method get">GET</span><div><div class="path">/render</div><div class="desc">Current frame as PNG image</div></div></div>
  <div class="endpoint"><span class="method get">GET</span><div><div class="path">/analytics</div><div class="desc">Episode analytics (queue heatmap, phase timeline, violations)</div></div></div>
  <div class="endpoint"><span class="method post">POST</span><div><div class="path">/grade</div><div class="desc">Grade current episode trajectory, return score ∈ [0,1]</div></div></div>
  <div class="endpoint"><span class="method get">GET</span><div><div class="path">/docs</div><div class="desc">Interactive Swagger UI</div></div></div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/reset")
async def reset_env(request: Optional[ResetRequest] = None) -> JSONResponse:
    """Reset environment to initial state.

    Accepts optional `task_id` and `seed` in request body.
    If no body, defaults to `easy` task and seed `42`.
    """
    global _env, _task_id

    if request is None:
        request = ResetRequest()

    _task_id = request.task_id
    seed = request.seed


@app.post("/step")
async def step(req: StepRequest) -> JSONResponse:
    env = _get_env()
    action = req.action
    obs, reward, done, info = env.step(action)
    return JSONResponse({
        "observation":       _obs_to_dict(obs),
        "reward":            reward,
        "done":              done,
        "info":              info,
    })


@app.get("/state")
async def get_state() -> JSONResponse:
    env = _get_env()
    state = env.state()
    return JSONResponse(_state_to_dict(state))


@app.get("/render")
async def render() -> Response:
    env = _get_env()
    frame = env.render()
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    # Scale up for human readability
    img = img.resize((420, 420), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/analytics")
async def analytics() -> JSONResponse:
    env = _get_env()
    result = build_analytics(env.trajectory, _task_id)
    return JSONResponse(result)


@app.post("/grade")
async def grade() -> JSONResponse:
    env = _get_env()
    grader = _graders.get(_task_id)
    if grader is None:
        raise HTTPException(status_code=400, detail=f"No grader for task: {_task_id}")
    score = grader.grade(env.trajectory)
    analytics_data = build_analytics(env.trajectory, _task_id)
    return JSONResponse({
        "task_id": _task_id,
        "score":   round(score, 4),
        "analytics": analytics_data,
    })
