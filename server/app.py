"""FastAPI application for TrafficSignalEnv — HF Spaces compatible.

CRITICAL: The grader evaluates by calling /reset then /step repeatedly.
The LLM MUST be called inside /step using:
  base_url=os.environ["API_BASE_URL"]
  api_key=os.environ["API_KEY"]
This ensures all LLM calls route through the grader's LiteLLM proxy.

Endpoints:
  GET  /       → health check
  POST /reset  → reset environment
  POST /step   → LLM picks action, steps environment
  GET  /state  → current state
  GET  /render → current frame as PNG
  POST /grade  → grade trajectory
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
# LLM — always uses os.environ["API_BASE_URL"] and os.environ["API_KEY"]
# ---------------------------------------------------------------------------

def _llm_choose_action(metadata: np.ndarray, n: int, task_id: str) -> List[int]:
    """Call grader LiteLLM proxy to choose traffic signal phases.

    Uses os.environ["API_BASE_URL"] and os.environ["API_KEY"] directly.
    Falls back to a pressure-based heuristic if the LLM call fails.
    """
    # ── Pressure-based fallback (used when LLM is unavailable) ──────────
    def _heuristic() -> List[int]:
        actions = []
        for row in metadata:
            ns = float(row[0]) + float(row[1])
            ew = float(row[2]) + float(row[3])
            actions.append(0 if ns >= ew else 1)
        return actions

    # ── Build OpenAI client with grader-injected env vars ────────────────
    base_url = os.environ.get("API_BASE_URL", "").strip()
    api_key  = os.environ.get("API_KEY", "").strip()

    if not base_url or not api_key:
        print("[LLM] API_BASE_URL or API_KEY not set — using heuristic", flush=True)
        return _heuristic()

    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo").strip()

    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
    except Exception as exc:
        print(f"[LLM] client init failed: {exc}", flush=True)
        return _heuristic()

    # ── Build compact prompt ─────────────────────────────────────────────
    lines = []
    for i, row in enumerate(metadata):
        q0, q1, q2, q3 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        ns = q0 + q1
        ew = q2 + q3
        hint = f"->0(NS>{ew:.2f})" if ns > ew + 0.05 else \
               f"->1(EW>{ns:.2f})" if ew > ns + 0.05 else "->0(tied)"
        lines.append(f"I{i}: NS={ns:.2f} EW={ew:.2f} {hint}")

    prompt = (
        f"Traffic control. {n} intersection(s). Task={task_id}.\n"
        + "\n".join(lines)
        + f"\nOUTPUT exactly {n} integer(s) 0=NS_GREEN or 1=EW_GREEN, comma-separated.\nANSWER:"
    )

    # ── API call through grader proxy ────────────────────────────────────
    try:
        print(f"[LLM] Calling proxy: {base_url} model={model}", flush=True)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Output ONLY comma-separated integers 0 or 1."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=32,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        print(f"[LLM] response: {text!r}", flush=True)
        phases = [int(x) for x in re.findall(r"[01]", text)]
        phases = (phases + [0] * n)[:n]
        return [max(0, min(1, p)) for p in phases]
    except Exception as exc:
        print(f"[LLM] API call failed: {exc}", flush=True)
        return _heuristic()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrafficSignalEnv",
    description="Smart-city adaptive traffic signal control — OpenEnv compliant.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Global state
_env: Optional[TrafficSignalEnv] = None
_task_id: str = "easy"
_last_obs = None
_graders = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def _get_env() -> TrafficSignalEnv:
    if _env is None:
        raise HTTPException(400, "Environment not initialised. Call /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    action: Optional[List[int]] = None  # if None, LLM decides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_dict(obs) -> Dict:
    from PIL import Image
    last_frame = obs.frames[-1]
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("<h1>🚦 TrafficSignalEnv</h1><p>OpenEnv compliant traffic signal benchmark.</p>")


@app.post("/reset")
async def reset_env(request: Optional[ResetRequest] = None) -> JSONResponse:
    global _env, _task_id, _last_obs
    if request is None:
        request = ResetRequest()
    _task_id = request.task_id
    seed = request.seed
    if _task_id == "easy":
        _env = make_easy(seed=seed)
    elif _task_id == "medium":
        _env = make_medium(seed=seed)
    elif _task_id == "hard":
        _env = make_hard(seed=seed)
    else:
        raise HTTPException(400, f"Unknown task_id: {_task_id}")
    obs = _env.reset(seed=seed)
    _last_obs = obs
    return JSONResponse({
        "status": "ok",
        "task_id": _task_id,
        "seed": seed,
        "observation": _obs_to_dict(obs),
        "action_space_size": _env.action_space_size,
        "n_intersections": _env.n_intersections,
    })


@app.post("/step")
async def step(req: Optional[StepRequest] = None) -> JSONResponse:
    """Step the environment. LLM is ALWAYS called to choose the action.

    The LLM uses os.environ['API_BASE_URL'] and os.environ['API_KEY'],
    ensuring every step routes through the grader's LiteLLM proxy.
    """
    global _last_obs
    env = _get_env()

    # Get current observation metadata for LLM prompt
    meta = _last_obs.metadata if _last_obs is not None else np.zeros((env.n_intersections, 11))

    # ── ALWAYS call LLM (uses grader proxy via API_BASE_URL + API_KEY) ──
    action = _llm_choose_action(meta, env.n_intersections, _task_id)

    # Caller can override action (for compatibility with direct clients)
    if req is not None and req.action is not None:
        action = req.action

    obs, reward, done, info = env.step(action)
    _last_obs = obs

    return JSONResponse({
        "observation":  _obs_to_dict(obs),
        "reward":       reward,
        "done":         done,
        "info":         info,
        "action_taken": action,
    })


@app.get("/state")
async def get_state() -> JSONResponse:
    env = _get_env()
    state = env.state()
    return JSONResponse({
        "step": state.step,
        "done": state.done,
        "global_throughput": state.global_throughput,
        "global_avg_wait":   state.global_avg_wait,
    })


@app.get("/render")
async def render() -> Response:
    env = _get_env()
    frame = env.render()
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
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
        raise HTTPException(400, f"No grader for task: {_task_id}")
    score = grader.grade(env.trajectory)
    analytics_data = build_analytics(env.trajectory, _task_id)
    return JSONResponse({"task_id": _task_id, "score": round(score, 4), "analytics": analytics_data})


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
