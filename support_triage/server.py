"""FastAPI HTTP server for the support ticket triage OpenEnv.

Provides REST endpoints for the OpenEnv interface:
  - GET  /       : Health check
  - POST /reset  : Initialize episode (with optional query params)
  - GET  /reset  : Initialize episode via GET (compatibility)
  - POST /step   : Execute action
  - GET  /state  : Get current episode state

Thread-safe with a global lock protecting environment state.
Supports both query parameters and JSON request bodies for flexibility.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from threading import Lock
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .environment import SupportTicketEnv
from .models import ResetRequest, SupportTicketAction

app = FastAPI(title="Support Ticket Triage OpenEnv", version="1.0.0")
_env = SupportTicketEnv()
_lock = Lock()


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict[str, Any]


def _reset_env(task_id: str | None = None, seed: int | None = None):
    with _lock:
        try:
            observation = _env.reset(task_id=task_id, seed=seed)
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return observation


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "env": "support-ticket-triage"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    return _reset_env(task_id=request.task_id, seed=request.seed)


@app.get("/reset")
async def reset_get(task_id: str | None = None, seed: int | None = None):
    return _reset_env(task_id=task_id, seed=seed)


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    """POST /step: Execute an action in the current episode.
    
    Request body:
        action: dict with keys from {classification, priority, route_to, summary, response_draft, escalate, confidence}
    
    Returns:
        observation: Updated observation after action
        reward: RewardBreakdown with component scores and episode score
        done: Whether episode is complete
        info: Metadata (task_id, score, done, step_count, progress, history)
    """
    with _lock:
        try:
            observation, reward, done, info = _env.step(SupportTicketAction.model_validate(request.action))
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    """GET /state: Retrieve the current episode state.
    
    Returns:
        episode_id: Unique ID for this episode
        task_id: The active task (billing_double_charge, etc.)
        task_title: Human-readable task description
        difficulty: easy | medium | hard
        score: Current cumulative score [0.0, 1.0]
        done: Whether episode is complete
        progress: Dict of component_name → score
        history: List of (step, action, feedback) dicts
    """
    with _lock:
        try:
            current_state = _env.state()
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return current_state.model_dump()


def main() -> None:
    parser = ArgumentParser(description="Run the support ticket triage OpenEnv server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
