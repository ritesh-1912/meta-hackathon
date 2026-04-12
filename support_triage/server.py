from __future__ import annotations

import os
from argparse import ArgumentParser
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .environment import SupportTicketEnv
from .models import SupportTicketAction

app = FastAPI(title="Support Ticket Triage OpenEnv", version="1.0.0")
_env = SupportTicketEnv()
_lock = Lock()


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    seed: int | None = None


def _reset_env(task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
    with _lock:
        try:
            observation = _env.reset(task_id=task_id, seed=seed)
            state = _env.state()
        except Exception as exc:  # pragma: no cover - surfaced through HTTP response
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "state": state.model_dump(),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "support-ticket-triage-openenv",
        "status": "ok",
        "message": "OpenEnv support-ticket triage service is running.",
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    payload = request or ResetRequest()
    return _reset_env(task_id=payload.task_id, seed=payload.seed)


@app.get("/reset")
def reset_get(task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
    return _reset_env(task_id=task_id, seed=seed)


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
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
