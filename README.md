# Support Ticket Triage OpenEnv

A compact OpenEnv benchmark built around a realistic support operations workflow. The agent must triage customer tickets, route them correctly, and draft policy-safe responses while making steady progress across a short episode.

## Why this domain

Support triage is a real task that humans perform every day. It is a good fit for OpenEnv because it has clear state, deterministic grading, meaningful partial progress, and obvious failure modes such as misrouting, unsafe promises, or weak prioritization.

## Environment overview

The environment exposes a typed OpenEnv-style interface with `reset()`, `step(action)`, and `state()`. Episodes sample one of three tasks with increasing difficulty:

- Easy: duplicate charge refund request
- Medium: suspicious account lockout
- Hard: enterprise API degradation response

Each episode rewards partial progress as the agent completes more of the triage checklist.

## Observation space

The observation includes:

- Episode and task identifiers
- Difficulty level
- Remaining steps
- Ticket title and customer message
- Ticket metadata such as channel, sentiment, urgency, and account tier
- Allowed action fields
- Current progress and last feedback
- Task instructions and the action schema

## Action space

The action is a typed JSON object with these optional fields:

- `classification`
- `priority`
- `route_to`
- `summary`
- `response_draft`
- `escalate`
- `confidence`

The environment scores each field deterministically and gives incremental reward as new components are completed.

## Tasks

### 1. billing_double_charge

- Difficulty: easy
- Objective: classify a duplicate charge, prioritize it correctly, route it to payments, and write a safe response that does not promise an instant refund.

### 2. security_lockout_triage

- Difficulty: medium
- Objective: treat the ticket as security-sensitive, assign urgent priority, route it to trust and safety, and draft a policy-safe response that protects account security.

### 3. enterprise_api_degradation

- Difficulty: hard
- Objective: handle an enterprise incident, assign the correct priority, route it to platform engineering, and write a careful customer response that avoids claiming a root cause or fix time.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

Start the API:

```bash
uvicorn support_triage.server:app --host 0.0.0.0 --port 7860
```

Run the baseline agent:

```bash
python inference.py
```

To run one task only:

```bash
TASK_ID=billing_double_charge python inference.py
```

## Environment variables

The baseline script reads:

- `API_BASE_URL` for the OpenAI-compatible endpoint
- `MODEL_NAME` for the model identifier
- `HF_TOKEN` or `API_KEY` for credentials
- `LOCAL_IMAGE_NAME` for compatibility with docker-image based setups
- `TASK_ID` to target a single task
- `BENCHMARK_NAME` for logging

## Baseline performance

The repository includes a deterministic fallback policy so the baseline can run even without an API key. The current local fallback scores are:

- billing_double_charge: 0.95
- security_lockout_triage: 0.89
- enterprise_api_degradation: 0.89

## Validation

The submission should pass:

- `docker build .`
- `docker run`
- `openenv validate`
- the `/reset` ping used by the validator

## Files of interest

- [openenv.yaml](openenv.yaml)
- [Dockerfile](Dockerfile)
- [inference.py](inference.py)
- [support_triage/environment.py](support_triage/environment.py)
- [support_triage/scenarios.py](support_triage/scenarios.py)
- [support_triage/models.py](support_triage/models.py)
- [support_triage/server.py](support_triage/server.py)
