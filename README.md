---
title: Model Card Compliance Auditor
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Model Card Compliance Auditor

## The Problem

HuggingFace hosts 1.2M+ models. Most model cards are incomplete, misleading, or non-compliant with EU AI Act documentation requirements — missing training data provenance, omitting bias evaluations, making SOTA claims unsupported by benchmarks, or declaring licenses incompatible with their base models. Manual review by governance officers is impossible at scale.

This environment trains agents to audit model cards automatically — the same task a governance team at a model hub would run on every upload. Judges from Meta and HuggingFace write model cards professionally. They know this problem firsthand.

## Quick Start (5 minutes)

```bash
git clone https://github.com/s9b/model-card-auditor
cd model-card-auditor
pip install -e .
uvicorn model_card_auditor.server.app:app --host 0.0.0.0 --port 7860
```

Then in another terminal:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_groq_key_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

## How It Works

The agent calls `POST /reset?task_id=easy|medium|hard` to start a new episode and receives an initial observation listing the available model card sections. It then calls `POST /step` repeatedly with actions (reading sections, flagging issues), receiving an observation and reward after each step. When the agent has identified all compliance violations, it calls `submit_audit` to end the episode and receive a final graded score.

## Environment Overview

An agent plays the role of an AI governance auditor. It receives a synthetic model card and must:
1. Read available sections
2. Detect completely missing required sections
3. Identify inadequate sections (too vague, no specifics, or inconsistent with other sections)
4. Cross-reference sections to find subtle compliance violations
5. Submit its audit report

### Action Space

| action_type | target | secondary_target | description |
|-------------|--------|-----------------|-------------|
| `read_section` | section name | — | Read a named section; reveals its content |
| `check_field` | field name | — | Check whether a field exists (+0.01) |
| `compare_sections` | section name | second section | Cross-reference two sections for contradictions (+0.03) |
| `flag_missing` | field name | — | Flag a required section as completely absent |
| `flag_inadequate` | field name | — | Flag a section that exists but fails compliance |
| `flag_compliant` | field name | — | Mark a section as passing all requirements |
| `submit_audit` | `"final"` | — | Submit the audit report and end the episode |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | bool | Episode over |
| `reward` | float | Step reward |
| `current_section_content` | str | Content returned by last action |
| `sections_available` | List[str] | All sections present in this model card |
| `sections_reviewed` | List[str] | Sections the agent has already read |
| `findings_count` | int | Total issues flagged so far |
| `partial_score` | float | Running compliance score 0.0–1.0 |
| `last_action_feedback` | str | Server feedback on last action |
| `steps_remaining` | int | Steps before forced episode end |
| `last_action_error` | str or None | Error message if action failed |

### Reward Shaping

Each step returns a reward signal that helps the agent learn which actions are productive.

| Action | Condition | Reward |
|--------|-----------|--------|
| `read_section` | New section | +0.02 |
| `read_section` | Already read | −0.02 |
| `read_section` | Not found | −0.02 |
| `check_field` | Any | +0.01 |
| `compare_sections` | Any | +0.03 |
| `flag_*` | Correct finding | +0.15 |
| `flag_*` | False positive | −0.04 |
| `submit_audit` | Terminal | `final_score × 0.40` |
| Any | Max steps hit | −0.10 |

## Tasks

| ID | Difficulty | What the agent must find | Challenge |
|----|-----------|--------------------------|-----------|
| `easy` | Easy | 4 completely missing required sections | Direct lookup — no cross-referencing needed |
| `medium` | Medium | 5 issues: mix of missing AND inadequate sections | Must classify correctly (flag_missing vs flag_inadequate) |
| `hard` | Hard | 5 subtle cross-section violations | Requires reading multiple sections and reasoning across them |

Hard task violations include: MIT license incompatible with LLaMA-2 base model, unfiltered web crawl data combined with minor-facing deployment, SOTA reasoning claims unsupported by evaluation benchmarks, geographic bias gap in a globally-deployed model, and a CO₂ calculation that is mathematically wrong (states 3.2 kgCO₂, correct value is 6.71 kgCO₂).

## Baseline Scores

Scores produced with temperature=0, seed=42 for reproducibility.

| Model | Provider | easy | medium | hard | avg |
|-------|----------|------|--------|------|-----|
| `llama-3.1-8b-instant` | Groq | 1.00 | 0.60 | 0.20 | 0.60 |
| `llama-3.3-70b-versatile` | Groq | 1.00 | 1.00 | 0.24 | 0.75 |
| `qwen-3-235b-a22b-instruct-2507` | Cerebras | 1.00 | 1.00 | 1.00 | 1.00 |

The pre-flight detection in `inference.py` programmatically flags absent required sections before invoking the LLM, guaranteeing easy = 1.00 regardless of model. The hard task requires cross-section reasoning; `qwen-3-235b-a22b-instruct-2507` achieves a perfect score by correctly identifying all 5 subtle violations.

## Setup

```bash
git clone https://github.com/s9b/model-card-auditor
cd model-card-auditor
pip install -e .
```

## Running the Environment

The live HuggingFace Space is already deployed at https://sazqt-model-card-auditor.hf.space — you can run inference against it without any local setup.

To run locally:

```bash
uvicorn model_card_auditor.server.app:app --host 0.0.0.0 --port 7860
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint base URL |
| `MODEL_NAME` | Yes | Model identifier string |
| `HF_TOKEN` | Yes | API key for the provider |
| `ENV_URL` | No | Environment URL (default: localhost:7860) |

## Running the Baseline Agent

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_token_here"
export ENV_URL="https://sazqt-model-card-auditor.hf.space"
python inference.py
```

Alternatively, against the HuggingFace Inference API:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
```

## Running Unit Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t model-card-auditor .
docker run -p 7860:7860 model-card-auditor
```

Health check: `curl http://localhost:7860/health`

## Technical Implementation

- **Framework**: OpenEnv (`openenv-core`) — implements `Environment`, `Action`, `Observation`, `State` base classes
- **Server**: FastAPI via `create_fastapi_app(ModelCardAuditEnvironment)` — exposes `/reset`, `/step`, `/state`, `/health`, `/ws`
- **Graders**: Fully deterministic, input-dependent — easy (4-field missing check), medium (5-issue mixed classification), hard (5-issue weighted cross-section violations with evidence bonus)
- **Scenarios**: Three synthetic model cards with planted compliance issues; ground truth hidden from agent
- **Concurrent sessions**: `SUPPORTS_CONCURRENT_SESSIONS = True` — stateless per-request design
