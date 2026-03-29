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

Tested via Groq (temperature=0, seed=42).

| Model | Task | Score | Notes |
|-------|------|-------|-------|
| `llama-3.3-70b-versatile` | easy | **1.0000** | Perfect — all 4 missing sections found |
| `llama-3.3-70b-versatile` | medium | **1.0000** | Perfect — 2 missing (pre-flight) + 3 inadequate (LLM) |
| `llama-3.3-70b-versatile` | hard | **0.2000** | Finds CO₂ error; cross-section flags use compound targets |
| `llama-3.3-70b-versatile` | **average** | **0.7333** | |
| `llama-3.1-8b-instant` *(8B baseline)* | easy | 1.0000 | Perfect |
| `llama-3.1-8b-instant` *(8B baseline)* | medium | 0.6000 | Misses 2 absent sections without pre-flight |
| `llama-3.1-8b-instant` *(8B baseline)* | hard | 0.2000 | Finds License violation only |
| `llama-3.1-8b-instant` *(8B baseline)* | **average** | **0.6000** | |

The pre-flight detection in `inference.py` programmatically flags absent required sections before invoking the LLM — lifting medium from 0.6 (8B) to 1.0 (70B). The hard task requires cross-section reasoning that benefits from stronger models; the 70B model identifies the correct violations but uses compound section names as targets, which the grader scores as false positives.

## Setup

```bash
git clone https://github.com/saazbhargav/model-card-auditor
cd model-card-auditor
pip install -e .
```

## Running the Environment

```bash
uvicorn model_card_auditor.server.app:app --host 0.0.0.0 --port 7860
```

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
