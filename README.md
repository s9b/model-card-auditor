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

## Description & Motivation

HuggingFace hosts over 1.2 million models. The vast majority have incomplete, inaccurate, or non-compliant model cards. Manual compliance review at this scale is impossible. This environment trains agents to automate AI governance — identifying missing required fields, inadequate documentation, and subtle cross-section inconsistencies that would take a human expert hours to catch.

This is a task the HuggingFace platform needs at every model upload. The same problem exists at Meta, Google, and every enterprise AI team deploying models responsibly.

## Action Space

| action_type | target | secondary_target | reason | severity | evidence |
|-------------|--------|-----------------|--------|----------|----------|
| read_section | section name | — | — | — | — |
| check_field | field name | — | — | — | — |
| compare_sections | section name | second section name | — | — | — |
| flag_missing | field name | — | why it's missing | low/medium/high/critical | — |
| flag_inadequate | field name | — | why it fails | low/medium/high/critical | exact quote |
| flag_compliant | field name | — | why it passes | — | — |
| submit_audit | "final" | — | — | — | — |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| done | bool | Is episode over |
| reward | float | Step reward |
| current_section_content | str | Content returned by last action |
| sections_available | List[str] | All sections in model card |
| sections_reviewed | List[str] | Sections already read |
| findings_count | int | Issues flagged so far |
| partial_score | float | Running score 0.0–1.0 |
| last_action_feedback | str | Feedback on last action |
| steps_remaining | int | Steps before forced end |
| last_action_error | str or None | Error if action failed |

## Tasks

| ID | Difficulty | Objective | Expected Score (baseline) |
|----|-----------|-----------|--------------------------|
| easy | Easy | Find 4 completely missing required fields | ~0.80 |
| medium | Medium | Identify 5 missing/inadequate sections (must classify correctly) | ~0.50 |
| hard | Hard | Detect 5 subtle cross-section compliance violations | ~0.20 |

## Setup

```bash
pip install git+https://github.com/saazbhargav/model-card-auditor
```

## Local Run

```bash
git clone https://github.com/saazbhargav/model-card-auditor
cd model-card-auditor
pip install -e .
uvicorn model_card_auditor.server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t model-card-auditor .
docker run -p 7860:7860 model-card-auditor
```

## Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="https://sazqt-model-card-auditor.hf.space"
python inference.py
```

## Baseline Scores

Measured using `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference Router against the live Space:

- easy:    0.0000
- medium:  0.0000
- hard:    0.0000
- average: 0.0000

Note: The baseline agent correctly navigates the environment (reads sections, receives rewards/penalties per step) but exhausts HF inference credits before completing full audits. Stronger agents that efficiently flag issues within the step budget will score significantly higher.
