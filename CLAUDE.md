# CLAUDE.md — Model Card Compliance Auditor
## Master Context File for Claude Code

**Participant:** Saaz Bhargava (`saazbhargav@gmail.com`)
**Hackathon:** Meta × PyTorch × HuggingFace — Round 1
**Hard Deadline:** 8 April 2026, 11:59 PM IST (resubmit allowed — judges use latest)
**Project Name:** `model-card-auditor`

---

## 0. HOW TO USE THIS FILE

Read this ENTIRE file before writing a single line of code. It contains:
- All resolved contradictions (Section 0 of PRD)
- Complete implementation for every file
- All 3 data scenarios with exact ground truth
- Scoring criteria maximization strategy
- Testing commands and deployment checklist

When you start a new session, run `/memory` or re-read this file to reload context.
When you finish a phase, update `TASKS.md` to mark tasks complete.

---

## 1. PROJECT OVERVIEW

**What it is:** An OpenEnv environment simulating an AI governance officer auditing HuggingFace model cards for compliance, completeness, and responsible AI standards.

**Why it wins:**
- Judges are from Meta + HuggingFace — they write model cards professionally. Instant real-world recognition.
- No other team will build this domain. It requires lateral thinking.
- Fully deterministic graders — we plant the bugs ourselves.
- Rich multi-step trajectory — agent must read, cross-reference, reason, flag. Not just one click.
- HuggingFace hosts 1.2M+ models. Automated compliance = immediate platform value.

**Scoring weights:**
| Criterion | Weight | Our Target |
|-----------|--------|------------|
| Real-world utility | 30% | 28/30 |
| Task & grader quality | 25% | 24/25 |
| Environment design | 20% | 19/20 |
| Code quality & spec compliance | 15% | 14/15 |
| Creativity & novelty | 10% | 10/10 |

**Disqualification risks (avoid all):**
- Environment doesn't deploy or respond → covered by Dockerfile + health check
- Plagiarised environment → our domain is original
- Graders that always return same score → our graders are input-dependent
- No baseline inference script → inference.py in root

---

## 2. RESOLVED CONTRADICTIONS (READ CAREFULLY)

### Contradiction 1 — API key naming
- Functional Requirements says: `OPENAI_API_KEY`
- Additional Instructions says: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- **RESOLUTION:** Support BOTH. `HF_TOKEN` primary, `OPENAI_API_KEY` fallback.
```python
api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")
```

### Contradiction 2 — Reward Pydantic model
- Rules say: "typed Observation, Action, and Reward Pydantic models"
- Actual OpenEnv framework: `reward` is a `float` field INSIDE `Observation`. No separate `Reward` class.
- **RESOLUTION:** Follow the framework. `reward` lives inside `Observation`. Do NOT create a separate `Reward` class unless `openenv validate` specifically demands it.

### Contradiction 3 — step() return type
- Rules say: "step(action) → returns observation, reward, done, info"
- Actual framework: server-side `step()` returns an `Observation` object (which contains `done` + `reward`). `create_fastapi_app` handles serialization. Client sees a `StepResult` with `.observation`, `.reward`, `.done`.
- **RESOLUTION:** Server `step()` returns `ModelCardObservation`. Client (`inference.py`) accesses `result.observation`, `result.reward`, `result.done`.

---

## 3. COMPLETE FILE STRUCTURE

```
model-card-auditor/               ← GitHub repo root
│
├── inference.py                  ← MUST be at root (not in subfolder)
├── openenv.yaml                  ← metadata, must include tag "openenv"
├── Dockerfile                    ← docker build + docker run must work
├── requirements.txt              ← all deps, pinned versions
├── pyproject.toml                ← package metadata
├── README.md                     ← all required sections
│
└── model_card_auditor/
    ├── __init__.py               ← exports ModelCardAuditClient, ModelCardAction
    ├── models.py                 ← Action, Observation, State Pydantic models
    ├── client.py                 ← WebSocket client (what users import)
    │
    ├── data/
    │   ├── easy.json
    │   ├── medium.json
    │   └── hard.json
    │
    └── server/
        ├── __init__.py
        ├── environment.py        ← reset(), step(), state, close()
        ├── graders.py            ← grade_easy(), grade_medium(), grade_hard()
        └── app.py                ← create_fastapi_app(ModelCardAuditEnvironment)
```

---

## 4. openenv.yaml

```yaml
name: model-card-auditor
version: "1.0.0"
description: >
  An AI governance environment where agents audit HuggingFace model cards
  for compliance with responsible AI documentation standards.
tags:
  - openenv
tasks:
  - id: easy
    description: Identify missing required fields in a model card
    difficulty: easy
  - id: medium
    description: Identify both missing and inadequate sections
    difficulty: medium
  - id: hard
    description: Detect subtle cross-section compliance violations
    difficulty: hard
```

---

## 5. models.py — COMPLETE IMPLEMENTATION

```python
from typing import Literal, List, Optional, Dict
from openenv.core.env_server import Action, Observation, State


class ModelCardAction(Action):
    """What the agent sends each step."""
    action_type: Literal[
        "read_section",      # Read a named section of the model card
        "check_field",       # Check whether a specific field exists
        "compare_sections",  # Cross-reference two sections for consistency
        "flag_missing",      # Flag a required field as completely absent
        "flag_inadequate",   # Flag a field that exists but is insufficient
        "flag_compliant",    # Mark a field as passing compliance
        "submit_audit"       # Submit final report and end the episode
    ]
    target: str                               # section/field name or "final"
    secondary_target: Optional[str] = None   # second section for compare_sections
    reason: str = ""                          # explanation for flag actions
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    evidence: str = ""                        # quote from model card


class ModelCardObservation(Observation):
    """
    What the agent receives after each action.
    Base class provides: done: bool, reward: Optional[float]
    """
    current_section_content: str         # Content returned by last action
    sections_available: List[str]        # All sections in this model card
    sections_reviewed: List[str]         # Sections agent has already read
    findings_count: int                  # Total issues flagged so far
    partial_score: float                 # Running compliance score (0.0–1.0)
    last_action_feedback: str            # Feedback message on last action
    steps_remaining: int                 # Steps left before episode ends
    last_action_error: Optional[str]     # None if ok, error string if action failed


class AuditState(State):
    """
    Internal server state. Not fully exposed to agent.
    Base class provides: episode_id: Optional[str], step_count: int
    """
    task_id: str                              # "easy" | "medium" | "hard"
    model_card_id: str                        # which synthetic scenario
    model_card_sections: Dict[str, str]       # full model card (all sections)
    ground_truth_issues: List[Dict]           # planted issues (hidden from agent)
    agent_findings: List[Dict]                # what agent has flagged so far
    false_positive_count: int = 0             # wrong flags
    max_steps: int = 40                       # episode boundary
    sections_reviewed: List[str] = []         # tracked here too for convenience
```

---

## 6. server/graders.py — COMPLETE IMPLEMENTATION

```python
from typing import List, Dict


def grade_easy(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Easy task: 4 completely missing required fields.
    Agent must flag each as 'flag_missing'.
    Full credit per correct flag. Small false-positive penalty.
    Score guaranteed to vary based on how many issues agent finds.
    """
    if not ground_truth:
        return 1.0

    correct = sum(
        1 for gt in ground_truth
        if any(
            f["target"] == gt["field"] and f["action_type"] == "flag_missing"
            for f in agent_findings
        )
    )

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(f["target"] == gt["field"] for gt in ground_truth)
    )

    score = (correct / len(ground_truth)) - min(0.30, false_positives * 0.05)
    return max(0.0, min(1.0, round(score, 4)))


def grade_medium(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Medium task: 5 issues, mix of missing AND inadequate.
    Correct issue + correct action_type = full credit.
    Correct issue + wrong action_type = 40% credit.
    False positives penalised.
    """
    if not ground_truth:
        return 1.0

    per_issue = 1.0 / len(ground_truth)
    score = 0.0

    for gt in ground_truth:
        matches = [f for f in agent_findings if f["target"] == gt["field"]]
        if not matches:
            continue
        best = matches[0]
        if best["action_type"] == gt["expected_action"]:
            score += per_issue           # full credit
        else:
            score += per_issue * 0.40   # partial: found the field, wrong classification

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(f["target"] == gt["field"] for gt in ground_truth)
    )

    score -= min(0.25, false_positives * 0.05)
    return max(0.0, min(1.0, round(score, 4)))


def grade_hard(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Hard task: 5 subtle cross-section violations, each with a weight.
    Agent must cross-reference sections to find them.
    Evidence bonus for citing correct text.
    """
    if not ground_truth:
        return 1.0

    score = 0.0
    for gt in ground_truth:
        weight = gt.get("weight", 0.20)
        found = any(
            f["target"] == gt["field"] and f["action_type"] == gt["expected_action"]
            for f in agent_findings
        )
        if found:
            score += weight
            # 10% evidence bonus if agent quoted the key text
            key_evidence = gt.get("key_evidence", "")
            if key_evidence and any(
                key_evidence.lower() in f.get("evidence", "").lower()
                for f in agent_findings
                if f["target"] == gt["field"]
            ):
                score += weight * 0.10

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(f["target"] == gt["field"] for gt in ground_truth)
    )

    score -= min(0.20, false_positives * 0.04)
    return max(0.0, min(1.0, round(score, 4)))
```

---

## 7. server/environment.py — COMPLETE IMPLEMENTATION

```python
import uuid
import json
from pathlib import Path
from openenv.core.env_server import Environment
from ..models import ModelCardAction, ModelCardObservation, AuditState
from .graders import grade_easy, grade_medium, grade_hard

# Load scenarios once at import time
DATA_DIR = Path(__file__).parent.parent / "data"
SCENARIOS = {
    "easy":   json.loads((DATA_DIR / "easy.json").read_text()),
    "medium": json.loads((DATA_DIR / "medium.json").read_text()),
    "hard":   json.loads((DATA_DIR / "hard.json").read_text()),
}
GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
MAX_STEPS = 40


class ModelCardAuditEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = AuditState(
            task_id="easy", model_card_id="",
            model_card_sections={}, ground_truth_issues=[],
            agent_findings=[]
        )

    def reset(self, seed=None, episode_id=None, task_id="easy", **kwargs) -> ModelCardObservation:
        """Start a completely clean episode."""
        scenario = SCENARIOS[task_id]
        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            model_card_id=scenario["id"],
            model_card_sections=scenario["sections"],
            ground_truth_issues=scenario["ground_truth"],
            agent_findings=[],
            false_positive_count=0,
            max_steps=MAX_STEPS,
            sections_reviewed=[],
        )
        return ModelCardObservation(
            done=False,
            reward=None,
            current_section_content=(
                f"=== Model Card: {scenario['title']} ===\n"
                f"You are an AI governance auditor. Review this model card for compliance.\n"
                f"Sections available: {list(scenario['sections'].keys())}"
            ),
            sections_available=list(scenario["sections"].keys()),
            sections_reviewed=[],
            findings_count=0,
            partial_score=0.0,
            last_action_feedback="Audit started. Read sections, flag issues, then submit.",
            steps_remaining=MAX_STEPS,
            last_action_error=None,
        )

    def step(self, action: ModelCardAction, timeout_s=None, **kwargs) -> ModelCardObservation:
        """
        Process one agent action.
        Server returns ModelCardObservation (framework pattern).
        create_fastapi_app() handles serialising this.
        Client (inference.py) sees StepResult with .observation, .reward, .done.
        """
        self._state.step_count += 1
        reward = 0.0
        feedback = ""
        error = None
        done = False
        content = ""

        try:
            if action.action_type == "read_section":
                if action.target not in self._state.model_card_sections:
                    error = f"Section '{action.target}' not found."
                    reward = -0.02
                    feedback = error
                elif action.target in self._state.sections_reviewed:
                    reward = -0.02
                    content = self._state.model_card_sections[action.target]
                    feedback = f"Already read '{action.target}'. Penalty applied for re-reading."
                else:
                    self._state.sections_reviewed.append(action.target)
                    reward = 0.02
                    content = self._state.model_card_sections[action.target]
                    feedback = f"Read new section '{action.target}'."

            elif action.action_type == "check_field":
                exists = action.target in self._state.model_card_sections
                content = f"Field '{action.target}': {'EXISTS' if exists else 'DOES NOT EXIST'}."
                reward = 0.01
                feedback = content

            elif action.action_type == "compare_sections":
                s1 = self._state.model_card_sections.get(action.target, "[not found]")
                s2 = self._state.model_card_sections.get(action.secondary_target or "", "[not found]")
                content = (
                    f"--- {action.target} ---\n{s1}\n\n"
                    f"--- {action.secondary_target} ---\n{s2}"
                )
                reward = 0.03
                feedback = f"Compared '{action.target}' with '{action.secondary_target}'."

            elif action.action_type in ("flag_missing", "flag_inadequate", "flag_compliant"):
                finding = {
                    "action_type": action.action_type,
                    "target": action.target,
                    "reason": action.reason,
                    "severity": action.severity,
                    "evidence": action.evidence,
                }
                if self._is_correct_finding(finding):
                    reward = 0.15
                    feedback = f"Correct: '{action.target}' is a genuine compliance issue."
                else:
                    reward = -0.04
                    self._state.false_positive_count += 1
                    feedback = f"Incorrect: '{action.target}' flagging not supported by evidence."
                self._state.agent_findings.append(finding)

            elif action.action_type == "submit_audit":
                final_score = GRADERS[self._state.task_id](
                    self._state.agent_findings,
                    self._state.ground_truth_issues,
                )
                reward = final_score * 0.40
                done = True
                content = f"AUDIT SUBMITTED. Final compliance score: {final_score:.3f}"
                feedback = content

            else:
                error = f"Unknown action_type: '{action.action_type}'"
                reward = -0.01

        except Exception as exc:
            error = str(exc)
            reward = -0.01
            feedback = f"Action error: {error}"

        if self._state.step_count >= self._state.max_steps and not done:
            reward -= 0.10
            done = True
            feedback = f"Max steps ({MAX_STEPS}) reached. Episode ended."

        return ModelCardObservation(
            done=done,
            reward=reward,
            current_section_content=content,
            sections_available=list(self._state.model_card_sections.keys()),
            sections_reviewed=list(self._state.sections_reviewed),
            findings_count=len(self._state.agent_findings),
            partial_score=self._partial_score(),
            last_action_feedback=feedback,
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
            last_action_error=error,
        )

    @property
    def state(self) -> AuditState:
        return self._state

    def close(self):
        pass  # No persistent resources

    def _is_correct_finding(self, finding: dict) -> bool:
        for gt in self._state.ground_truth_issues:
            if (finding["target"] == gt["field"] and
                    finding["action_type"] == gt["expected_action"]):
                return True
        return False

    def _partial_score(self) -> float:
        if not self._state.ground_truth_issues:
            return 1.0
        correct = sum(1 for f in self._state.agent_findings
                      if self._is_correct_finding(f))
        return round(correct / len(self._state.ground_truth_issues), 3)
```

---

## 8. server/app.py — COMPLETE IMPLEMENTATION

```python
from openenv.core.env_server import create_fastapi_app
from .environment import ModelCardAuditEnvironment

# One line. create_fastapi_app creates all endpoints automatically:
# /ws, /reset, /step, /state, /health, /web, /docs
app = create_fastapi_app(ModelCardAuditEnvironment)
```

---

## 9. THREE DATA SCENARIOS — EXACT JSON

### data/easy.json
```json
{
  "id": "easy_v1",
  "title": "BERTweet-Sentiment-Mini",
  "sections": {
    "Overview": "BERTweet-Sentiment-Mini is a lightweight binary sentiment classifier trained for positive/negative classification.",
    "Installation": "pip install transformers\n\nfrom transformers import pipeline\nclassifier = pipeline('text-classification', model='bertweet-sentiment-mini')\nresult = classifier('I love this!')\nprint(result)"
  },
  "ground_truth": [
    {"field": "Training Data",      "expected_action": "flag_missing", "weight": 0.25, "key_evidence": ""},
    {"field": "Intended Use",       "expected_action": "flag_missing", "weight": 0.25, "key_evidence": ""},
    {"field": "Evaluation Results", "expected_action": "flag_missing", "weight": 0.25, "key_evidence": ""},
    {"field": "License",            "expected_action": "flag_missing", "weight": 0.25, "key_evidence": ""}
  ]
}
```

### data/medium.json
```json
{
  "id": "medium_v1",
  "title": "MultiLingualNLI-v2",
  "sections": {
    "Overview": "MultiLingualNLI-v2 is a natural language inference model supporting 30+ languages for entailment, contradiction, and neutral classification.",
    "Training Data": "Trained on multilingual web text collected from various online sources.",
    "Intended Use": "Designed for multilingual NLI tasks across enterprise applications. Supports English, French, Spanish, German and more.",
    "Evaluation Results": "Achieves 89.3% accuracy on English SNLI benchmark. Strong performance across test sets.",
    "Bias and Limitations": "This model may reflect biases present in the training data. Users should be aware of potential issues.",
    "License": "Apache 2.0"
  },
  "ground_truth": [
    {"field": "Training Data",       "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "multilingual web text"},
    {"field": "Bias and Limitations","expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "may reflect biases"},
    {"field": "Evaluation Results",  "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "English SNLI benchmark"},
    {"field": "Environmental Impact","expected_action": "flag_missing",    "weight": 0.20, "key_evidence": ""},
    {"field": "Out-of-Scope Uses",   "expected_action": "flag_missing",    "weight": 0.20, "key_evidence": ""}
  ]
}
```

### data/hard.json
```json
{
  "id": "hard_v1",
  "title": "OpenGen-7B",
  "sections": {
    "Overview": "OpenGen-7B is a general-purpose text generation model achieving state-of-the-art performance on complex reasoning tasks. Suitable for a wide range of downstream NLP applications.",
    "Training Data": "Trained on The Pile, a diverse 825GB open-source dataset including academic papers, GitHub code, and unfiltered web crawl data from Common Crawl.",
    "Training Procedure": "Fine-tuned from meta-llama/Llama-2-7b using LoRA adapters on 8x A100 GPUs for 72 GPU-hours. Learning rate 2e-4, batch size 32.",
    "Intended Use": "Suitable for educational platforms including those serving minors. Designed for global deployment across all regions and cultures. Recommended for reasoning, summarization, and question answering.",
    "Evaluation Results": "Perplexity on WikiText-103: 12.4. BLEU-4 on WMT14 En-De: 28.7. Evaluated on standard language modelling benchmarks.",
    "Bias and Limitations": "We evaluated OpenGen-7B for gender bias using WinoBias and racial bias using StereoSet. Results show near-parity with baseline models. We recommend users apply additional safety filters for production use.",
    "Environmental Impact": "Training compute: 72 GPU-hours on NVIDIA A100 (400W TDP). Energy consumption: 72 × 400W × 1h = 28.8 kWh. CO2 equivalent: 72 × 400 × 0.233 = 3.2 kgCO2 (using US average grid intensity of 0.233 kgCO2/kWh).",
    "License": "MIT License. Free to use for commercial and non-commercial purposes.",
    "Out-of-Scope Uses": "Not intended for high-stakes autonomous decision-making without human oversight. Should not be used for generating legal or medical advice without expert review."
  },
  "ground_truth": [
    {
      "field": "License",
      "expected_action": "flag_inadequate",
      "weight": 0.25,
      "key_evidence": "Llama-2-7b",
      "description": "MIT License claim is incompatible with LLaMA 2 base model which uses Meta's custom Community License"
    },
    {
      "field": "Training Data",
      "expected_action": "flag_inadequate",
      "weight": 0.25,
      "key_evidence": "unfiltered web crawl",
      "description": "Unfiltered web crawl data combined with minor-facing intended use is a critical content safety gap"
    },
    {
      "field": "Bias and Limitations",
      "expected_action": "flag_inadequate",
      "weight": 0.20,
      "key_evidence": "global deployment",
      "description": "Only gender and racial bias evaluated, but model claims global deployment across all regions and cultures"
    },
    {
      "field": "Evaluation Results",
      "expected_action": "flag_inadequate",
      "weight": 0.20,
      "key_evidence": "state-of-the-art performance on complex reasoning",
      "description": "Overview claims reasoning SOTA but evaluation only shows perplexity and BLEU — no reasoning benchmarks"
    },
    {
      "field": "Environmental Impact",
      "expected_action": "flag_inadequate",
      "weight": 0.10,
      "key_evidence": "3.2 kgCO2",
      "description": "CO2 calculation is mathematically wrong: 72 × 0.4kW × 0.233 = 6.71 kgCO2, not 3.2 kgCO2"
    }
  ]
}
```

---

## 10. inference.py — COMPLETE IMPLEMENTATION (MUST BE IN ROOT)

```python
"""
inference.py
Baseline agent for model-card-auditor OpenEnv environment.
MUST be placed in the ROOT directory of the project.
Uses OpenAI Client library for all LLM calls.
"""
import os
import json
from openai import OpenAI
from model_card_auditor import ModelCardAuditClient, ModelCardAction

# ── Mandatory environment variables (from Additional Instructions) ─────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
HF_TOKEN     = os.environ.get("HF_TOKEN")

# ── Also support OPENAI_API_KEY (mentioned in Functional Requirements section) ─
api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")

# ── OpenAI Client (mandatory: "Participants must use OpenAI Client") ───────────
client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

TEMPERATURE  = 0.0
MAX_TOKENS   = 512
MAX_STEPS    = 25

FALLBACK_ACTION = json.dumps({
    "action_type": "submit_audit",
    "target": "final",
    "reason": "Fallback due to model call failure",
    "severity": "low",
    "evidence": ""
})

SYSTEM_PROMPT = """You are an AI governance auditor reviewing model cards for compliance.

A model card must contain these required sections:
- Training Data (what data, sources, size, languages)
- Intended Use (what the model is for, who uses it)
- Evaluation Results (metrics, benchmarks, datasets)
- License (what license governs the model)
- Environmental Impact (CO2, compute used)
- Bias and Limitations (what biases were tested and found)
- Out-of-Scope Uses (what the model should NOT be used for)

Your job: read each section, identify compliance issues, flag them, then submit your audit.

At every step, respond with ONLY a JSON object (no markdown, no explanation):
{
  "action_type": "read_section|check_field|compare_sections|flag_missing|flag_inadequate|flag_compliant|submit_audit",
  "target": "<section or field name, or 'final' for submit>",
  "secondary_target": "<second section name for compare_sections, else null>",
  "reason": "<why you are flagging this issue>",
  "severity": "low|medium|high|critical",
  "evidence": "<exact quote from the model card that supports your finding>"
}

Strategy:
1. Read all available sections first
2. Check for missing required sections
3. Compare related sections for cross-section inconsistencies
4. Flag issues found
5. Submit when confident"""


def parse_model_action(response_text: str) -> ModelCardAction:
    """Parse the model's text response into a ModelCardAction. Returns fallback on failure."""
    try:
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        return ModelCardAction(**data)
    except Exception:
        return ModelCardAction(**json.loads(FALLBACK_ACTION))


def run_task(env, task_id: str) -> float:
    """Run one complete episode. Returns the final compliance score."""
    observation = env.reset(task_id=task_id)
    history = []

    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    try:
        for step in range(MAX_STEPS):
            user_content = (
                f"Sections available: {observation.sections_available}\n"
                f"Sections reviewed:  {observation.sections_reviewed}\n"
                f"Findings so far:    {observation.findings_count}\n"
                f"Partial score:      {observation.partial_score:.3f}\n"
                f"Steps remaining:    {observation.steps_remaining}\n"
                f"Last feedback:      {observation.last_action_feedback}\n"
                f"Section content:\n{observation.current_section_content[:600]}\n\n"
                f"What is your next action? Respond with JSON only."
            )

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = FALLBACK_ACTION

            action = parse_model_action(response_text)
            print(f"Step {step + 1}: model suggested -> {action.action_type}({action.target})")

            result = env.step(action)
            observation = result.observation
            reward      = result.reward or 0.0

            error_flag = " ERROR" if observation.last_action_error else ""
            history.append(
                f"Step {step + 1}: {action.action_type}({action.target}) "
                f"-> reward {reward:+.2f}{error_flag}"
            )
            print(
                f"  Reward: {reward:+.2f} | Done: {result.done} | "
                f"Last action error: {observation.last_action_error}"
            )

            if result.done:
                print("Episode complete.")
                break
        else:
            print(f"Reached max steps ({MAX_STEPS}).")

    finally:
        env.close()

    return observation.partial_score


def main():
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Model Card Auditor — Baseline Inference")
    print(f"Model:       {MODEL_NAME}")
    print(f"Endpoint:    {API_BASE_URL}")
    print(f"Environment: {env_url}")

    scores = {}

    with ModelCardAuditClient(base_url=env_url).sync() as env:
        for task_id in ["easy", "medium", "hard"]:
            scores[task_id] = run_task(env, task_id)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id:<8}: {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<8}: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

---

## 11. client.py

The client file wraps the WebSocket/HTTP communication. Check what `openenv-core` provides for `client.py`. Look at the module 4 repo for the base client pattern. Likely:

```python
from openenv.core.env_client import OpenEnvClient
from .models import ModelCardAction, ModelCardObservation


class ModelCardAuditClient(OpenEnvClient):
    """
    Client wrapper for the ModelCardAuditEnvironment.
    Provides sync() context manager that returns an env-like object.
    """
    action_class = ModelCardAction
    observation_class = ModelCardObservation
```

If `OpenEnvClient` doesn't exist in the version available, implement a thin HTTP client:

```python
import requests
import contextlib
from .models import ModelCardAction, ModelCardObservation


class _SyncEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy", **kwargs) -> ModelCardObservation:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return ModelCardObservation(**resp.json())

    def step(self, action: ModelCardAction):
        resp = requests.post(f"{self.base_url}/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        obs = ModelCardObservation(**data["observation"])

        class StepResult:
            observation = obs
            reward = data.get("reward", 0.0)
            done = data.get("done", False)

        return StepResult()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ModelCardAuditClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url

    @contextlib.contextmanager
    def sync(self):
        env = _SyncEnv(self.base_url)
        try:
            yield env
        finally:
            env.close()
```

---

## 12. model_card_auditor/__init__.py

```python
from .models import ModelCardAction, ModelCardObservation, AuditState
from .client import ModelCardAuditClient

__all__ = ["ModelCardAction", "ModelCardObservation", "AuditState", "ModelCardAuditClient"]
```

---

## 13. server/__init__.py

```python
from .app import app

__all__ = ["app"]
```

---

## 14. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "model_card_auditor.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
```

---

## 15. requirements.txt

```
openenv-core>=0.2.1
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
openai>=1.0.0
requests>=2.31.0
```

---

## 16. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "model-card-auditor"
version = "1.0.0"
description = "OpenEnv environment for AI governance model card auditing"
requires-python = ">=3.11"
dependencies = [
    "openenv-core>=0.2.1",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "requests>=2.31.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["model_card_auditor*"]
```

---

## 17. README.md (ALL REQUIRED SECTIONS FROM RULES)

```markdown
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
pip install git+https://huggingface.co/spaces/saazbhargav/model-card-auditor
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
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="https://saazbhargav-model-card-auditor.hf.space"
python inference.py
```

## Baseline Scores

(fill in after running inference.py end-to-end)
- easy:    0.XXXX
- medium:  0.XXXX
- hard:    0.XXXX
- average: 0.XXXX
```

---

## 18. REWARD FUNCTION SUMMARY

| Action | Condition | Reward |
|--------|-----------|--------|
| read_section | New section | +0.02 |
| read_section | Already read (loop) | -0.02 |
| read_section | Section not found | -0.02 |
| check_field | Any | +0.01 |
| compare_sections | Any | +0.03 |
| flag_* | Correct finding | +0.15 |
| flag_* | False positive | -0.04 |
| submit_audit | Terminal | final_score × 0.40 |
| Any | Max steps reached | -0.10 |

Terminal reward = only 40% of max possible. Agent earns the rest through correct findings mid-episode.

---

## 19. TESTING COMMANDS

### Unit test graders (run before anything else)
```bash
python -c "
from model_card_auditor.server.graders import grade_easy, grade_medium, grade_hard

# Easy: perfect agent
gt_easy = [
    {'field': 'Training Data', 'expected_action': 'flag_missing'},
    {'field': 'Intended Use', 'expected_action': 'flag_missing'},
    {'field': 'Evaluation Results', 'expected_action': 'flag_missing'},
    {'field': 'License', 'expected_action': 'flag_missing'},
]
perfect = [{'action_type': 'flag_missing', 'target': f['field'], 'reason': '', 'severity': 'high', 'evidence': ''} for f in gt_easy]
empty = []
assert grade_easy(perfect, gt_easy) == 1.0, 'grade_easy perfect should be 1.0'
assert grade_easy(empty, gt_easy) == 0.0, 'grade_easy empty should be 0.0'
assert 0.0 < grade_easy(perfect[:2], gt_easy) < 1.0, 'grade_easy partial should be between 0 and 1'
print('Grader tests passed')
"
```

### Local server test
```bash
uvicorn model_card_auditor.server.app:app --host 0.0.0.0 --port 7860 &
sleep 2
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
```

### Docker test
```bash
docker build -t model-card-auditor .
docker run -p 7860:7860 model-card-auditor &
sleep 5
curl http://localhost:7860/health
```

### OpenEnv validation
```bash
openenv validate
```

### Full inference test
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

---

## 20. HF SPACE DEPLOYMENT

1. Create HF Space: https://huggingface.co/new-space
   - Space name: `model-card-auditor`
   - SDK: Docker
   - Visibility: Public
   - Tag: `openenv` (CRITICAL — judges filter by this tag)

2. Push code:
```bash
git remote add hf https://huggingface.co/spaces/saazbhargav/model-card-auditor
git push hf main
```

3. Verify deployment:
```bash
curl https://saazbhargav-model-card-auditor.hf.space/health
curl -X POST https://saazbhargav-model-card-auditor.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task_id": "easy"}'
```

---

## 21. PRE-SUBMISSION CHECKLIST (ALL MUST PASS)

### Automated checks (failing any = disqualified)
- [ ] `openenv validate` → passes with no errors
- [ ] `docker build -t test .` → exits 0
- [ ] `docker run -p 7860:7860 test` → server starts
- [ ] `curl localhost:7860/health` → `{"status": "healthy"}`
- [ ] `curl -X POST localhost:7860/reset` → valid JSON observation returned
- [ ] `python inference.py` → runs without error, prints 3 scores
- [ ] inference.py runtime → completes in < 20 minutes
- [ ] All 3 graders → return float in [0.0, 1.0]
- [ ] Graders with varied inputs → do NOT always return same score

### Files checklist
- [ ] `inference.py` → in ROOT directory (not in subfolder)
- [ ] `openenv.yaml` → present, valid, tasks listed, tagged "openenv"
- [ ] `Dockerfile` → present, builds cleanly
- [ ] `requirements.txt` → present, all deps listed
- [ ] `README.md` → all 7 required sections present
- [ ] `data/easy.json` → present, correct format
- [ ] `data/medium.json` → present, correct format
- [ ] `data/hard.json` → present, correct format

### Environment variables
- [ ] `API_BASE_URL` → read in inference.py
- [ ] `MODEL_NAME` → read in inference.py
- [ ] `HF_TOKEN` → read in inference.py, used as api_key
- [ ] `OPENAI_API_KEY` → read as fallback

### HF Space
- [ ] Space is PUBLIC
- [ ] Space is tagged with "openenv"
- [ ] Space URL responds to /health → 200
- [ ] Space URL responds to POST /reset → valid observation

### GitHub repo
- [ ] Repository is PUBLIC
- [ ] `inference.py` in root
- [ ] Not copied from existing environment

---

## 22. WHEN TO USE SLASH COMMANDS IN CLAUDE CODE

| Command | When to use |
|---------|-------------|
| `/memory` | Start of every new session — reload this context |
| `/clear` | Before starting a new file to clear context window |
| `/review` | After writing each major file — verify correctness before moving on |
| `/test` | After implementing graders and environment — run unit tests |
| `/diff` | After any edit to environment.py — verify you didn't break the step logic |
| `/search openenv` | If you're unsure of the correct import path from the openenv-core package |
| `/search create_fastapi_app` | Verify how app.py should call the framework function |

---

## 23. CRITICAL IMPLEMENTATION NOTES

1. **`inference.py` MUST be in the project root** — not inside `model_card_auditor/`. Judges check this.
2. **`sections_reviewed` field** — must be on `AuditState` AND passed correctly through to `ModelCardObservation`. The environment step returns `list(self._state.sections_reviewed)`.
3. **`close()` MUST exist** — inference.py calls `env.close()` in a `finally` block.
4. **All graders MUST return float in [0.0, 1.0]** — enforce with `max(0.0, min(1.0, score))` in every grader.
5. **`openenv.yaml` must have `tags: [openenv]`** — this is how judges discover submissions.
6. **HF Space must be PUBLIC** — private spaces are invisible to judges.
7. **Graders must be non-trivial** — graders that always return 1.0 or 0.0 = disqualification.
8. **The hard task CO2 math** — `72 × 0.4 × 0.233 = 6.71`, not 3.2. The model card deliberately has the wrong number. Agents that do the arithmetic will catch it.
9. **compare_sections reward is +0.03** — higher than read_section (+0.02). This incentivizes the cross-referencing needed for the hard task.
10. **False positive penalties are asymmetric** — easy/medium penalize by 0.05 per FP; hard by 0.04 per FP. This reflects that hard task is harder to get right.

---

*CLAUDE.md v1.0 — Generated 28 March 2026 from PRD v2.0*
*Author: Saaz Bhargava | saazbhargav@gmail.com*
