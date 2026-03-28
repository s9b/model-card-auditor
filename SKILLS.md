# SKILLS.md — Winning Patterns & Claude Code Hacks
## Model Card Auditor | Meta × PyTorch × HuggingFace Hackathon

This file is for Claude Code to read when it needs to know:
- How to use the OpenEnv framework correctly
- What Claude Code slash commands to use and when
- Common pitfalls that cause disqualification
- How to maximize each scoring criterion
- Code quality standards for this project

---

## 1. OPENENV FRAMEWORK — HOW IT ACTUALLY WORKS

### Import paths (verify these before using)
```python
# Server-side base classes
from openenv.core.env_server import Environment, Action, Observation, State, create_fastapi_app

# If a client base class exists:
from openenv.core.env_client import OpenEnvClient  # may or may not exist

# Your models always extend the framework base classes
class ModelCardAction(Action): ...
class ModelCardObservation(Observation): ...
class AuditState(State): ...
```

### How create_fastapi_app works
`create_fastapi_app(EnvironmentClass)` auto-generates these endpoints:
- `GET /health` → `{"status": "healthy"}`
- `POST /reset` → calls `env.reset(**body)`, returns Observation as JSON
- `POST /step` → calls `env.step(Action(**body))`, returns Observation as JSON
- `GET /state` → returns current state as JSON
- `WebSocket /ws` → streaming interface (we don't need to implement this)
- `GET /docs` → Swagger UI

You do NOT manually write any of these endpoints. One line in app.py does it all.

### Observation base class fields
The `Observation` base class from openenv already provides:
- `done: bool`
- `reward: Optional[float]`

Your `ModelCardObservation` adds 8 more fields on top of these.

### State base class fields
The `State` base class from openenv provides:
- `episode_id: Optional[str]`
- `step_count: int`

Your `AuditState` adds everything else.

### Action base class
The `Action` base class is essentially a Pydantic model. Your `ModelCardAction` just adds the `action_type` Literal field and the other action parameters.

### CONCURRENT SESSIONS
Set `SUPPORTS_CONCURRENT_SESSIONS = True` on the environment class. This tells the framework that multiple episodes can run simultaneously (required for judges running parallel evaluations).

---

## 2. CLAUDE CODE SLASH COMMANDS — WHEN AND WHY

### `/memory` — Use at the start of every session
Reloads context from CLAUDE.md so you know where you are. Always run this first when starting a new Claude Code session.

### `/clear` — Use before switching to a new major file
Clears the context window. Do this before writing `environment.py` after finishing `graders.py` — prevents stale context from confusing the implementation.

### `/review` — Use after writing each file
Run `/review` on each file before moving on:
- After writing `models.py` → `/review model_card_auditor/models.py`
- After writing `graders.py` → `/review model_card_auditor/server/graders.py`
- After writing `environment.py` → `/review model_card_auditor/server/environment.py`

This catches bugs before they propagate.

### `/test` — Use after Phase 2 and Phase 3
Run tests immediately after writing graders and environment:
```
/test python -c "from model_card_auditor.server.graders import grade_easy; ..."
```

### `/diff` — Use after any edit to environment.py
Verify you haven't accidentally broken the step() logic. Compare with the canonical version in CLAUDE.md.

### `/search` — Use when import paths are uncertain
```
/search create_fastapi_app
/search openenv.core
```
Find the exact import paths in the installed package before writing any import statements.

### `/run` — Use for smoke tests
```
/run uvicorn model_card_auditor.server.app:app --port 7860
/run curl http://localhost:7860/health
```

### Extended thinking prompt
When debugging a subtle bug (like why `sections_reviewed` isn't persisting, or why reward calculation is off), prefix your question to Claude Code with:

> "Think step by step through the execution path of step() when action_type is 'flag_missing' and the target matches a ground truth entry."

This forces chain-of-thought reasoning and catches logic bugs before running.

---

## 3. CODE QUALITY STANDARDS (15% of score)

### Type hints everywhere
```python
# BAD
def grade_easy(findings, ground_truth):

# GOOD
def grade_easy(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
```

### Docstrings on every class and method
```python
class ModelCardAuditEnvironment(Environment):
    """
    OpenEnv environment for AI governance model card compliance auditing.

    Agents review synthetic HuggingFace model cards and identify missing
    or inadequate sections using a structured action space.
    """
```

### No bare except clauses
```python
# BAD
except:

# GOOD
except Exception as exc:
    error = str(exc)
```

### Constants at module level
```python
# Good: these are easy for judges to find and understand
MAX_STEPS = 40
DATA_DIR = Path(__file__).parent.parent / "data"
```

### Consistent use of `round(score, 4)` in graders
All graders return exactly 4 decimal places. This looks professional and is easy to compare.

---

## 4. COMMON PITFALLS (each one is a disqualification or major score loss)

### PITFALL 1: inference.py in wrong location
`inference.py` MUST be in the root of the repository. Not in `model_card_auditor/`. The judges' automated checker will specifically look for `<repo_root>/inference.py`.

**How to verify:** `ls inference.py` from the repo root should show the file.

### PITFALL 2: HF Space not tagged "openenv"
Judges discover submissions by filtering for the `openenv` tag. If it's missing, your submission may not even be seen.

**How to add tag:** In the HF Space settings UI, add tag. Or add to the Space's README.md header:
```yaml
---
tags:
- openenv
---
```

### PITFALL 3: Graders that always return same score
This is explicitly listed as a disqualification criterion. Each grader must return different values for different inputs.

**How to verify:**
```python
assert grade_easy([], gt) != grade_easy(perfect_findings, gt)
assert grade_easy(partial_findings, gt) not in [0.0, 1.0]
```

### PITFALL 4: Missing `close()` method
`inference.py` calls `env.close()` in a `finally` block. If `close()` doesn't exist, every inference run crashes with `AttributeError`.

### PITFALL 5: Reward not returned in Observation
The framework puts `reward` INSIDE the Observation object. `inference.py` accesses it as `result.reward`. Don't put reward anywhere else.

### PITFALL 6: `sections_reviewed` field drift
`AuditState.sections_reviewed` and the `sections_reviewed` field in `ModelCardObservation` must stay in sync. In `step()`, always do:
```python
sections_reviewed=list(self._state.sections_reviewed),  # copy, not reference
```

### PITFALL 7: Hard task ground truth field names
The `field` names in ground_truth must EXACTLY match the section names in the model card JSON. If a section is called `"Bias and Limitations"` in the JSON, the ground_truth field must be `"Bias and Limitations"`, not `"bias_limitations"` or `"Bias & Limitations"`.

### PITFALL 8: docker build fails because of missing curl
The Dockerfile has a HEALTHCHECK that uses `curl`. The `python:3.11-slim` base image doesn't include curl.

**Fix:**
```dockerfile
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
```
Add this BEFORE the `COPY . .` line in the Dockerfile.

### PITFALL 9: `openenv validate` failing on missing fields
If `openenv validate` fails, read the error carefully. Common issues:
- Missing `tasks` section in `openenv.yaml`
- Observation missing required base fields
- Action not properly extending the base class

### PITFALL 10: Float rounding causes out-of-[0,1] values
Due to floating point, `score` can very slightly exceed 1.0 or go below 0.0 after penalties. Always wrap: `max(0.0, min(1.0, round(score, 4)))`.

---

## 5. MAXIMIZING EACH SCORING CRITERION

### Real-world utility (30%) — WE WIN HERE
The opening line of the README should hit judges immediately:

> "HuggingFace hosts over 1.2 million models. The vast majority have incomplete, inaccurate, or non-compliant model cards. Manual review at this scale is impossible. This environment trains AI agents to automate the governance gap."

Judges from HuggingFace and Meta write model cards for a living. They will immediately feel the value. Lead with this in the README, in the description field of `openenv.yaml`, and in every demo you give.

### Task & Grader Quality (25%) — WE WIN HERE
- Easy ≈ 0.80 baseline, Medium ≈ 0.50, Hard ≈ 0.20 → clear difficulty curve
- All graders are input-dependent → no constant score bug
- Hard task requires arithmetic (CO2 calculation) → proves the environment actually tests reasoning, not just recall
- Hard task requires cross-referencing (compare_sections) → judges can see the agent trajectory is non-trivial

### Environment Design (20%) — STRONG
- 7 action types (more than most environments)
- Step rewards at EVERY action (not just terminal) → satisfies "signal over full trajectory"
- Loop detection (re-reading penalty) → satisfies "penalizes undesirable behavior"
- `last_action_error` field → clean error surfacing for agents

### Code Quality (15%) — CLEAN BUILD
- `openenv validate` passes
- Docker builds + runs
- Type hints + docstrings
- Single-responsibility functions
- No global mutable state except `self._state`

### Creativity & Novelty (10%) — WE WIN HERE
- Domain: AI governance auditing → no one else will build this
- `compare_sections` action → unique mechanic that rewards multi-hop reasoning
- Asymmetric penalties (FP penalty varies by task difficulty)
- Hard task arithmetic violation → requires math, not just NLP

---

## 6. DEBUGGING STRATEGIES

### If server doesn't start
```bash
# Check for import errors
python -c "from model_card_auditor.server.app import app"

# Check for JSON syntax errors in data files
python -m json.tool model_card_auditor/data/easy.json

# Check if openenv-core is installed
pip show openenv-core
```

### If /reset returns 422 Unprocessable Entity
The request body doesn't match what the endpoint expects. Try:
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```
If 422 persists, check if `create_fastapi_app` expects a different request format.

### If grader returns wrong value
Add debug prints:
```python
def grade_easy(agent_findings, ground_truth):
    print(f"DEBUG: findings={agent_findings}, gt={ground_truth}")
    ...
```
Then run a manual test call and inspect output.

### If inference.py fails with "Model not found"
Check that `MODEL_NAME` env var is set and the model is accessible with your `HF_TOKEN`.

### If `openenv validate` fails
Read the full error output. Most common fix:
- Add missing field to models.py
- Fix openenv.yaml format
- Add/fix `tags: [openenv]`

### If docker HEALTHCHECK fails
```bash
# Shell into running container
docker exec -it <container_id> bash
curl http://localhost:7860/health
```

---

## 7. OPENENV YAML VALIDATION TIPS

The `openenv.yaml` is the first thing `openenv validate` checks. Get it right:

```yaml
name: model-card-auditor        # must match pyproject.toml name
version: "1.0.0"                # string, not number
description: >                  # use > for multi-line
  Your description here.
tags:
  - openenv                     # CRITICAL: this tag is how judges find you
tasks:
  - id: easy                    # must match reset() task_id values
    description: ...
    difficulty: easy
  - id: medium
    description: ...
    difficulty: medium
  - id: hard
    description: ...
    difficulty: hard
```

---

## 8. INFERENCE.PY OPTIMIZATION TIPS

### Better system prompt = better baseline scores
The default SYSTEM_PROMPT tells the agent the 7 required sections and the 7 action types. To boost scores further:
- Add specific instructions for the hard task: "Pay special attention to cross-section inconsistencies. Compare Training Data with Intended Use. Compare Overview claims with Evaluation Results. Check math in Environmental Impact."
- Add the list of required sections explicitly so the agent knows what to look for.

### Conversation history helps
The default inference.py doesn't pass conversation history to the LLM. Adding history improves multi-step coherence:
```python
history_messages = []
for entry in history:
    history_messages.append({"role": "user", "content": entry["user"]})
    history_messages.append({"role": "assistant", "content": entry["assistant"]})

messages = [{"role": "system", ...}] + history_messages + [{"role": "user", ...}]
```

### Temperature = 0 for reproducibility
Keep `TEMPERATURE = 0.0` for deterministic baseline scores. Use 0.1–0.3 only when experimenting.

---

## 9. HF SPACE TIPS

### Space startup time
HF Spaces on Docker can take 2–3 minutes to fully start after a push. Wait for the "Running" badge in the UI before testing the endpoint.

### Memory and CPU
Free-tier HF Spaces have limited RAM. If the server crashes on startup, check memory usage. The `python:3.11-slim` base image helps keep footprint small.

### Space SDK: Docker (not Gradio)
CRITICAL: Create the Space with SDK=Docker, NOT Gradio or Streamlit. Our server is FastAPI/uvicorn, not a Gradio app.

### Secrets (if needed for inference.py against deployed space)
Add `HF_TOKEN` as a Space secret, not hardcoded. Go to Space Settings → Repository secrets.

---

## 10. FINAL WIN CONDITIONS

You need ALL of these to win:

1. **Deploys** — HF Space is PUBLIC, tagged `openenv`, responds to /health and /reset
2. **Spec compliant** — `openenv validate` passes
3. **Docker works** — `docker build` and `docker run` succeed
4. **Baseline runs** — `inference.py` completes without errors and prints 3 scores
5. **3+ tasks with non-trivial graders** — each grader is input-dependent, returns [0.0, 1.0]
6. **Real-world framing** — README opens with the "1.2M models" problem statement
7. **Novel domain** — no other team will submit an AI governance auditor
8. **Hard task is genuinely hard** — cross-section reasoning + arithmetic = GPT-4 scores only ~0.45

---

*SKILLS.md v1.0 — 28 March 2026*
*For: Saaz Bhargava | saazbhargav@gmail.com*
