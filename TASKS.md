# TASKS.md — Model Card Auditor Build Plan
## Meta × PyTorch × HuggingFace Hackathon

**Deadline:** 8 April 2026, 11:59 PM IST
**Today:** 28 March 2026 — 11 days left
**Strategy:** One-shot build, then polish + deploy

---

## HOW TO USE THIS FILE

- Check off tasks as you complete them: change `[ ]` to `[x]`
- If you hit a blocker, add a note under the task
- Never skip a test gate — failing a gate means the next phase is built on broken ground
- Update baseline scores in README once inference.py runs end-to-end

---

## PHASE 0 — ENVIRONMENT SETUP (before writing any code)

- [ ] Create GitHub repo: `model-card-auditor` (PUBLIC)
- [ ] Clone repo locally, set up Python 3.11 venv
- [ ] Install base deps: `pip install openenv-core fastapi uvicorn pydantic openai requests`
- [ ] Verify openenv-core is installed: `python -c "from openenv.core.env_server import Environment, Action, Observation, State, create_fastapi_app; print('OK')"`
- [ ] If import fails, check exact import paths: `python -c "import openenv; print(openenv.__file__)"` then inspect the module
- [ ] Create the full directory structure:
  ```
  model-card-auditor/
  ├── model_card_auditor/
  │   ├── __init__.py
  │   ├── models.py
  │   ├── client.py
  │   ├── data/
  │   │   ├── easy.json
  │   │   ├── medium.json
  │   │   └── hard.json
  │   └── server/
  │       ├── __init__.py
  │       ├── environment.py
  │       ├── graders.py
  │       └── app.py
  ├── inference.py        ← ROOT level
  ├── openenv.yaml
  ├── Dockerfile
  ├── requirements.txt
  ├── pyproject.toml
  └── README.md
  ```

**🚦 Phase 0 Gate:** `python -c "from openenv.core.env_server import Environment"` exits 0.

---

## PHASE 1 — DATA FILES (Day 1) — NO CODE YET

Write the 3 JSON scenario files. All ground truth is hardcoded here.

### 1.1 — data/easy.json
- [ ] Create `model_card_auditor/data/easy.json`
- [ ] Verify: only 2 sections present (`Overview`, `Installation`)
- [ ] Verify: exactly 4 ground_truth entries, all `flag_missing`
- [ ] Verify: fields are `Training Data`, `Intended Use`, `Evaluation Results`, `License`
- [ ] Verify: JSON is valid (`python -m json.tool data/easy.json`)

### 1.2 — data/medium.json
- [ ] Create `model_card_auditor/data/medium.json`
- [ ] Verify: 6 sections present (includes all EXCEPT `Environmental Impact` and `Out-of-Scope Uses`)
- [ ] Verify: exactly 5 ground_truth entries
- [ ] Verify: mix of `flag_inadequate` (3) and `flag_missing` (2)
- [ ] Verify: `Training Data`, `Bias and Limitations`, `Evaluation Results` → `flag_inadequate`
- [ ] Verify: `Environmental Impact`, `Out-of-Scope Uses` → `flag_missing`
- [ ] Verify: JSON is valid

### 1.3 — data/hard.json
- [ ] Create `model_card_auditor/data/hard.json`
- [ ] Verify: ALL 9 sections present (full model card that looks complete at first glance)
- [ ] Verify: exactly 5 ground_truth entries, all `flag_inadequate`
- [ ] Verify the 5 issues are planted:
  - [ ] `License` says MIT but base model is Llama-2 (Community License, not MIT-compatible)
  - [ ] `Training Data` says "unfiltered web crawl" + `Intended Use` says "serving minors" — content safety gap
  - [ ] `Bias and Limitations` only covers gender + racial bias but `Intended Use` claims global deployment
  - [ ] `Evaluation Results` shows perplexity + BLEU only but `Overview` claims reasoning SOTA
  - [ ] `Environmental Impact` CO2 math is wrong: says 3.2 kgCO2, correct is 6.71 kgCO2
- [ ] Verify: JSON is valid

**🚦 Phase 1 Gate:** `python -m json.tool model_card_auditor/data/easy.json model_card_auditor/data/medium.json model_card_auditor/data/hard.json` — all 3 exit 0.

---

## PHASE 2 — PYDANTIC MODELS + GRADERS (Day 2)

### 2.1 — models.py
- [ ] Create `model_card_auditor/models.py`
- [ ] `ModelCardAction` extends `Action` from openenv — includes `action_type`, `target`, `secondary_target`, `reason`, `severity`, `evidence`
- [ ] `ModelCardObservation` extends `Observation` from openenv — includes all 8 fields
- [ ] `AuditState` extends `State` from openenv — includes all state fields + `sections_reviewed: List[str] = []`
- [ ] Verify models import: `python -c "from model_card_auditor.models import ModelCardAction, ModelCardObservation, AuditState; print('OK')"`

### 2.2 — graders.py
- [ ] Create `model_card_auditor/server/graders.py`
- [ ] Implement `grade_easy()` — 4 issues, all flag_missing, FP penalty 0.05/flag
- [ ] Implement `grade_medium()` — 5 issues, mixed, partial credit (40%) for correct target + wrong action_type
- [ ] Implement `grade_hard()` — 5 weighted issues, evidence bonus (+10% of weight), FP penalty 0.04/flag
- [ ] ALL graders: wrap in `max(0.0, min(1.0, round(score, 4)))`

### 2.3 — Unit test graders
- [ ] Perfect agent → grade_easy returns 1.0
- [ ] Empty agent → grade_easy returns 0.0
- [ ] Partial agent (2/4 correct) → grade_easy returns ~0.5
- [ ] Grader with heavy FPs → never goes below 0.0 (clamped)
- [ ] grade_medium: correct target + wrong action_type → gets 40% credit
- [ ] grade_hard: correct finding with evidence keyword → gets +10% bonus
- [ ] grade_hard: correct finding without evidence → no bonus

**🚦 Phase 2 Gate:** All grader unit tests pass. No grader ever returns value outside [0.0, 1.0].

---

## PHASE 3 — ENVIRONMENT SERVER (Day 3)

### 3.1 — server/environment.py
- [ ] Create `model_card_auditor/server/environment.py`
- [ ] `reset()` — creates fresh AuditState with new episode_id, clears all findings/sections_reviewed
- [ ] `step()` — handles all 7 action types with correct reward logic
- [ ] `state` property — returns `self._state`
- [ ] `close()` — exists and does nothing (required by inference.py finally block)
- [ ] `_is_correct_finding()` helper — matches target + action_type against ground_truth
- [ ] `_partial_score()` helper — correct/total ground_truth issues

### 3.2 — Verify step() reward logic
- [ ] `read_section` new section → +0.02 reward, section added to `sections_reviewed`
- [ ] `read_section` already-read section → -0.02 reward (loop penalty)
- [ ] `read_section` nonexistent section → -0.02 reward, `last_action_error` is set
- [ ] `check_field` any target → +0.01 reward
- [ ] `compare_sections` → +0.03 reward, content shows both sections
- [ ] `flag_missing` correct → +0.15 reward, finding added
- [ ] `flag_missing` wrong (FP) → -0.04 reward, `false_positive_count` incremented
- [ ] `submit_audit` → `done=True`, `reward = final_score * 0.40`
- [ ] max_steps reached → `done=True`, reward -= 0.10

### 3.3 — server/app.py
- [ ] Create `model_card_auditor/server/app.py`
- [ ] One line: `app = create_fastapi_app(ModelCardAuditEnvironment)`

### 3.4 — Local server smoke test
- [ ] `uvicorn model_card_auditor.server.app:app --port 7860` starts without errors
- [ ] `curl http://localhost:7860/health` → `{"status": "healthy"}`
- [ ] `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'` → valid JSON with all observation fields
- [ ] `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "medium"}'` → valid JSON
- [ ] `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "hard"}'` → valid JSON

**🚦 Phase 3 Gate:** Server starts, /health returns 200, /reset for all 3 tasks returns valid observations.

---

## PHASE 4 — CLIENT + INFERENCE SCRIPT (Day 4)

### 4.1 — client.py
- [ ] Create `model_card_auditor/client.py`
- [ ] Check if `openenv.core.env_client` has a base client — if yes, extend it
- [ ] If no base client: implement `_SyncEnv` + `ModelCardAuditClient` with `sync()` context manager
- [ ] `reset(task_id)` sends POST to `/reset`
- [ ] `step(action)` sends POST to `/step`, returns object with `.observation`, `.reward`, `.done`
- [ ] `close()` exists (called in finally block)

### 4.2 — model_card_auditor/__init__.py
- [ ] Export `ModelCardAction`, `ModelCardObservation`, `AuditState`, `ModelCardAuditClient`

### 4.3 — inference.py (ROOT LEVEL)
- [ ] Create `inference.py` in the project ROOT (not inside any subdirectory)
- [ ] Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment
- [ ] Falls back to `OPENAI_API_KEY` if `HF_TOKEN` not set
- [ ] Creates `OpenAI(api_key=api_key, base_url=API_BASE_URL)` client
- [ ] `SYSTEM_PROMPT` explains all 7 required sections and all 7 action types
- [ ] `parse_model_action()` handles JSON with/without markdown code fences, returns fallback on failure
- [ ] `run_task()` loops up to `MAX_STEPS`, calls LLM, takes step, handles `result.done`
- [ ] `main()` runs all 3 tasks and prints scores table
- [ ] `env.close()` called in `finally` block

### 4.4 — End-to-end test (local server must be running)
- [ ] Set env vars (use a real HF token or a test endpoint)
- [ ] `python inference.py` completes without Python errors
- [ ] Prints 3 scores: easy, medium, hard
- [ ] Each score is a float between 0.0 and 1.0
- [ ] Total runtime < 20 minutes

**🚦 Phase 4 Gate:** `python inference.py` runs end-to-end and prints 3 valid scores.

---

## PHASE 5 — PACKAGING + VALIDATION (Day 5)

### 5.1 — openenv.yaml
- [ ] Create `openenv.yaml` in project root
- [ ] Has `name`, `version`, `description`, `tags: [openenv]`, `tasks` with all 3 task IDs
- [ ] `openenv validate` passes — CRITICAL

### 5.2 — pyproject.toml
- [ ] Create `pyproject.toml` with correct package name, version, dependencies
- [ ] `pip install -e .` works from scratch in a fresh virtualenv

### 5.3 — requirements.txt
- [ ] Lists all deps with minimum versions
- [ ] Includes: `openenv-core`, `fastapi`, `uvicorn[standard]`, `pydantic`, `openai`, `requests`

### 5.4 — Dockerfile
- [ ] Builds: `docker build -t model-card-auditor .` exits 0
- [ ] Runs: `docker run -p 7860:7860 model-card-auditor` starts server
- [ ] Health check: `curl localhost:7860/health` returns 200 from inside docker
- [ ] Reset check: `curl -X POST localhost:7860/reset -d '{"task_id":"easy"}'` returns valid observation

### 5.5 — README.md
- [ ] Has all 7 required sections: Description & Motivation, Action Space, Observation Space, Tasks, Setup, Local Run, Docker, Baseline Inference, Baseline Scores
- [ ] Baseline Scores filled in with real numbers from Phase 4 end-to-end test

### 5.6 — openenv validate
- [ ] `openenv validate` passes with zero errors
- [ ] If it fails: read error message, fix the specific issue, re-run

**🚦 Phase 5 Gate:** `openenv validate` passes. `docker build` exits 0. README complete with real scores.

---

## PHASE 6 — DEPLOYMENT (Day 6)

### 6.1 — HF Space setup
- [ ] Go to https://huggingface.co/new-space
- [ ] Space SDK: **Docker**
- [ ] Space name: `model-card-auditor`
- [ ] Visibility: **Public** (CRITICAL)
- [ ] Add tag: `openenv` (CRITICAL — judges filter by this)

### 6.2 — Push to HF Space
- [ ] `git remote add hf https://huggingface.co/spaces/saazbhargav/model-card-auditor`
- [ ] `git push hf main`
- [ ] Monitor build logs in HF Space UI until "Running" status

### 6.3 — Verify deployed Space
- [ ] `curl https://saazbhargav-model-card-auditor.hf.space/health` → 200
- [ ] `curl -X POST https://saazbhargav-model-card-auditor.hf.space/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'` → valid observation
- [ ] Run inference.py against deployed space: `ENV_URL=https://... python inference.py`

### 6.4 — GitHub repo
- [ ] Push all code to GitHub public repo
- [ ] Verify `inference.py` is visible in the root of the GitHub repo

### 6.5 — Final submission
- [ ] Go to hackathon submission form
- [ ] Submit GitHub repo URL + HF Space URL
- [ ] Screenshot confirmation

**🚦 Phase 6 Gate:** HF Space URL responds to /health and /reset. Submission confirmed.**

---

## ONGOING — POLISH + RESUBMISSION

Since you can resubmit until the deadline, use remaining time to:

- [ ] Run inference.py with a stronger model (GPT-4o or Llama-3.1-70B) — record scores
- [ ] Improve SYSTEM_PROMPT in inference.py to boost baseline scores
- [ ] Add unit test file `tests/test_graders.py` for code quality signal
- [ ] Add `tests/test_environment.py` for environment unit tests
- [ ] Add type hints and docstrings everywhere (improves code quality score)
- [ ] Consider adding a 4th bonus task scenario (novelty points)

---

## SCORE TRACKER

| Run | Model | Easy | Medium | Hard | Average |
|-----|-------|------|--------|------|---------|
| Baseline | tbd | tbd | tbd | tbd | tbd |

Update this table after every inference.py run.

---

*TASKS.md v1.0 — 28 March 2026*
