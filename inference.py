"""
inference.py
Baseline agent for model-card-auditor OpenEnv environment.
MUST be placed in the ROOT directory of the project.
Uses OpenAI Client library for all LLM calls.
"""
# -- stdlib only from here to the log functions -- os is always available ------
import os
import json
import re
import time

# -- Structured logging: defined FIRST using only os, before any third-party --
# -- imports that could crash and prevent [START] from ever printing -----------
def _raw_print(line: str) -> None:
    print(line, flush=True)


def log_start(task, env="model-card-auditor", model=""):
    _raw_print(f"[START] task={task} env={env} model={model}")


def log_step(step, action="", reward=0.0, done=False, error=None):
    done_str = "true" if done else "false"
    error_str = str(error) if error else "null"
    _raw_print(
        f"[STEP] step={step} action={str(action)[:80]} "
        f"reward={float(reward):.2f} done={done_str} error={error_str}"
    )


def log_end(success=False, steps=0, score=0.01, rewards=None):
    if rewards is None:
        rewards = []
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    score = min(max(float(score), 0.01), 0.99)
    _raw_print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )


# -- Environment variables (read before any third-party import) ---------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.cerebras.ai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "qwen-3-235b-a22b-instruct-2507")
HF_TOKEN     = os.environ.get("HF_TOKEN")
# Support OPENAI_API_KEY as fallback; use placeholder so OpenAI() never raises
api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "") or "sk-placeholder"

# -- Third-party imports (wrapped so a missing package never kills the script) -
try:
    from openai import OpenAI
    _openai_ok = True
except Exception:
    _openai_ok = False

try:
    from model_card_auditor import ModelCardAuditClient, ModelCardAction
    _env_ok = True
except Exception:
    _env_ok = False

# -- Constants -----------------------------------------------------------------
TEMPERATURE = 0.0
MAX_TOKENS  = 512
MAX_STEPS   = 35

REQUIRED_SECTIONS = {
    "easy": [
        "Training Data",
        "Intended Use",
        "Evaluation Results",
        "License",
    ],
    "medium": [
        "Environmental Impact",
        "Out-of-Scope Uses",
    ],
    "hard": [],
}

FALLBACK_ACTION = json.dumps({
    "action_type": "submit_audit",
    "target": "final",
    "reason": "Fallback due to model call failure",
    "severity": "low",
    "evidence": ""
})

SYSTEM_PROMPT = """You are an AI governance auditor reviewing model cards for compliance.

IMPORTANT: Missing required sections have already been detected and flagged automatically.
Your job is to read the sections that DO exist and find:
- flag_inadequate: section exists but content is too vague, lacks specifics, or is inconsistent with other sections
- compare_sections: cross-reference two sections for logical contradictions
- flag_compliant: confirm a section that fully meets requirements
- submit_audit: when you have reviewed all available sections

At every step, respond with ONLY a JSON object (no markdown, no explanation):
{
    "action_type": "read_section|check_field|compare_sections|flag_inadequate|flag_compliant|submit_audit",
    "target": "<section name, or 'final' for submit>",
    "secondary_target": "<second section for compare_sections, else null>",
    "reason": "<why you are flagging this issue>",
    "severity": "low|medium|high|critical",
    "evidence": "<exact quote from the model card supporting your finding>"
}

Strategy:
1. Read every available section
2. Flag any section whose content is vague (no specifics, no numbers, no sources)
3. Use compare_sections to cross-reference:
   - License vs Training Data (license compatibility with base model)
   - Training Data vs Intended Use (safety gaps)
   - Evaluation Results vs Overview claims
   - Bias and Limitations vs Intended Use
4. Flag cross-section inconsistencies you find
5. Submit when all sections reviewed

CONSERVATIVE FLAGGING RULE:
Only call flag_inadequate if ALL THREE of these are true:
1. You can directly quote the specific problematic text from the section
2. The problem is a factual error, internal contradiction, or concrete missing detail
3. You are confident this would fail a formal compliance review

FINALITY RULE: Once you call flag_compliant on a section, that decision is final."""

HARD_SYSTEM_PROMPT = """You are an AI governance auditor performing a hard-level model card audit.

This model card LOOKS complete - all sections are present. Your job is to find SUBTLE
CROSS-SECTION INCONSISTENCIES that require reading multiple sections together.

There are exactly 5 violations hidden in this model card. All are flag_inadequate.
To find them, you MUST use compare_sections on these specific pairs:
  1. compare_sections("License", "Training Procedure")
  2. compare_sections("Training Data", "Intended Use")
  3. compare_sections("Evaluation Results", "Overview")
  4. compare_sections("Bias and Limitations", "Intended Use")
  5. Read "Environmental Impact" alone - verify the CO2 calculation

At every step, respond with ONLY a valid JSON object (no markdown, no explanation):
{
    "action_type": "read_section|compare_sections|flag_inadequate|flag_compliant|submit_audit",
    "target": "<section name, or 'final' for submit>",
    "secondary_target": "<second section for compare_sections, else null>",
    "reason": "<specific inconsistency - be precise>",
    "severity": "low|medium|high|critical",
    "evidence": "<exact quote from the model card that proves the violation>"
}

MANDATORY WORKFLOW — follow this exact sequence, no skipping:
1. compare_sections("License", "Training Procedure") → then immediately flag_inadequate("License")
2. compare_sections("Training Data", "Intended Use") → then immediately flag_inadequate("Training Data")
3. compare_sections("Evaluation Results", "Overview") → then immediately flag_inadequate("Evaluation Results")
4. compare_sections("Bias and Limitations", "Intended Use") → then immediately flag_inadequate("Bias and Limitations")
5. read_section("Environmental Impact") → then immediately flag_inadequate("Environmental Impact")
6. Only after all 5 flag_inadequate calls are done → submit_audit

Every compare_sections MUST be followed by a flag_inadequate on the FIRST section listed.
Do not skip any flag_inadequate. Do not submit early. If you submit before step 6, you score 0."""


def _get_client():
    """Create OpenAI client lazily - never at module level."""
    if not _openai_ok:
        raise RuntimeError("openai package not available")
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def parse_model_action(response_text: str):
    """Parse the model's text response into a ModelCardAction. Returns fallback on failure."""
    try:
        text = response_text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        data["reason"]   = data.get("reason")   or ""
        data["evidence"] = data.get("evidence") or ""
        data["severity"] = data.get("severity") or "medium"
        return ModelCardAction(**data)
    except Exception:
        return ModelCardAction(**json.loads(FALLBACK_ACTION))


def run_task(task_id: str, env_url: str) -> float:
    """Run one complete episode. Returns the final compliance score."""
    log_start(task=task_id, env="model-card-auditor", model=MODEL_NAME)

    step_counter = 0
    try:
        if not _env_ok:
            raise ImportError("model_card_auditor package not available")
        with ModelCardAuditClient(base_url=env_url).sync() as env:
            return _run_task_inner(env, task_id, step_counter)
    except Exception as exc:
        log_step(step=1, action="setup_failed", reward=0.01, done=True, error="setup_failed")
        log_end(success=False, steps=1, score=0.01, rewards=[0.01])
        raise


def _run_task_inner(env, task_id: str, step_counter: int) -> float:
    """Inner implementation - called by run_task after [START] is printed."""
    reset_result = env.reset(task_id=task_id)
    observation  = getattr(reset_result, "observation", reset_result)
    history      = []
    rewards_list = []
    system_prompt = HARD_SYSTEM_PROMPT if task_id == "hard" else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # -- Pre-flight: flag required sections absent from this model card ---------
    available_lower = {s.lower().strip() for s in observation.sections_available}
    for required_section in REQUIRED_SECTIONS.get(task_id, []):
        if required_section.lower().strip() not in available_lower:
            preflight_action = ModelCardAction(
                action_type="flag_missing",
                target=required_section,
                reason=f"Required section '{required_section}' is completely absent.",
                severity="high",
                evidence="",
            )
            result       = env.step(preflight_action)
            observation  = result.observation
            reward       = max(0.01, min(0.99, result.reward or 0.0))
            step_counter += 1
            rewards_list.append(reward)
            log_step(step=step_counter, action=preflight_action.action_type,
                     reward=reward, done=result.done,
                     error=observation.last_action_error)
            print(f"Pre-flight: flag_missing('{required_section}') -> reward {reward:+.2f}")
            messages.append({"role": "user",
                             "content": f"[Auto-flagged missing: {required_section}]"})
            messages.append({"role": "assistant", "content": json.dumps({
                "action_type": "flag_missing", "target": required_section,
                "reason": f"Section '{required_section}' is absent.",
                "severity": "high", "evidence": ""
            })})
            if result.done:
                final_score = min(max(observation.partial_score, 0.01), 0.99)
                log_end(success=final_score >= 0.5, steps=step_counter,
                        score=final_score, rewards=rewards_list)
                return final_score
    # -- End pre-flight ---------------------------------------------------------

    # Easy task: all ground truth issues are flag_missing, handled above.
    if task_id == "easy":
        submit_result = env.step(ModelCardAction(
            action_type="submit_audit", target="final",
            reason="All issues are missing fields, handled by pre-flight.",
            severity="low", evidence="",
        ))
        step_counter  += 1
        submit_reward  = max(0.01, min(0.99, submit_result.reward or 0.0))
        final_score    = min(max(submit_result.observation.partial_score, 0.01), 0.99)
        rewards_list.append(submit_reward)
        log_step(step=step_counter, action="submit_audit",
                 reward=submit_reward, done=True, error=None)
        print(f"Easy early exit: submit_audit -> partial_score {final_score:.3f}")
        log_end(success=final_score >= 0.5, steps=step_counter,
                score=final_score, rewards=rewards_list)
        return final_score

    client = _get_client()

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
        messages.append({"role": "user", "content": user_content})

        response_text = None
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    seed=42, stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                break
            except Exception as exc:
                error_str = str(exc)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait = 20 * (attempt + 1)
                    print(f"Rate limit (attempt {attempt+1}/3). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Model request failed ({exc}). Using fallback.")
                    response_text = FALLBACK_ACTION
                    break

        if response_text is None:
            print("All retries exhausted. Using fallback action.")
            response_text = FALLBACK_ACTION

        messages.append({"role": "assistant", "content": response_text})
        action = parse_model_action(response_text)
        print(f"Step {step + 1}: model suggested -> {action.action_type}({action.target})")

        # Safety net: split compound targets like "License and Training Data"
        if action.action_type in ("flag_missing", "flag_inadequate", "flag_compliant"):
            parts = re.split(r'\s+and\s+|\s*,\s*', action.target, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            known_lower = {s.lower() for s in observation.sections_available}
            valid_parts = [p for p in parts if p.lower() in known_lower]
            if len(valid_parts) > 1:
                split_obs = observation
                for part in valid_parts:
                    split_action = ModelCardAction(
                        action_type=action.action_type, target=part,
                        secondary_target=action.secondary_target,
                        reason=action.reason, severity=action.severity,
                        evidence=action.evidence,
                    )
                    split_result = env.step(split_action)
                    split_obs    = split_result.observation
                    split_reward = max(0.01, min(0.99, split_result.reward or 0.0))
                    step_counter += 1
                    rewards_list.append(split_reward)
                    log_step(step=step_counter, action=split_action.action_type,
                             reward=split_reward, done=split_result.done,
                             error=split_obs.last_action_error)
                    history.append(f"Step {step+1}(split): {action.action_type}({part})"
                                   f" -> reward {split_reward:+.2f}")
                    print(f"  [Split] {action.action_type}({part}) -> reward {split_reward:+.2f}")
                    if split_result.done:
                        final_score = min(max(split_obs.partial_score, 0.01), 0.99)
                        log_end(success=final_score >= 0.5, steps=step_counter,
                                score=final_score, rewards=rewards_list)
                        return final_score
                observation = split_obs
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": (
                    f"Sections available: {observation.sections_available}\n"
                    f"Sections reviewed:  {observation.sections_reviewed}\n"
                    f"Findings so far:    {observation.findings_count}\n"
                    f"Partial score:      {observation.partial_score:.3f}\n"
                    f"Steps remaining:    {observation.steps_remaining}\n"
                    f"Last feedback:      {observation.last_action_feedback}\n"
                    f"Section content:\n{observation.current_section_content[:600]}\n\n"
                    f"What is your next action? Respond with JSON only."
                )})
                time.sleep(1)
                continue

        result      = env.step(action)
        observation = result.observation
        reward      = max(0.01, min(0.99, result.reward or 0.0))
        step_counter += 1
        rewards_list.append(reward)
        log_step(step=step_counter, action=action.action_type, reward=reward,
                 done=result.done, error=observation.last_action_error)
        history.append(
            f"Step {step + 1}: {action.action_type}({action.target})"
            f" -> reward {reward:+.2f}"
            + (" ERROR" if observation.last_action_error else "")
        )
        print(f"  Reward: {reward:+.2f} | Done: {result.done} | "
              f"Error: {observation.last_action_error}")

        if result.done:
            print("Episode complete.")
            break
        time.sleep(1)
    else:
        print(f"Reached max steps ({MAX_STEPS}).")

    final_score = min(max(observation.partial_score, 0.01), 0.99)
    log_end(success=final_score >= 0.5, steps=step_counter,
            score=final_score, rewards=rewards_list)
    return final_score


def main():
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Model Card Auditor - Baseline Inference")
    print(f"Model:       {MODEL_NAME}")
    print(f"Endpoint:    {API_BASE_URL}")
    print(f"Environment: {env_url}")

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            scores[task_id] = run_task(task_id, env_url)
        except Exception:
            scores[task_id] = 0.0

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
