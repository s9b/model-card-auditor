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
MAX_STEPS    = 35

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
    "hard": [],  # all 9 sections present — pre-flight does nothing for hard
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
   - Training Data vs Intended Use (safety gaps — e.g. unfiltered data + serving minors)
   - Evaluation Results vs Overview claims (does evidence support stated capabilities?)
   - Bias and Limitations vs Intended Use (is bias coverage adequate for deployment scope?)
4. Flag cross-section inconsistencies you find
5. Submit when all sections reviewed"""

HARD_SYSTEM_PROMPT = """You are an AI governance auditor performing a hard-level model card audit.

This model card LOOKS complete — all sections are present. Your job is to find SUBTLE
CROSS-SECTION INCONSISTENCIES that require reading multiple sections together.

There are exactly 5 violations hidden in this model card. All are flag_inadequate.
To find them, you MUST use compare_sections on these specific pairs:
  1. compare_sections("License", "Training Procedure")
     → Look for: is the stated license compatible with the base model's actual license?
  2. compare_sections("Training Data", "Intended Use")
     → Look for: does the training data pose safety risks for the stated deployment audience?
  3. compare_sections("Evaluation Results", "Overview")
     → Look for: do the evaluation metrics actually prove the capability claims?
  4. compare_sections("Bias and Limitations", "Intended Use")
     → Look for: does the bias testing coverage match the deployment scope claimed?
  5. Read "Environmental Impact" alone
     → Look for: verify the CO₂ figure by checking: GPU_hours × power_kW × carbon_intensity
        (a correct calculation typically yields a number different from what is stated)

Strategy:
  Step 1: Read all sections to build context
  Step 2: compare_sections for each of the 4 pairs above
  Step 3: flag_inadequate for each inconsistency found — include exact evidence quotes
  Step 4: submit_audit when all sections reviewed

At every step, respond with ONLY a valid JSON object (no markdown, no explanation):
{
    "action_type": "read_section|compare_sections|flag_inadequate|flag_compliant|submit_audit",
    "target": "<section name, or 'final' for submit>",
    "secondary_target": "<second section for compare_sections, else null>",
    "reason": "<specific inconsistency — be precise>",
    "severity": "low|medium|high|critical",
    "evidence": "<exact quote from the model card that proves the violation>"
}"""


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
        # Coerce null values to valid defaults so Pydantic doesn't reject the action
        data["reason"]   = data.get("reason")   or ""
        data["evidence"] = data.get("evidence") or ""
        data["severity"] = data.get("severity") or "medium"
        return ModelCardAction(**data)
    except Exception:
        return ModelCardAction(**json.loads(FALLBACK_ACTION))


def run_task(env, task_id: str) -> float:
    """Run one complete episode. Returns the final compliance score."""
    reset_result = env.reset(task_id=task_id)
    # EnvClient.reset() returns a StepResult; extract the observation
    observation = getattr(reset_result, "observation", reset_result)
    history = []
    # Accumulate conversation history so the model sees its prior actions
    system_prompt = HARD_SYSTEM_PROMPT if task_id == "hard" else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # ── Pre-flight: flag required sections absent from this model card ─────────
    available_lower = {s.lower().strip() for s in observation.sections_available}
    for required_section in REQUIRED_SECTIONS.get(task_id, []):
        if required_section.lower().strip() not in available_lower:
            preflight_action = ModelCardAction(
                action_type="flag_missing",
                target=required_section,
                reason=f"Required section '{required_section}' is completely absent from the model card.",
                severity="high",
                evidence="",
            )
            result = env.step(preflight_action)
            observation = result.observation
            reward = result.reward or 0.0
            print(f"Pre-flight: flag_missing('{required_section}') -> reward {reward:+.2f}")
            messages.append({
                "role": "user",
                "content": f"[Auto-flagged missing section: {required_section}]"
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action_type": "flag_missing",
                    "target": required_section,
                    "reason": f"Section '{required_section}' is absent.",
                    "severity": "high",
                    "evidence": ""
                })
            })
            if result.done:
                return observation.partial_score
    # ── End pre-flight ─────────────────────────────────────────────────────────

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

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=42,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"Model request failed ({exc}). Using fallback action.")
            response_text = FALLBACK_ACTION

        # Append assistant turn to history so subsequent steps have full context
        messages.append({"role": "assistant", "content": response_text})

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

    return observation.partial_score


def main():
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Model Card Auditor — Baseline Inference")
    print(f"Model:       {MODEL_NAME}")
    print(f"Endpoint:    {API_BASE_URL}")
    print(f"Environment: {env_url}")

    scores = {}

    for task_id in ["easy", "medium", "hard"]:
        with ModelCardAuditClient(base_url=env_url).sync() as env:
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
