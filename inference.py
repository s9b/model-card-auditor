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

    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

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
                        "content": SYSTEM_PROMPT,
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
