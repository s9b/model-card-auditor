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
    """
    OpenEnv environment for AI governance model card compliance auditing.

    Agents review synthetic HuggingFace model cards and identify missing
    or inadequate sections using a structured action space.
    """

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
                # Case-insensitive lookup
                target_key = next(
                    (k for k in self._state.model_card_sections
                     if k.lower().strip() == action.target.lower().strip()),
                    None
                )
                if target_key is None:
                    error = f"Section '{action.target}' not found."
                    reward = -0.02
                    feedback = error
                elif target_key in self._state.sections_reviewed:
                    reward = -0.02
                    content = self._state.model_card_sections[target_key]
                    feedback = f"Already read '{target_key}'. Penalty applied for re-reading."
                else:
                    self._state.sections_reviewed.append(target_key)
                    reward = 0.02
                    content = self._state.model_card_sections[target_key]
                    feedback = f"Read new section '{target_key}'."

            elif action.action_type == "check_field":
                exists = action.target in self._state.model_card_sections
                content = f"Field '{action.target}': {'EXISTS' if exists else 'DOES NOT EXIST'}."
                reward = 0.01
                feedback = content

            elif action.action_type == "compare_sections":
                # Case-insensitive lookup for both targets
                key1 = next(
                    (k for k in self._state.model_card_sections
                     if k.lower().strip() == action.target.lower().strip()),
                    None
                )
                key2 = next(
                    (k for k in self._state.model_card_sections
                     if k.lower().strip() == (action.secondary_target or "").lower().strip()),
                    None
                )
                s1 = self._state.model_card_sections[key1] if key1 else "[not found]"
                s2 = self._state.model_card_sections[key2] if key2 else "[not found]"
                content = (
                    f"--- {key1 or action.target} ---\n{s1}\n\n"
                    f"--- {key2 or action.secondary_target} ---\n{s2}"
                )
                reward = 0.03
                feedback = f"Compared '{key1 or action.target}' with '{key2 or action.secondary_target}'."

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
        """No persistent resources to clean up."""
        pass

    def _is_correct_finding(self, finding: dict) -> bool:
        """Check if a finding matches any ground truth issue (case-insensitive)."""
        target = finding["target"].lower().strip()
        for gt in self._state.ground_truth_issues:
            if (target == gt["field"].lower().strip() and
                    finding["action_type"] == gt["expected_action"]):
                return True
        return False

    def _partial_score(self) -> float:
        """Compute running score using the actual task grader for accuracy."""
        if not self._state.ground_truth_issues:
            return 1.0
        grader = GRADERS.get(self._state.task_id)
        if grader is None:
            return 0.0
        return grader(self._state.agent_findings, self._state.ground_truth_issues)
