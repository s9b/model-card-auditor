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
