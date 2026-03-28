from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import ModelCardAction, ModelCardObservation, AuditState


class ModelCardAuditClient(EnvClient[ModelCardAction, ModelCardObservation, AuditState]):
    """
    WebSocket client for the ModelCardAuditEnvironment.

    Provides both async and sync (via .sync()) interfaces.
    """

    def _step_payload(self, action: ModelCardAction) -> Dict[str, Any]:
        """Convert a ModelCardAction to the JSON payload the server expects."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ModelCardObservation]:
        """Convert a JSON response from the server to StepResult[ModelCardObservation]."""
        obs_data = payload.get("observation", payload)
        obs = ModelCardObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AuditState:
        """Convert a JSON response from /state to an AuditState."""
        return AuditState(**payload)
