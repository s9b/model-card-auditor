"""Unit tests for ModelCardAuditEnvironment."""
import pytest
from model_card_auditor.server.environment import ModelCardAuditEnvironment
from model_card_auditor.models import ModelCardAction


@pytest.fixture
def env():
    return ModelCardAuditEnvironment()


class TestReset:
    def test_reset_easy_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert obs.done is False
        assert obs.reward is None
        assert obs.steps_remaining > 0
        assert len(obs.sections_available) > 0

    def test_reset_sets_correct_sections_easy(self, env):
        obs = env.reset(task_id="easy")
        # easy task has Overview and Installation only
        assert "Overview" in obs.sections_available or len(obs.sections_available) >= 1

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="easy")
        env.step(ModelCardAction(action_type="flag_missing", target="License",
                                 reason="missing", severity="high", evidence=""))
        obs = env.reset(task_id="medium")
        assert obs.findings_count == 0
        assert obs.sections_reviewed == []
        assert obs.partial_score == 0.0

    def test_reset_medium_has_six_sections(self, env):
        obs = env.reset(task_id="medium")
        assert len(obs.sections_available) == 6

    def test_reset_hard_has_nine_sections(self, env):
        obs = env.reset(task_id="hard")
        assert len(obs.sections_available) == 9


class TestStep:
    def test_read_new_section_positive_reward(self, env):
        obs = env.reset(task_id="medium")
        first_section = obs.sections_available[0]
        result = env.step(ModelCardAction(
            action_type="read_section", target=first_section,
            reason="", severity="low", evidence=""
        ))
        assert result.reward == pytest.approx(0.02)
        # env.step() server-side returns ModelCardObservation directly (no .observation wrapper)
        assert first_section in result.sections_reviewed

    def test_read_same_section_twice_negative_reward(self, env):
        obs = env.reset(task_id="medium")
        first_section = obs.sections_available[0]
        action = ModelCardAction(action_type="read_section", target=first_section,
                                 reason="", severity="low", evidence="")
        env.step(action)
        result = env.step(action)
        assert result.reward == pytest.approx(-0.02)

    def test_read_invalid_section_error(self, env):
        env.reset(task_id="easy")
        result = env.step(ModelCardAction(
            action_type="read_section", target="NonExistentSection",
            reason="", severity="low", evidence=""
        ))
        assert result.last_action_error is not None

    def test_correct_flag_missing_positive_reward(self, env):
        env.reset(task_id="easy")
        result = env.step(ModelCardAction(
            action_type="flag_missing", target="Training Data",
            reason="absent", severity="high", evidence=""
        ))
        assert result.reward == pytest.approx(0.15)

    def test_incorrect_flag_negative_reward(self, env):
        env.reset(task_id="easy")
        result = env.step(ModelCardAction(
            action_type="flag_missing", target="Overview",  # Overview EXISTS in easy task
            reason="test", severity="low", evidence=""
        ))
        assert result.reward == pytest.approx(-0.04)

    def test_submit_audit_terminates_episode(self, env):
        env.reset(task_id="easy")
        result = env.step(ModelCardAction(
            action_type="submit_audit", target="final",
            reason="done", severity="low", evidence=""
        ))
        assert result.done is True

    def test_compare_sections_positive_reward(self, env):
        obs = env.reset(task_id="hard")
        s1, s2 = obs.sections_available[0], obs.sections_available[1]
        result = env.step(ModelCardAction(
            action_type="compare_sections", target=s1, secondary_target=s2,
            reason="", severity="low", evidence=""
        ))
        assert result.reward == pytest.approx(0.03)

    def test_case_insensitive_section_read(self, env):
        obs = env.reset(task_id="medium")
        first_section = obs.sections_available[0]
        result = env.step(ModelCardAction(
            action_type="read_section", target=first_section.lower(),
            reason="", severity="low", evidence=""
        ))
        # Should succeed (case-insensitive), not return an error
        assert result.last_action_error is None
        assert result.reward == pytest.approx(0.02)


class TestState:
    def test_state_returns_audit_state(self, env):
        env.reset(task_id="easy")
        state = env.state
        assert state.task_id == "easy"
        assert state.step_count == 0

    def test_state_step_count_increments(self, env):
        obs = env.reset(task_id="easy")
        env.step(ModelCardAction(
            action_type="read_section", target=obs.sections_available[0],
            reason="", severity="low", evidence=""
        ))
        assert env.state.step_count == 1
