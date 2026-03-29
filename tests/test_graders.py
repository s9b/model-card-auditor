"""Unit tests for model_card_auditor graders."""
import pytest
from model_card_auditor.server.graders import grade_easy, grade_medium, grade_hard


# ── Shared fixtures ────────────────────────────────────────────────────────────

EASY_GT = [
    {"field": "Training Data", "expected_action": "flag_missing"},
    {"field": "Intended Use",  "expected_action": "flag_missing"},
    {"field": "Evaluation Results", "expected_action": "flag_missing"},
    {"field": "License", "expected_action": "flag_missing"},
]

MEDIUM_GT = [
    {"field": "Training Data",       "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "multilingual web text"},
    {"field": "Bias and Limitations","expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "may reflect biases"},
    {"field": "Evaluation Results",  "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "English SNLI benchmark"},
    {"field": "Environmental Impact","expected_action": "flag_missing",    "weight": 0.20, "key_evidence": ""},
    {"field": "Out-of-Scope Uses",   "expected_action": "flag_missing",    "weight": 0.20, "key_evidence": ""},
]

HARD_GT = [
    {"field": "License",              "expected_action": "flag_inadequate", "weight": 0.25, "key_evidence": "MIT License"},
    {"field": "Training Data",        "expected_action": "flag_inadequate", "weight": 0.25, "key_evidence": "unfiltered web crawl"},
    {"field": "Bias and Limitations", "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "gender"},
    {"field": "Evaluation Results",   "expected_action": "flag_inadequate", "weight": 0.20, "key_evidence": "perplexity"},
    {"field": "Environmental Impact", "expected_action": "flag_inadequate", "weight": 0.10, "key_evidence": "3.2 kgCO2"},
]


# ── grade_easy tests ───────────────────────────────────────────────────────────

class TestGradeEasy:
    def test_no_findings_scores_zero(self):
        assert grade_easy([], EASY_GT) == 0.0

    def test_all_correct_scores_one(self):
        findings = [{"action_type": "flag_missing", "target": gt["field"]} for gt in EASY_GT]
        assert grade_easy(findings, EASY_GT) == 1.0

    def test_partial_correct(self):
        findings = [{"action_type": "flag_missing", "target": "Training Data"}]
        score = grade_easy(findings, EASY_GT)
        assert 0.0 < score < 1.0

    def test_wrong_action_type_no_credit(self):
        findings = [{"action_type": "flag_inadequate", "target": "Training Data"}]
        score = grade_easy(findings, EASY_GT)
        assert score == 0.0

    def test_false_positives_penalized(self):
        # All correct plus a false positive
        findings = [{"action_type": "flag_missing", "target": gt["field"]} for gt in EASY_GT]
        findings.append({"action_type": "flag_missing", "target": "Overview"})
        score = grade_easy(findings, EASY_GT)
        assert score < 1.0

    def test_case_insensitive_target(self):
        findings = [{"action_type": "flag_missing", "target": "training data"}]
        score = grade_easy(findings, EASY_GT)
        assert score > 0.0

    def test_empty_ground_truth_returns_one(self):
        assert grade_easy([], []) == 1.0


# ── grade_medium tests ─────────────────────────────────────────────────────────

class TestGradeMedium:
    def test_no_findings_scores_zero(self):
        assert grade_medium([], MEDIUM_GT) == 0.0

    def test_all_correct_scores_one(self):
        findings = [
            {"action_type": gt["expected_action"], "target": gt["field"]}
            for gt in MEDIUM_GT
        ]
        assert grade_medium(findings, MEDIUM_GT) == 1.0

    def test_correct_field_wrong_action_type_partial_credit(self):
        # flag_missing instead of flag_inadequate → 40% credit
        findings = [{"action_type": "flag_missing", "target": "Training Data"}]
        score = grade_medium(findings, MEDIUM_GT)
        expected = (1.0 / len(MEDIUM_GT)) * 0.40
        assert abs(score - expected) < 0.01

    def test_false_positives_penalized(self):
        findings = [
            {"action_type": gt["expected_action"], "target": gt["field"]}
            for gt in MEDIUM_GT
        ]
        findings.append({"action_type": "flag_missing", "target": "Overview"})
        score = grade_medium(findings, MEDIUM_GT)
        assert score < 1.0

    def test_case_insensitive_target(self):
        findings = [{"action_type": "flag_missing", "target": "environmental impact"}]
        score = grade_medium(findings, MEDIUM_GT)
        assert score > 0.0


# ── grade_hard tests ───────────────────────────────────────────────────────────

class TestGradeHard:
    def test_no_findings_scores_zero(self):
        assert grade_hard([], HARD_GT) == 0.0

    def test_license_violation_correct_weight(self):
        findings = [{"action_type": "flag_inadequate", "target": "License", "evidence": ""}]
        score = grade_hard(findings, HARD_GT)
        assert abs(score - 0.25) < 0.01

    def test_evidence_bonus(self):
        # License finding WITH key evidence
        no_ev   = [{"action_type": "flag_inadequate", "target": "License", "evidence": ""}]
        with_ev = [{"action_type": "flag_inadequate", "target": "License", "evidence": "MIT License is incompatible with LLaMA-2"}]
        score_no_ev   = grade_hard(no_ev, HARD_GT)
        score_with_ev = grade_hard(with_ev, HARD_GT)
        assert score_with_ev > score_no_ev

    def test_all_correct_scores_one(self):
        findings = [
            {"action_type": gt["expected_action"], "target": gt["field"], "evidence": gt["key_evidence"]}
            for gt in HARD_GT
        ]
        score = grade_hard(findings, HARD_GT)
        assert score >= 1.0  # could be slightly over 1.0 before clamping

    def test_false_positives_penalized(self):
        findings = [{"action_type": "flag_missing", "target": "FakeSection"}]
        score = grade_hard(findings, HARD_GT)
        assert score <= 0.0

    def test_case_insensitive_target(self):
        findings = [{"action_type": "flag_inadequate", "target": "license", "evidence": ""}]
        score = grade_hard(findings, HARD_GT)
        assert score > 0.0

    def test_empty_ground_truth_returns_one(self):
        assert grade_hard([], []) == 1.0
