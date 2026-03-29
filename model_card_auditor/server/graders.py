from typing import List, Dict


def _norm(s: str) -> str:
    """Normalize field names for robust comparison."""
    return s.lower().strip()


def grade_easy(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Easy task: 4 completely missing required fields.
    Agent must flag each as 'flag_missing'.
    Full credit per correct flag. Small false-positive penalty.
    Score guaranteed to vary based on how many issues agent finds.
    """
    if not ground_truth:
        return 1.0

    correct = sum(
        1 for gt in ground_truth
        if any(
            _norm(f["target"]) == _norm(gt["field"]) and f["action_type"] == "flag_missing"
            for f in agent_findings
        )
    )

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(_norm(f["target"]) == _norm(gt["field"]) for gt in ground_truth)
    )

    score = (correct / len(ground_truth)) - min(0.30, false_positives * 0.05)
    return max(0.0, min(1.0, round(score, 4)))


def grade_medium(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Medium task: 5 issues, mix of missing AND inadequate.
    Correct issue + correct action_type = full credit.
    Correct issue + wrong action_type = 40% credit.
    False positives penalised.
    """
    if not ground_truth:
        return 1.0

    per_issue = 1.0 / len(ground_truth)
    score = 0.0

    for gt in ground_truth:
        matches = [f for f in agent_findings if _norm(f["target"]) == _norm(gt["field"])]
        if not matches:
            continue
        best = matches[0]
        if best["action_type"] == gt["expected_action"]:
            score += per_issue           # full credit
        else:
            score += per_issue * 0.40   # partial: found the field, wrong classification

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(_norm(f["target"]) == _norm(gt["field"]) for gt in ground_truth)
    )

    score -= min(0.25, false_positives * 0.05)
    return max(0.0, min(1.0, round(score, 4)))


def grade_hard(agent_findings: List[Dict], ground_truth: List[Dict]) -> float:
    """
    Hard task: 5 subtle cross-section violations, each with a weight.
    Agent must cross-reference sections to find them.
    Evidence bonus for citing correct text.
    """
    if not ground_truth:
        return 1.0

    score = 0.0
    for gt in ground_truth:
        weight = gt.get("weight", 0.20)
        found = any(
            _norm(f["target"]) == _norm(gt["field"]) and f["action_type"] == gt["expected_action"]
            for f in agent_findings
        )
        if found:
            score += weight
            # 10% evidence bonus if agent quoted the key text
            key_evidence = gt.get("key_evidence", "")
            if key_evidence and any(
                key_evidence.lower() in f.get("evidence", "").lower()
                for f in agent_findings
                if _norm(f["target"]) == _norm(gt["field"])
            ):
                score += weight * 0.10

    false_positives = sum(
        1 for f in agent_findings
        if f["action_type"] in ("flag_missing", "flag_inadequate")
        and not any(_norm(f["target"]) == _norm(gt["field"]) for gt in ground_truth)
    )

    score -= min(0.20, false_positives * 0.04)
    return max(0.0, min(1.0, round(score, 4)))
