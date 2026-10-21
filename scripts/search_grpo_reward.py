"""Search-R1 EM reward for token-preserving verl agent trajectories.

The scalar optimized by GRPO is answer EM only.  Format is diagnostic.  Never
parse ``solution_str``: it can include the prompt example, retrieved passages,
and special-token-stripped reasoning.  The agent supplies its role-separated
trajectory instead; both the agent and verl's custom-reward hook use this code.
"""

from __future__ import annotations

import json

from sft_reward_monitor import parse_action, score_trajectory, valid_step_format


TERMINAL_STATUSES = frozenset({
    "ok", "turn_limit", "context_limit", "invalid_action_limit", "generated_token_limit",
})


def gold_aliases(ground_truth):
    """Accept this project's aliases and Search-R1's ``{'target': [...]}``."""
    if hasattr(ground_truth, "tolist"):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, str):
        if ground_truth[:1] in ("[", "{"):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                pass
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
    if isinstance(ground_truth, dict):
        for key in ("target", "golden_answers", "gold_answers", "answers"):
            if key in ground_truth:
                return gold_aliases(ground_truth[key])
        raise ValueError("GRPO ground truth lacks target/answer aliases")
    if (not isinstance(ground_truth, (list, tuple)) or not ground_truth
            or any(not isinstance(item, str) or not item.strip() for item in ground_truth)):
        raise ValueError("GRPO ground truth must contain nonempty string aliases")
    return list(ground_truth)


def score_record(record, ground_truth, data_source):
    """Recompute reward only from real assistant actions, never environment text."""
    if not isinstance(record, dict) or record.get("status") not in TERMINAL_STATUSES:
        raise ValueError("Missing/invalid trusted Search-R1 terminal trajectory")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("Trusted Search-R1 trajectory lacks role-separated events")
    assistant = []
    all_valid, invalid, searches, answer = True, 0, 0, ""
    answer_event_index = None
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("role") not in ("assistant", "environment"):
            raise ValueError("Malformed Search-R1 trajectory event")
        if event["role"] == "environment":
            if event.get("kind") == "search_limit_feedback":
                all_valid = False
            elif event.get("kind") == "retrieval":
                searches += 1
            continue
        text = event.get("content")
        if not isinstance(text, str):
            raise ValueError("Assistant event lacks its original generated text")
        if answer_event_index is not None:
            raise ValueError("Trajectory continued after a final assistant answer")
        assistant.append(event)
        action = parse_action(text)
        all_valid = all_valid and valid_step_format(text)
        if action is None:
            invalid += 1
        elif action["action"] == "answer":
            answer = action["value"]
            answer_event_index = index
    completed = record["status"] == "ok"
    if completed != (answer_event_index is not None):
        raise ValueError("Terminal status disagrees with actual assistant answer")
    if completed and answer_event_index != len(events) - 1:
        raise ValueError("Final answer must be the final trajectory event")
    if record.get("answer", answer) != answer:
        raise ValueError("Stored answer disagrees with actual assistant action")
    scores = score_trajectory(answer, gold_aliases(ground_truth), data_source,
                              completed=completed, all_steps_valid=all_valid)
    generated = sum(event.get("generated_tokens", 0) for event in assistant)
    return {
        "score": scores["total_reward"], **scores,
        "search_count": searches, "invalid_action_count": invalid,
        "model_calls": len(assistant), "generated_tokens": generated,
        **{"status_" + status: int(record["status"] == status)
           for status in sorted(TERMINAL_STATUSES)},
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl v0.6.1 custom_reward_function-compatible entry point.

    Both NaiveRewardManager and NaiveRewardLoopManager use these named arguments.
    The latter merges AgentLoopOutput.extra_fields into ``extra_info``.  Missing
    trajectory metadata is a wiring error and raises instead of awarding zero.
    """
    del solution_str, kwargs
    extra_info = extra_info or {}
    record = extra_info.get("search_grpo_trajectory")
    if record is None and isinstance(extra_info.get("tool_extra_fields"), dict):
        record = extra_info["tool_extra_fields"].get("search_grpo_trajectory")
    if not isinstance(record, dict):
        raise ValueError("Reward requires trusted search_grpo_trajectory, not decoded mixed-role text")
    return score_record(record, ground_truth, data_source)
