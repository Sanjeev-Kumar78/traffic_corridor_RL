from typing import Any, Dict, List


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_reward(total_reward: float, best_reward: float, worst_reward: float) -> float:
    """
    Map total reward to [0, 1].
    best_reward: expected strong run (higher / less negative)
    worst_reward: expected poor run (lower / more negative)
    """
    if best_reward <= worst_reward:
        raise ValueError("best_reward must be greater than worst_reward")
    return clamp01((total_reward - worst_reward) / (best_reward - worst_reward))


def extract_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    if not history:
        return {
            "total_reward": 0.0,
            "min_reward": 0.0,
            "final_queue": 0.0,
            "emergency_active_steps": 0.0,
        }

    rewards = [float(step.get("reward", 0.0)) for step in history]
    total_reward = sum(rewards)
    min_reward = min(rewards)

    final_state = history[-1].get("state", [])
    final_queue = 0
    for inter in final_state:
        lanes = inter.get("lanes", {})
        for lane in lanes.values():
            final_queue += int(lane.get("queue", 0))

    emergency_active_steps = 0
    for step in history:
        state = step.get("state", [])
        has_emergency = any(
            lane.get("emergency", False)
            for inter in state
            for lane in inter.get("lanes", {}).values()
        )
        if has_emergency:
            emergency_active_steps += 1

    return {
        "total_reward": float(total_reward),
        "min_reward": float(min_reward),
        "final_queue": float(final_queue),
        "emergency_active_steps": float(emergency_active_steps),
    }


def grade_easy_4_phase(history: List[Dict[str, Any]]) -> float:
    metrics = extract_metrics(history)
    reward_score = normalize_reward(
        metrics["total_reward"], best_reward=-300.0, worst_reward=-2200.0
    )
    queue_score = clamp01((20.0 - metrics["final_queue"]) / 20.0)
    score = 0.85 * reward_score + 0.15 * queue_score
    return round(clamp01(score), 2)


def grade_medium_asymmetric(history: List[Dict[str, Any]]) -> float:
    metrics = extract_metrics(history)
    reward_score = normalize_reward(
        metrics["total_reward"], best_reward=-600.0, worst_reward=-3600.0
    )
    queue_score = clamp01((24.0 - metrics["final_queue"]) / 24.0)
    score = 0.8 * reward_score + 0.2 * queue_score
    return round(clamp01(score), 2)


def grade_hard_corridor_emergency(history: List[Dict[str, Any]]) -> float:
    metrics = extract_metrics(history)

    # Hard task combines reward with reliability signals:
    # - severe single-step collapses (very negative min reward)
    # - unresolved congestion at the end
    # - prolonged emergency presence
    reward_score = normalize_reward(
        metrics["total_reward"], best_reward=-700.0, worst_reward=-3800.0
    )
    worst_step_score = clamp01((metrics["min_reward"] - (-30.0)) / 30.0)
    queue_score = clamp01((15.0 - metrics["final_queue"]) / 15.0)
    emergency_score = clamp01((20.0 - metrics["emergency_active_steps"]) / 20.0)

    score = (
        0.2 * reward_score
        + 0.4 * worst_step_score
        + 0.2 * queue_score
        + 0.2 * emergency_score
    )
    return round(clamp01(score), 2)


TASK_GRADERS = {
    "easy_4_phase": grade_easy_4_phase,
    "medium_asymmetric": grade_medium_asymmetric,
    "hard_corridor_emergency": grade_hard_corridor_emergency,
}


def grade_task(task_id: str, history: List[Dict[str, Any]]) -> float:
    if task_id not in TASK_GRADERS:
        raise ValueError(f"Unknown task: {task_id}")
    return TASK_GRADERS[task_id](history)
