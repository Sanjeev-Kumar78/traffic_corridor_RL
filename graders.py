from typing import Any, Dict, List


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _count_phase_changes(history: List[Dict[str, Any]]) -> int:
    phase_changes = 0
    previous_phases: Dict[int, int] = {}
    for step in history:
        for inter in step.get("state", []):
            iid = int(inter.get("id", 0))
            phase = int(inter.get("current_phase", 0))
            if iid in previous_phases and previous_phases[iid] != phase:
                phase_changes += 1
            previous_phases[iid] = phase
    return phase_changes


def _extract_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    if not history:
        return {
            "avg_reward": 0.0,
            "final_queue": 100.0,
            "phase_changes": 0.0,
            "emergency_steps_any": 0.0,
            "emergency_steps_ns": 0.0,
            "queue_component_avg": 0.0,
        }

    rewards = [float(step.get("reward", 0.0)) for step in history]
    queue_components = [float(step.get("queue_component", 0.0))
                        for step in history]
    final_state = history[-1].get("state", [])

    final_queue = 0
    emergency_steps_any = 0
    emergency_steps_ns = 0

    for step in history:
        state = step.get("state", [])
        has_any = any(
            bool(lane.get("emergency", False))
            for inter in state
            for lane in inter.get("lanes", {}).values()
        )
        has_ns = any(
            bool(inter.get("lanes", {}).get(
                "N_S_Straight", {}).get("emergency", False))
            for inter in state
        )
        if has_any:
            emergency_steps_any += 1
        if has_ns:
            emergency_steps_ns += 1

    for inter in final_state:
        for lane in inter.get("lanes", {}).values():
            final_queue += int(lane.get("queue", 0))

    return {
        "avg_reward": sum(rewards) / len(rewards),
        "final_queue": float(final_queue),
        "phase_changes": float(_count_phase_changes(history)),
        "emergency_steps_any": float(emergency_steps_any),
        "emergency_steps_ns": float(emergency_steps_ns),
        "queue_component_avg": sum(queue_components) / len(queue_components),
    }


def grade_easy_4_phase(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((18.0 - metrics["final_queue"]) / 18.0)
    phase_score = clamp01((metrics["phase_changes"] - 8.0) / 24.0)
    score = 0.60 * reward_score + 0.25 * queue_score + 0.15 * phase_score
    return round(clamp01(score), 3)


def grade_medium_asymmetric(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((22.0 - metrics["final_queue"]) / 22.0)
    switch_score = clamp01((metrics["phase_changes"] - 10.0) / 30.0)
    score = 0.55 * reward_score + 0.25 * queue_score + 0.20 * switch_score
    return round(clamp01(score), 3)


def grade_hard_corridor_emergency(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((28.0 - metrics["final_queue"]) / 28.0)
    emergency_score = clamp01((45.0 - metrics["emergency_steps_ns"]) / 45.0)
    stability_score = clamp01(metrics["queue_component_avg"])
    score = (
        0.35 * reward_score
        + 0.20 * queue_score
        + 0.30 * emergency_score
        + 0.15 * stability_score
    )
    return round(clamp01(score), 3)


TASK_GRADERS = {
    "easy_4_phase": grade_easy_4_phase,
    "medium_asymmetric": grade_medium_asymmetric,
    "hard_corridor_emergency": grade_hard_corridor_emergency,
}


def grade_task(task_id: str, history: List[Dict[str, Any]]) -> float:
    if task_id not in TASK_GRADERS:
        raise ValueError(f"Unknown task: {task_id}")
    return TASK_GRADERS[task_id](history)
