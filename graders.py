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
            "avg_total_queue": 100.0,
            "final_queue": 100.0,
            "phase_changes": 0.0,
            "emergency_steps_any": 0.0,
            "emergency_active_total": 0.0,
            "avg_emergency_wait": 50.0,
            "emergency_spawned": 0.0,
            "emergency_cleared": 0.0,
            "emergency_clearance_rate": 0.0,
            "queue_component_avg": 0.0,
        }

    rewards = [float(step.get("reward", 0.0)) for step in history]
    queue_components = [float(step.get("queue_component", 0.0)) for step in history]
    final_state = history[-1].get("state", [])
    metrics_blocks = [step.get("metrics", {}) for step in history]

    final_queue = 0
    emergency_steps_any = 0
    emergency_active_total = 0.0
    emergency_wait_total = 0.0
    avg_total_queue_acc = 0.0
    emergency_spawned = 0.0
    emergency_cleared = 0.0

    for step, metrics in zip(history, metrics_blocks):
        state = step.get("state", [])
        has_any = any(
            bool(lane.get("emergency", False))
            for inter in state
            for lane in inter.get("lanes", {}).values()
        )
        if has_any:
            emergency_steps_any += 1

        if metrics:
            avg_total_queue_acc += float(metrics.get("total_queue", 0.0))
            emergency_active_total += float(metrics.get("emergency_active_count", 0.0))
            emergency_wait_total += float(metrics.get("emergency_wait_sum", 0.0))
            emergency_spawned += float(metrics.get("emergency_spawned", 0.0))
            emergency_cleared += float(metrics.get("emergency_cleared", 0.0))
        else:
            total_queue = 0.0
            active_count = 0.0
            for inter in state:
                for lane in inter.get("lanes", {}).values():
                    total_queue += float(lane.get("queue", 0.0))
                    if bool(lane.get("emergency", False)):
                        active_count += 1.0
            avg_total_queue_acc += total_queue
            emergency_active_total += active_count

    for inter in final_state:
        for lane in inter.get("lanes", {}).values():
            final_queue += int(lane.get("queue", 0))

    clearance_rate = emergency_cleared / emergency_spawned if emergency_spawned > 0 else 1.0
    avg_emergency_wait = (
        emergency_wait_total / emergency_active_total if emergency_active_total > 0 else 0.0
    )

    return {
        "avg_reward": sum(rewards) / len(rewards),
        "avg_total_queue": avg_total_queue_acc / len(history),
        "final_queue": float(final_queue),
        "phase_changes": float(_count_phase_changes(history)),
        "emergency_steps_any": float(emergency_steps_any),
        "emergency_active_total": float(emergency_active_total),
        "avg_emergency_wait": float(avg_emergency_wait),
        "emergency_spawned": float(emergency_spawned),
        "emergency_cleared": float(emergency_cleared),
        "emergency_clearance_rate": float(clearance_rate),
        "queue_component_avg": sum(queue_components) / len(queue_components),
    }


def grade_easy_4_phase(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((12.0 - metrics["avg_total_queue"]) / 12.0)
    final_queue_score = clamp01((18.0 - metrics["final_queue"]) / 18.0)
    phase_score = clamp01((metrics["phase_changes"] - 6.0) / 20.0)
    score = (
        0.45 * reward_score
        + 0.25 * queue_score
        + 0.20 * final_queue_score
        + 0.10 * phase_score
    )
    return round(clamp01(score), 3)


def grade_medium_asymmetric(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((14.0 - metrics["avg_total_queue"]) / 14.0)
    final_queue_score = clamp01((22.0 - metrics["final_queue"]) / 22.0)
    switch_score = clamp01((metrics["phase_changes"] - 8.0) / 26.0)
    score = (
        0.40 * reward_score
        + 0.25 * queue_score
        + 0.20 * final_queue_score
        + 0.15 * switch_score
    )
    return round(clamp01(score), 3)


def grade_hard_corridor_emergency(history: List[Dict[str, Any]]) -> float:
    metrics = _extract_metrics(history)
    reward_score = clamp01(metrics["avg_reward"])
    queue_score = clamp01((18.0 - metrics["avg_total_queue"]) / 18.0)
    final_queue_score = clamp01((28.0 - metrics["final_queue"]) / 28.0)
    emergency_clearance_score = clamp01(metrics["emergency_clearance_rate"])
    emergency_latency_score = clamp01((6.0 - metrics["avg_emergency_wait"]) / 6.0)
    stability_score = clamp01(metrics["queue_component_avg"])
    score = (
        0.25 * reward_score
        + 0.15 * queue_score
        + 0.15 * final_queue_score
        + 0.25 * emergency_clearance_score
        + 0.15 * emergency_latency_score
        + 0.05 * stability_score
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
