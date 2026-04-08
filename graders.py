def calculate_score(history, max_penalty, min_penalty) -> float:
    total_reward = sum(step["reward"] for step in history)
    # total_reward is negative. Closer to 0 is better.
    # Map to 0.0 - 1.0
    score = (total_reward - max_penalty) / (min_penalty - max_penalty)
    return round(max(0.0, min(1.0, score)), 2)


def grade_easy_4_phase(history) -> float:
    # Max penalty assumes terrible switching or starvation
    return calculate_score(history, max_penalty=-2000.0, min_penalty=-300.0)


def grade_medium_asymmetric(history) -> float:
    # Requires skipping low-traffic phases to optimize
    return calculate_score(history, max_penalty=-3500.0, min_penalty=-500.0)


def grade_hard_corridor_emergency(history) -> float:
    # Massive penalties if emergencies are ignored or wave breaks
    return calculate_score(history, max_penalty=-8000.0, min_penalty=-1200.0)


TASK_GRADERS = {
    "easy_4_phase": grade_easy_4_phase,
    "medium_asymmetric": grade_medium_asymmetric,
    "hard_corridor_emergency": grade_hard_corridor_emergency,
}


def grade_task(task_id, history) -> float:
    if task_id not in TASK_GRADERS:
        raise ValueError(f"Unknown task: {task_id}")
    return TASK_GRADERS[task_id](history)
