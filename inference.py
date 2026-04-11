"""
Inference Script - Traffic Corridor Pro
======================================
Mandatory environment variables:
  API_BASE_URL, MODEL_NAME, HF_TOKEN
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from graders import grade_task


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")
BENCHMARK = os.getenv("BENCHMARK", "traffic_corridor_pro")
TASKS = [
    part.strip()
    for part in os.getenv(
        "TRAFFIC_CORRIDOR_TASKS",
        "easy_4_phase,medium_asymmetric,hard_corridor_emergency",
    ).split(",")
    if part.strip()
]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.65"))
TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "40"))

LANE_PHASES = ["N_S_Straight", "N_S_Left", "E_W_Straight", "E_W_Left"]
BACKPRESSURE_TRANSITION_PENALTY = _env_float(
    "BACKPRESSURE_TRANSITION_PENALTY", 1.0)
BACKPRESSURE_RELATIVE_WEIGHT = _env_float("BACKPRESSURE_RELATIVE_WEIGHT", 0.16)
BACKPRESSURE_WAIT_WEIGHT = _env_float("BACKPRESSURE_WAIT_WEIGHT", 0.05)
BACKPRESSURE_BLOCK_THRESHOLD = _env_float("BACKPRESSURE_BLOCK_THRESHOLD", 12.0)
BACKPRESSURE_BLOCK_WEIGHT = _env_float("BACKPRESSURE_BLOCK_WEIGHT", 0.14)
STARVATION_WAIT_THRESHOLD = _env_float("STARVATION_WAIT_THRESHOLD", 20.0)
STARVATION_BONUS = _env_float("STARVATION_BONUS", 5.0)
STICKINESS_BONUS = _env_float("STICKINESS_BONUS", 0.50)
HARD_MIN_SWITCH_MARGIN = _env_float("HARD_MIN_SWITCH_MARGIN", 1.08)
HARD_STICKINESS_BONUS = _env_float("HARD_STICKINESS_BONUS", 0.65)
HARD_STARVATION_BONUS = _env_float("HARD_STARVATION_BONUS", 5.50)
HARD_NS_EMERGENCY_PHASE_BONUS = _env_float(
    "HARD_NS_EMERGENCY_PHASE_BONUS", 8.0)
POLICY_HINT_ALPHA = _env_float("POLICY_HINT_ALPHA", 0.00)
QUEUE_SQRT_WEIGHT = _env_float("QUEUE_SQRT_WEIGHT", 0.55)
QUEUE_LINEAR_WEIGHT = _env_float("QUEUE_LINEAR_WEIGHT", 0.10)
STARVATION_LINEAR_RATE = _env_float("STARVATION_LINEAR_RATE", 0.12)
STARVATION_EXP_RAMP = _env_float("STARVATION_EXP_RAMP", 0.015)
STARVATION_EXP_THRESHOLD = _env_float("STARVATION_EXP_THRESHOLD", 30.0)
STARVATION_EXP_SCALE = _env_float("STARVATION_EXP_SCALE", 2.5)
STICKINESS_DECAY = _env_float("STICKINESS_DECAY", 0.88)
STICKINESS_BASE = _env_float("STICKINESS_BASE", 0.45)
QUEUE_GROWTH_WEIGHT = _env_float("QUEUE_GROWTH_WEIGHT", 0.30)
CORRIDOR_LOOKAHEAD = int(os.getenv("CORRIDOR_LOOKAHEAD", "2"))
USE_POLICY_HINT_FOR_HARD = os.getenv("USE_POLICY_HINT_FOR_HARD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SYSTEM_PROMPT = """You are a traffic signal timing engineer tuning heuristic parameters for multi-intersection signal control.

PARAMETER GUIDE:
queue_weight (0.5-2.0): queue-length urgency.
wait_weight (0.01-0.10): fairness for long waits.
switch_margin (0.3-1.6): score gap required to switch phase.
ns_bias (0.0-1.0): N-S straight directional preference.

Return ONLY a JSON object with these keys: queue_weight, wait_weight, switch_margin, ns_bias.
"""

TASK_DESCRIPTIONS = {
    "easy_4_phase": "Single isolated 4-way intersection with roughly balanced demand.",
    "medium_asymmetric": "Single intersection with one dominant direction carrying 2-3x traffic.",
    "hard_corridor_emergency": "Three-intersection N-S corridor with emergency vehicles and downstream backpressure.",
}


def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-hf-token")
session = _build_session()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{item:.2f}" for item in rewards)
    success_str = str(success).lower()
    print(
        f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def reset_task(task_id: str) -> None:
    response = session.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def get_state() -> Dict[str, Any]:
    response = session.get(f"{ENV_BASE_URL}/state", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def step_env(actions: List[Dict[str, int]]) -> Dict[str, Any]:
    response = session.post(
        f"{ENV_BASE_URL}/step",
        json={"actions": actions},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_history() -> List[Dict[str, Any]]:
    response = session.get(f"{ENV_BASE_URL}/history", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("history", [])


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start >= 0:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: idx + 1])
                    except json.JSONDecodeError:
                        break

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.startswith("{"):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return {}


def get_policy_hint(task_id: str) -> Dict[str, float]:
    if not HF_TOKEN:
        return {}
    if task_id == "hard_corridor_emergency" and not USE_POLICY_HINT_FOR_HARD:
        return {}

    task_desc = TASK_DESCRIPTIONS.get(task_id, f"Task: {task_id}")
    user_prompt = (
        f"TASK: {task_id}\n"
        f"SCENARIO: {task_desc}\n"
        "Return JSON with numeric keys queue_weight, wait_weight, switch_margin, ns_bias."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You tune a traffic-control heuristic. Return plain JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=120,
            stream=False,
        )
        payload = _extract_json_object(
            (completion.choices[0].message.content or "").strip())
        suggested = {
            "queue_weight": float(payload.get("queue_weight", 1.0)),
            "wait_weight": float(payload.get("wait_weight", 0.03)),
            "switch_margin": float(payload.get("switch_margin", 0.8)),
            "ns_bias": float(payload.get("ns_bias", 0.0)),
        }
        suggested["queue_weight"] = max(
            0.5, min(2.0, suggested["queue_weight"]))
        suggested["wait_weight"] = max(
            0.01, min(0.10, suggested["wait_weight"]))
        suggested["switch_margin"] = max(
            0.3, min(1.6, suggested["switch_margin"]))
        suggested["ns_bias"] = max(0.0, min(1.0, suggested["ns_bias"]))

        base = {
            "queue_weight": 1.0,
            "wait_weight": 0.03,
            "switch_margin": 0.8,
            "ns_bias": 0.0,
        }
        alpha = max(0.0, min(1.0, POLICY_HINT_ALPHA))
        blended = {
            key: (1.0 - alpha) * base[key] + alpha * suggested[key]
            for key in base
        }
        if task_id == "hard_corridor_emergency":
            blended["switch_margin"] = base["switch_margin"]
            blended["ns_bias"] = min(blended["ns_bias"], 0.6)
        return blended
    except Exception:
        return {}


def _find_intersection(intersections: List[Dict[str, Any]], target_id: int) -> Optional[Dict[str, Any]]:
    for item in intersections:
        if int(item.get("id", -1)) == target_id:
            return item
    return None


def _corridor_downstream_pressure(task_id: str, intersections: List[Dict[str, Any]], intersection_id: int) -> float:
    if task_id != "hard_corridor_emergency":
        return 0.0

    downstream = _find_intersection(intersections, intersection_id + 1)
    if downstream is None:
        return 0.0

    downstream_lane = downstream.get("lanes", {}).get("N_S_Straight", {})
    downstream_queue = float(downstream_lane.get("queue", 0))
    downstream_wait = float(downstream_lane.get("wait_time", 0.0))
    pressure = downstream_queue + BACKPRESSURE_WAIT_WEIGHT * downstream_wait
    if bool(downstream.get("in_transition", False)) and downstream_queue >= 4:
        pressure += BACKPRESSURE_TRANSITION_PENALTY
    if int(downstream.get("current_phase", 0)) != 0 and not bool(downstream.get("in_transition", False)) and downstream_queue >= 6:
        pressure += 0.8
    return pressure


def _queue_urgency(queue: float, queue_weight: float) -> float:
    if queue <= 0:
        return 0.0
    return queue_weight * QUEUE_SQRT_WEIGHT * math.sqrt(queue)


def _starvation_bonus(wait_time: float, hard: bool) -> float:
    base = STARVATION_LINEAR_RATE * wait_time
    threshold = STARVATION_EXP_THRESHOLD * (1.1 if hard else 1.0)
    if wait_time > threshold:
        overshoot = wait_time - threshold
        base += STARVATION_EXP_SCALE * \
            (math.exp(STARVATION_EXP_RAMP * overshoot) - 1.0)
    if hard:
        base += 0.25 * HARD_STARVATION_BONUS
    return base


def _estimate_phase_duration(history: List[Dict[str, Any]], intersection_id: int, current_phase: int) -> int:
    duration = 0
    for snapshot in reversed(history):
        state = snapshot.get("state", [])
        if not isinstance(state, list):
            break
        match = _find_intersection(state, intersection_id)
        if match is None:
            break
        if int(match.get("current_phase", -1)) != current_phase:
            break
        duration += 1
    return duration


def _queue_growth_rate(
    intersection_id: int,
    lane: str,
    current_queue: float,
    history: List[Dict[str, Any]],
) -> float:
    if not history:
        return 0.0
    prev_state = history[-1].get("state", [])
    if not isinstance(prev_state, list):
        return 0.0
    prev_inter = _find_intersection(prev_state, intersection_id)
    if prev_inter is None:
        return 0.0
    prev_queue = float(prev_inter.get(
        "lanes", {}).get(lane, {}).get("queue", 0))
    return current_queue - prev_queue


def _decaying_stickiness(current_phase: int, phase: int, hard: bool, phase_duration: int) -> float:
    if phase != current_phase:
        return 0.0
    base = HARD_STICKINESS_BONUS if hard else STICKINESS_BASE
    decay = 0.85 if hard else STICKINESS_DECAY
    return base * (decay ** phase_duration)


def choose_phase(
    task_id: str,
    inter: Dict[str, Any],
    policy: Dict[str, float],
    intersections: List[Dict[str, Any]],
) -> int:
    lanes = inter["lanes"]
    current_phase = int(inter["current_phase"])
    in_transition = bool(inter["in_transition"])
    intersection_id = int(inter["id"])

    if in_transition:
        return current_phase

    is_hard = task_id == "hard_corridor_emergency"
    if is_hard:
        ns_lane = lanes.get("N_S_Straight", {})
        if bool(ns_lane.get("emergency", False)):
            return 0

    emergencies: List[tuple] = []
    for phase, lane in enumerate(LANE_PHASES):
        lane_state = lanes[lane]
        if lane_state.get("emergency", False):
            emergencies.append(
                (float(lane_state.get("wait_time", 0.0)), phase))
    if emergencies:
        emergencies.sort(reverse=True)
        return emergencies[0][1]

    queue_weight = policy.get("queue_weight", 1.0)
    wait_weight = policy.get("wait_weight", 0.03)
    switch_margin = policy.get("switch_margin", 0.8)
    ns_bias = policy.get("ns_bias", 0.0)
    if is_hard:
        switch_margin = max(switch_margin, HARD_MIN_SWITCH_MARGIN)
    downstream_pressure = _corridor_downstream_pressure(
        task_id, intersections, intersection_id)
    current_pressure = float(lanes["N_S_Straight"].get("queue", 0)) + BACKPRESSURE_WAIT_WEIGHT * float(
        lanes["N_S_Straight"].get("wait_time", 0.0)
    )
    scores = []
    for phase, lane in enumerate(LANE_PHASES):
        queue = float(lanes[lane].get("queue", 0))
        wait_time = float(lanes[lane].get("wait_time", 0.0))
        score = queue_weight * queue + wait_weight * wait_time
        if task_id != "easy_4_phase" and phase == 0:
            score += ns_bias
        if phase == 0 and is_hard:
            ns_lane = lanes.get("N_S_Straight", {})
            if bool(ns_lane.get("emergency", False)):
                score += HARD_NS_EMERGENCY_PHASE_BONUS
            if downstream_pressure > BACKPRESSURE_BLOCK_THRESHOLD:
                score -= min(downstream_pressure,
                             BACKPRESSURE_BLOCK_THRESHOLD) * BACKPRESSURE_BLOCK_WEIGHT
            else:
                score += (current_pressure - downstream_pressure) * \
                    BACKPRESSURE_RELATIVE_WEIGHT
        if wait_time > STARVATION_WAIT_THRESHOLD:
            score += HARD_STARVATION_BONUS if is_hard else STARVATION_BONUS
        if phase == current_phase:
            score += HARD_STICKINESS_BONUS if is_hard else STICKINESS_BONUS
        scores.append(score)

    best_phase = int(max(range(4), key=lambda idx: scores[idx]))
    if scores[best_phase] <= scores[current_phase] + switch_margin:
        return current_phase
    return best_phase


def run_task(task_id: str) -> float:
    reset_task(task_id)
    policy = get_policy_hint(task_id)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    try:
        state = get_state()
        max_steps = int(state.get("max_steps", 100))
        for step in range(1, max_steps + 1):
            intersections = state.get("intersections", [])
            actions: List[Dict[str, int]] = []
            for inter in intersections:
                iid = int(inter["id"])
                phase = choose_phase(task_id, inter, policy, intersections)
                actions.append({"intersection_id": iid, "phase": phase})

            action_str = json.dumps(
                actions, separators=(",", ":"), sort_keys=True)
            result = step_env(actions)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            info = result.get("info", {})
            error = None
            if isinstance(info, dict):
                error = info.get("last_action_error") or info.get("error")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error)

            state = get_state()

            if done:
                break

        history = get_history()
        score = grade_task(task_id, history)
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


def main() -> int:
    overall_scores = []
    for task_id in TASKS:
        try:
            task_score = run_task(task_id)
        except Exception:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            task_score = 0.0
        overall_scores.append(task_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
