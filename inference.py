import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from graders import grade_task

LANE_GROUPS = ["N_S_Straight", "N_S_Left", "E_W_Straight", "E_W_Left"]
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
# DeepSeek-R1 is a strong reasoning default for policy tuning on this benchmark.
# If unavailable in your HF account/region, override MODEL_NAME to Qwen/Qwen2.5-72B-Instruct.
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "traffic_corridor_pro")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
POLICY_MAX_TOKENS = int(os.getenv("POLICY_MAX_TOKENS", "220"))
TASKS = [
    task.strip()
    for task in os.getenv(
        "TRAFFIC_CORRIDOR_TASKS",
        "easy_4_phase,medium_asymmetric,hard_corridor_emergency",
    ).split(",")
    if task.strip()
]

session = requests.Session()
client: Optional[Any] = None

DEFAULT_POLICY = {
    "queue_weight": 4.0,
    "wait_time_weight": 0.08,
    "emergency_bonus": 1000.0,
    "keep_current_bonus": 5.0,
    "switch_margin": 6.0,
    "corridor_bias": 4.0,
}


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_client() -> Optional[Any]:
    global client

    if client is not None:
        return client
    if not API_KEY:
        return None

    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client


def one_line(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={one_line(task)} env={one_line(env)} model={one_line(model)}",
        flush=True,
    )


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = one_line(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={one_line(action)} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("Unterminated JSON object in model output")


def get_policy(task_id: str, state: Dict[str, Any]) -> Dict[str, float]:
    llm_client = get_client()
    if llm_client is None:
        return DEFAULT_POLICY.copy()

    prompt = f"""
You are tuning a traffic-light control heuristic for an OpenEnv benchmark.

Task:
{task_id}

Environment state snapshot:
{json.dumps(state, indent=2)}

Return only a JSON object with numeric keys:
- queue_weight
- wait_time_weight
- emergency_bonus
- keep_current_bonus
- switch_margin
- corridor_bias

Use values that prioritize emergencies, avoid flicker, and support corridor flow.
"""

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=POLICY_MAX_TOKENS,
    )
    raw_content = response.choices[0].message.content or ""
    parsed = extract_json_object(raw_content)

    policy = DEFAULT_POLICY.copy()
    for key in policy:
        value = parsed.get(key)
        if isinstance(value, (int, float)):
            policy[key] = float(value)
    return policy


def lane_priority(lane_state: Dict[str, Any], policy: Dict[str, float]) -> float:
    score = lane_state["queue"] * policy["queue_weight"]
    score += lane_state.get("wait_time", 0.0) * policy["wait_time_weight"]
    if lane_state.get("emergency"):
        score += policy["emergency_bonus"]
    return score


def choose_phase(
    intersection: Dict[str, Any], corridor_bias: float, policy: Dict[str, float]
) -> int:
    current_phase = intersection["current_phase"]
    if intersection.get("in_transition"):
        return current_phase

    scores: List[float] = []
    for phase, lane_name in enumerate(LANE_GROUPS):
        score = lane_priority(intersection["lanes"][lane_name], policy)
        if phase == 0:
            score += corridor_bias
        if phase == current_phase:
            score += policy["keep_current_bonus"]
        scores.append(score)

    best_phase = max(range(len(scores)), key=scores.__getitem__)
    current_score = scores[current_phase]
    best_score = scores[best_phase]

    if (
        best_phase != current_phase
        and best_score < current_score + policy["switch_margin"]
    ):
        return current_phase
    return best_phase


def heuristic_actions(
    state: Dict[str, Any], policy: Dict[str, float]
) -> List[Dict[str, int]]:
    intersections = state["intersections"]
    corridor_bias = 0.0

    if len(intersections) > 1:
        ns_pressure = sum(
            lane_priority(inter["lanes"]["N_S_Straight"], policy)
            for inter in intersections
        )
        if ns_pressure >= 12.0:
            corridor_bias = policy["corridor_bias"]

    return [
        {
            "intersection_id": inter["id"],
            "phase": choose_phase(inter, corridor_bias, policy),
        }
        for inter in intersections
    ]


def action_to_string(actions: List[Dict[str, int]]) -> str:
    return json.dumps(actions, separators=(",", ":"), sort_keys=True)


def info_error(info: Any) -> Optional[str]:
    if isinstance(info, dict):
        error = info.get("last_action_error") or info.get("error")
        return str(error) if error else None
    if isinstance(info, str) and info:
        return info
    return None


def fetch_state() -> Dict[str, Any]:
    response = session.get(f"{ENV_BASE_URL}/state", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_history() -> List[Dict[str, Any]]:
    response = session.get(f"{ENV_BASE_URL}/history", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["history"]


def reset_task(task_id: str) -> None:
    response = session.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def step_env(actions: List[Dict[str, int]]) -> Dict[str, Any]:
    response = session.post(
        f"{ENV_BASE_URL}/step",
        json={"actions": actions},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def run_episode(task_id: str) -> bool:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    failed = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_task(task_id)
        state = fetch_state()

        try:
            policy = get_policy(task_id, state)
        except Exception as exc:
            policy = DEFAULT_POLICY.copy()
            stderr(f"[WARN] Falling back to default heuristic policy: {exc}")

        done = False
        while not done:
            state = fetch_state()
            actions = heuristic_actions(state, policy)
            result = step_env(actions)

            reward = float(result.get("reward") or 0.0)
            done = bool(result.get("done"))
            error = info_error(result.get("info"))

            rewards.append(reward)
            steps_taken += 1

            log_step(
                step=steps_taken,
                action=action_to_string(actions),
                reward=reward,
                done=done,
                error=error,
            )

        score = grade_task(task_id, fetch_history())
        score = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        failed = True
        stderr(f"[ERROR] Episode failed for task {task_id}: {exc}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return not failed


def main() -> int:
    if not TASKS:
        stderr("[ERROR] No tasks configured. Set TRAFFIC_CORRIDOR_TASKS.")
        return 1

    try:
        outcomes = [run_episode(task_id) for task_id in TASKS]
    finally:
        session.close()

    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
