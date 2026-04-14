import json
import os
import random
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Traffic Corridor Pro")
DEFAULT_HF_SPACE_URL = os.getenv("HF_SPACE_URL", "").strip()
UI_TEMPLATE_PATH = Path(__file__).resolve().parent / "server" / "ui.html"


LANE_GROUPS = ["N_S_Straight", "N_S_Left", "E_W_Straight", "E_W_Left"]
TASK_IDS = ["easy_4_phase", "medium_asymmetric", "hard_corridor_emergency"]
DEFAULT_TASK_ID = "easy_4_phase"
DEFAULT_SESSION_ID = "default"
TASK_BASE_SEEDS = {
    "easy_4_phase": 17,
    "medium_asymmetric": 29,
    "hard_corridor_emergency": 43,
}
TASK_MAX_STEPS = {
    "easy_4_phase": 100,
    "medium_asymmetric": 120,
    "hard_corridor_emergency": 140,
}
TASK_RATE_SPREAD = {
    "easy_4_phase": 0.025,
    "medium_asymmetric": 0.035,
    "hard_corridor_emergency": 0.03,
}
SWITCH_COOLDOWN_STEPS = 2
DISCHARGE_RATE = 3
EMERGENCY_SPAWN_RATE = 0.03


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class ResetPayload(BaseModel):
    task_id: str = DEFAULT_TASK_ID
    seed: Optional[int] = Field(default=None, ge=0)
    session_id: Optional[str] = None


class IntersectionAction(BaseModel):
    intersection_id: int = Field(..., ge=0)
    phase: Literal[0, 1, 2, 3]


class TrafficAction(BaseModel):
    actions: List[IntersectionAction]


class LaneObservation(BaseModel):
    queue: int
    wait_time: float
    emergency: bool


class IntersectionObservation(BaseModel):
    id: int
    lanes: Dict[str, LaneObservation]
    current_phase: Literal[0, 1, 2, 3]
    in_transition: bool
    switch_cooldown_remaining: int


class RewardModel(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    queue_component: float = Field(..., ge=0.0, le=1.0)
    switching_component: float = Field(..., ge=0.0, le=1.0)
    emergency_component: float = Field(..., ge=0.0, le=1.0)


class TrafficObservation(BaseModel):
    session_id: str
    task_id: str
    seed: int
    step: int
    max_steps: int
    intersections: List[IntersectionObservation]
    reward_breakdown: Optional[RewardModel] = None


class StepResult(BaseModel):
    observation: TrafficObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    status: str
    message: str
    session_id: str
    state: TrafficObservation


class StateResponse(TrafficObservation):
    pass


class StepResponse(BaseModel):
    session_id: str
    observation: TrafficObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]


class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]


class InferenceRunEntry(BaseModel):
    task_id: str
    seed: int
    success: bool
    steps: int
    score: float
    rewards: List[float]


class InferenceRunResponse(BaseModel):
    success: bool
    summary: Dict[str, Any]
    runs: List[InferenceRunEntry]
    output: str


class _Intersection:
    def __init__(
        self,
        intersection_id: int,
        rates: Dict[str, float],
        rng: random.Random,
        enable_emergencies: bool = False,
        emergency_spawn_rate: float = EMERGENCY_SPAWN_RATE,
    ) -> None:
        self.id = intersection_id
        self.rates = rates
        self.rng = rng
        self.enable_emergencies = enable_emergencies
        self.emergency_spawn_rate = emergency_spawn_rate

        self.queues = {lane: 0 for lane in LANE_GROUPS}
        self.wait_times = {lane: 0.0 for lane in LANE_GROUPS}
        self.emergency_positions = {lane: None for lane in LANE_GROUPS}
        self.emergency_wait_steps = {lane: 0 for lane in LANE_GROUPS}
        self.departures = {lane: 0 for lane in LANE_GROUPS}
        self.released_emergency = {lane: False for lane in LANE_GROUPS}
        self.released_emergency_offset = {lane: None for lane in LANE_GROUPS}
        self.spawned_emergencies_step = 0
        self.spawned_emergencies_total = 0

        self.current_phase = 0
        self.switch_cooldown = 0

    def _spawn_vehicles(self) -> None:
        self.spawned_emergencies_step = 0
        for lane in LANE_GROUPS:
            if self.rng.random() < self.rates[lane]:
                queue_before = self.queues[lane]
                self.queues[lane] += 1
                if (
                    self.enable_emergencies
                    and self.emergency_positions[lane] is None
                    and self.rng.random() < self.emergency_spawn_rate
                ):
                    self.emergency_positions[lane] = queue_before
                    self.spawned_emergencies_step += 1
                    self.spawned_emergencies_total += 1

    def _discharge_active_phase(self) -> None:
        self.departures = {lane: 0 for lane in LANE_GROUPS}
        self.released_emergency = {lane: False for lane in LANE_GROUPS}
        self.released_emergency_offset = {lane: None for lane in LANE_GROUPS}

        if self.switch_cooldown > 0:
            return

        active_lane = LANE_GROUPS[self.current_phase]
        cleared = min(self.queues[active_lane], DISCHARGE_RATE)
        emergency_position = self.emergency_positions[active_lane]

        if emergency_position is not None:
            if emergency_position < cleared:
                self.released_emergency[active_lane] = True
                self.released_emergency_offset[active_lane] = emergency_position
                self.emergency_positions[active_lane] = None
            else:
                self.emergency_positions[active_lane] = emergency_position - cleared

        self.queues[active_lane] -= cleared
        self.departures[active_lane] = cleared

    def _update_wait_times(self) -> Tuple[float, float, float]:
        queue_cost = 0.0
        emergency_cost = 0.0
        for lane in LANE_GROUPS:
            queue = self.queues[lane]
            if queue > 0:
                self.wait_times[lane] += 1.0
            else:
                self.wait_times[lane] = 0.0
            queue_cost += 0.018 * queue + 0.0035 * (queue * queue)

            has_emergency = self.emergency_positions[lane] is not None
            if has_emergency:
                self.emergency_wait_steps[lane] += 1
                emergency_cost += 0.08 * (1.35 ** (self.emergency_wait_steps[lane] - 1))
            else:
                self.emergency_wait_steps[lane] = 0

        return queue_cost, emergency_cost, queue_cost + emergency_cost

    def step(self, target_phase: int) -> Tuple[float, float, float]:
        switch_cost = 0.0
        if target_phase != self.current_phase and self.switch_cooldown == 0:
            self.current_phase = target_phase
            self.switch_cooldown = SWITCH_COOLDOWN_STEPS
            switch_cost = 0.05

        self._spawn_vehicles()
        self._discharge_active_phase()
        queue_cost, emergency_cost, _ = self._update_wait_times()

        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        return queue_cost, switch_cost, emergency_cost

    def to_observation(self) -> IntersectionObservation:
        return IntersectionObservation(
            id=self.id,
            lanes={
                lane: LaneObservation(
                    queue=self.queues[lane],
                    wait_time=round(self.wait_times[lane], 2),
                    emergency=self.emergency_positions[lane] is not None,
                )
                for lane in LANE_GROUPS
            },
            current_phase=self.current_phase,
            in_transition=self.switch_cooldown > 0,
            switch_cooldown_remaining=self.switch_cooldown,
        )


class TrafficCorridorEnv:
    """OpenEnv-style environment with typed action/observation/reward models."""

    def __init__(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        self._lock = threading.RLock()
        self.rng = random.Random()
        self.session_id = session_id
        self.task_id = DEFAULT_TASK_ID
        self.seed = TASK_BASE_SEEDS[DEFAULT_TASK_ID]
        self.step_count = 0
        self.max_steps = TASK_MAX_STEPS[DEFAULT_TASK_ID]
        self.intersections: List[_Intersection] = []
        self.history: List[Dict[str, Any]] = []
        self.reset(DEFAULT_TASK_ID)

    def _sample_rate(self, base_rate: float, spread: float) -> float:
        return round(
            max(0.02, min(0.85, self.rng.uniform(base_rate - spread, base_rate + spread))),
            3,
        )

    def _sample_rates(self, base_rates: Dict[str, float], spread: float) -> Dict[str, float]:
        return {lane: self._sample_rate(rate, spread) for lane, rate in base_rates.items()}

    def _make_intersection(
        self,
        intersection_id: int,
        rates: Dict[str, float],
        enable_emergencies: bool = False,
    ) -> _Intersection:
        return _Intersection(intersection_id, rates, self.rng, enable_emergencies)

    def _build_task(self, task_id: str) -> List[_Intersection]:
        spread = TASK_RATE_SPREAD[task_id]
        intersections: List[_Intersection] = []

        if task_id == "easy_4_phase":
            rates = self._sample_rates({lane: 0.16 for lane in LANE_GROUPS}, spread)
            intersections.append(self._make_intersection(0, rates, False))

        elif task_id == "medium_asymmetric":
            rates = self._sample_rates(
                {
                    "N_S_Straight": 0.62,
                    "N_S_Left": 0.08,
                    "E_W_Straight": 0.18,
                    "E_W_Left": 0.07,
                },
                spread,
            )
            intersections.append(self._make_intersection(0, rates, False))

        elif task_id == "hard_corridor_emergency":
            corridor = [
                {
                    "N_S_Straight": 0.40,
                    "N_S_Left": 0.12,
                    "E_W_Straight": 0.15,
                    "E_W_Left": 0.08,
                },
                {
                    "N_S_Straight": 0.05,
                    "N_S_Left": 0.10,
                    "E_W_Straight": 0.14,
                    "E_W_Left": 0.08,
                },
                {
                    "N_S_Straight": 0.04,
                    "N_S_Left": 0.09,
                    "E_W_Straight": 0.13,
                    "E_W_Left": 0.08,
                },
            ]
            for intersection_id, base_rates in enumerate(corridor):
                intersections.append(
                    self._make_intersection(
                        intersection_id,
                        self._sample_rates(base_rates, spread),
                        True,
                    )
                )

        return intersections

    def reset(self, task_id: str = DEFAULT_TASK_ID, seed: Optional[int] = None) -> TrafficObservation:
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task: {task_id}")

        with self._lock:
            self.task_id = task_id
            self.seed = TASK_BASE_SEEDS[task_id] if seed is None else seed
            self.step_count = 0
            self.max_steps = TASK_MAX_STEPS[task_id]
            self.history = []
            self.rng.seed(self.seed)
            self.intersections = self._build_task(task_id)
            return self.state()

    def _propagate_corridor_flow(self) -> None:
        if self.task_id != "hard_corridor_emergency":
            return

        for upstream_idx, downstream_idx in ((0, 1), (1, 2)):
            upstream = self.intersections[upstream_idx]
            downstream = self.intersections[downstream_idx]
            flow = upstream.departures["N_S_Straight"]
            if flow <= 0:
                continue

            queue_before = downstream.queues["N_S_Straight"]
            downstream.queues["N_S_Straight"] += flow

            if (
                upstream.released_emergency["N_S_Straight"]
                and downstream.emergency_positions["N_S_Straight"] is None
            ):
                offset = upstream.released_emergency_offset["N_S_Straight"] or 0
                downstream.emergency_positions["N_S_Straight"] = queue_before + offset

    def _count_cleared_emergencies(self) -> int:
        cleared = 0
        for index, inter in enumerate(self.intersections):
            for lane, released in inter.released_emergency.items():
                if not released:
                    continue
                if (
                    self.task_id == "hard_corridor_emergency"
                    and lane == "N_S_Straight"
                    and index < len(self.intersections) - 1
                ):
                    continue
                cleared += 1
        return cleared

    def _collect_metrics(self) -> Dict[str, float]:
        total_queue = 0
        max_queue = 0
        total_wait = 0.0
        lane_count = 0
        emergency_active_count = 0
        emergency_wait_sum = 0.0
        emergency_spawned = 0

        for inter in self.intersections:
            emergency_spawned += inter.spawned_emergencies_step
            for lane in LANE_GROUPS:
                queue = inter.queues[lane]
                wait = inter.wait_times[lane]
                total_queue += queue
                max_queue = max(max_queue, queue)
                total_wait += wait
                lane_count += 1
                if inter.emergency_positions[lane] is not None:
                    emergency_active_count += 1
                    emergency_wait_sum += inter.emergency_wait_steps[lane]

        average_wait = total_wait / max(1, lane_count)
        return {
            "total_queue": float(total_queue),
            "max_queue": float(max_queue),
            "average_wait": round(average_wait, 4),
            "emergency_active_count": float(emergency_active_count),
            "emergency_wait_sum": float(emergency_wait_sum),
            "emergency_spawned": float(emergency_spawned),
            "emergency_cleared": float(self._count_cleared_emergencies()),
        }

    def _build_reward(
        self,
        queue_cost: float,
        switch_cost: float,
        emergency_cost: float,
    ) -> RewardModel:
        total_cost = queue_cost + switch_cost + emergency_cost
        total_reward = clamp01(1.0 - total_cost)
        queue_component = clamp01(1.0 - queue_cost)
        switching_component = clamp01(1.0 - switch_cost)
        emergency_component = clamp01(1.0 - emergency_cost)
        return RewardModel(
            total=round(total_reward, 4),
            queue_component=round(queue_component, 4),
            switching_component=round(switching_component, 4),
            emergency_component=round(emergency_component, 4),
        )

    def step(self, action: TrafficAction) -> StepResult:
        with self._lock:
            if self.step_count >= self.max_steps:
                observation = self.state()
                return StepResult(
                    observation=observation,
                    reward=0.0,
                    done=True,
                    info={"task_id": self.task_id, "seed": self.seed, "session_id": self.session_id},
                )

            action_map = {item.intersection_id: item.phase for item in action.actions}
            total_queue_cost = 0.0
            total_switch_cost = 0.0
            total_emergency_cost = 0.0

            for inter in self.intersections:
                phase = action_map.get(inter.id, inter.current_phase)
                queue_cost, switch_cost, emergency_cost = inter.step(phase)
                total_queue_cost += queue_cost
                total_switch_cost += switch_cost
                total_emergency_cost += emergency_cost

            self._propagate_corridor_flow()
            self.step_count += 1

            reward_breakdown = self._build_reward(
                total_queue_cost,
                total_switch_cost,
                total_emergency_cost,
            )
            done = self.step_count >= self.max_steps
            metrics = self._collect_metrics()

            observation = self.state()
            observation.reward_breakdown = reward_breakdown
            info = {
                "task_id": self.task_id,
                "seed": self.seed,
                "session_id": self.session_id,
                "reward_breakdown": reward_breakdown.model_dump(),
                "metrics": metrics,
            }

            self.history.append(
                {
                    "step": self.step_count,
                    "seed": self.seed,
                    "reward": reward_breakdown.total,
                    "queue_component": reward_breakdown.queue_component,
                    "switching_component": reward_breakdown.switching_component,
                    "emergency_component": reward_breakdown.emergency_component,
                    "metrics": metrics,
                    "state": [item.model_dump() for item in observation.intersections],
                }
            )

            return StepResult(
                observation=observation,
                reward=reward_breakdown.total,
                done=done,
                info=info,
            )

    def state(self) -> TrafficObservation:
        with self._lock:
            return TrafficObservation(
                session_id=self.session_id,
                task_id=self.task_id,
                seed=self.seed,
                step=self.step_count,
                max_steps=self.max_steps,
                intersections=[inter.to_observation() for inter in self.intersections],
                reward_breakdown=None,
            )

    def history_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.history]


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, TrafficCorridorEnv] = {
            DEFAULT_SESSION_ID: TrafficCorridorEnv(DEFAULT_SESSION_ID)
        }

    def get_or_create(self, session_id: str) -> TrafficCorridorEnv:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = TrafficCorridorEnv(session_id)
            return self._sessions[session_id]

    def reset(self, session_id: str, task_id: str, seed: Optional[int]) -> TrafficCorridorEnv:
        env = self.get_or_create(session_id)
        env.reset(task_id, seed)
        return env


session_store = SessionStore()


def _resolve_session_id(
    request: Request,
    body_session_id: Optional[str] = None,
    *,
    create_if_missing: bool = False,
) -> str:
    query_session_id = request.query_params.get("session_id")
    header_session_id = request.headers.get("X-Session-Id")
    session_id = (body_session_id or query_session_id or header_session_id or "").strip()
    if session_id:
        return session_id
    if create_if_missing:
        return str(uuid.uuid4())
    return DEFAULT_SESSION_ID


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request, payload: Optional[ResetPayload] = None) -> ResetResponse:
    task_id = payload.task_id if payload else request.query_params.get("task_id", DEFAULT_TASK_ID)
    seed = payload.seed if payload else None
    session_id = _resolve_session_id(
        request,
        payload.session_id if payload else None,
        create_if_missing=True,
    )

    try:
        env = session_store.reset(session_id, task_id, seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ResetResponse(
        status="success",
        message=f"Reset session {session_id} to {task_id}",
        session_id=session_id,
        state=env.state(),
    )


@app.get("/state", response_model=StateResponse)
def get_state(request: Request) -> StateResponse:
    session_id = _resolve_session_id(request)
    env = session_store.get_or_create(session_id)
    return StateResponse(**env.state().model_dump())


@app.get("/history", response_model=HistoryResponse)
def get_history(request: Request) -> HistoryResponse:
    session_id = _resolve_session_id(request)
    env = session_store.get_or_create(session_id)
    return HistoryResponse(session_id=session_id, history=env.history_snapshot())


def _parse_inference_output(output: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current_task = ""
    current_seed = 0
    for line in output.splitlines():
        if line.startswith("[START]"):
            current_task = ""
            current_seed = 0
            for token in line.split()[1:]:
                if token.startswith("task="):
                    current_task = token.split("=", 1)[1]
                elif token.startswith("seed="):
                    try:
                        current_seed = int(token.split("=", 1)[1])
                    except ValueError:
                        current_seed = 0
            continue

        if not line.startswith("[END]"):
            continue

        payload: Dict[str, Any] = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            payload[key] = value

        rewards_text = payload.get("rewards", "")
        rewards: List[float] = []
        if rewards_text:
            for item in rewards_text.split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    rewards.append(float(item))
                except ValueError:
                    continue

        try:
            run_seed = int(payload.get("seed", str(current_seed)) or current_seed)
        except ValueError:
            run_seed = current_seed

        runs.append(
            {
                "task_id": current_task,
                "seed": run_seed,
                "success": payload.get("success", "false").lower() == "true",
                "steps": int(payload.get("steps", "0") or 0),
                "score": float(payload.get("score", "0") or 0.0),
                "rewards": rewards,
            }
        )

    return runs


@app.post("/run-inference", response_model=InferenceRunResponse)
def run_inference(request: Request) -> InferenceRunResponse:
    project_root = Path(__file__).resolve().parent
    runtime_env = os.environ.copy()
    runtime_env["ENV_BASE_URL"] = str(request.base_url).rstrip("/")
    runtime_env.setdefault("API_BASE_URL", os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    runtime_env.setdefault("MODEL_NAME", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
    runtime_env.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

    try:
        completed = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=project_root,
            env=runtime_env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="inference.py timed out") from exc

    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    runs = _parse_inference_output(output)
    scores = [entry["score"] for entry in runs]
    task_ids = {entry["task_id"] for entry in runs}
    summary = {
        "task_count": len(task_ids),
        "run_count": len(runs),
        "average_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "best_score": round(max(scores), 3) if scores else 0.0,
        "worst_score": round(min(scores), 3) if scores else 0.0,
    }
    return InferenceRunResponse(
        success=completed.returncode == 0,
        summary=summary,
        runs=[InferenceRunEntry(**entry) for entry in runs],
        output=output,
    )


@app.post("/step", response_model=StepResponse)
def step(payload: TrafficAction, request: Request) -> StepResponse:
    session_id = _resolve_session_id(request)
    env = session_store.get_or_create(session_id)
    result = env.step(payload)
    return StepResponse(
        session_id=session_id,
        observation=result.observation,
        reward=round(result.reward, 4),
        done=result.done,
        info=result.info,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    try:
        template = UI_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return HTMLResponse(
            content="<h1>Traffic Corridor Pro</h1><p>UI template is missing.</p>",
            status_code=500,
        )

    html = template.replace("__DEFAULT_SPACE_URL__", json.dumps(DEFAULT_HF_SPACE_URL))
    return HTMLResponse(content=html)
