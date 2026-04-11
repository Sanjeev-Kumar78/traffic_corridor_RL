import json
import os
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Traffic Corridor Pro")
DEFAULT_HF_SPACE_URL = os.getenv("HF_SPACE_URL", "").strip()


LANE_GROUPS = ["N_S_Straight", "N_S_Left", "E_W_Straight", "E_W_Left"]
TASK_IDS = ["easy_4_phase", "medium_asymmetric", "hard_corridor_emergency"]
DEFAULT_TASK_ID = "easy_4_phase"
TASK_SEEDS = {
    "easy_4_phase": 17,
    "medium_asymmetric": 29,
    "hard_corridor_emergency": 43,
}
TASK_MAX_STEPS = {
    "easy_4_phase": 100,
    "medium_asymmetric": 120,
    "hard_corridor_emergency": 140,
}
SWITCH_COOLDOWN_STEPS = 2
DISCHARGE_RATE = 3
EMERGENCY_SPAWN_RATE = 0.03


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    task_id: str
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
    state: TrafficObservation


class StateResponse(TrafficObservation):
    pass


class StepResponse(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]


class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]


class _Intersection:
    def __init__(
        self,
        intersection_id: int,
        rates: Dict[str, float],
        rng: random.Random,
        enable_emergencies: bool = False,
    ) -> None:
        self.id = intersection_id
        self.rates = rates
        self.rng = rng
        self.enable_emergencies = enable_emergencies

        self.queues = {lane: 0 for lane in LANE_GROUPS}
        self.wait_times = {lane: 0.0 for lane in LANE_GROUPS}
        self.emergency_positions = {lane: None for lane in LANE_GROUPS}
        self.emergency_wait_steps = {lane: 0 for lane in LANE_GROUPS}
        self.departures = {lane: 0 for lane in LANE_GROUPS}
        self.released_emergency = {lane: False for lane in LANE_GROUPS}
        self.released_emergency_offset = {lane: None for lane in LANE_GROUPS}

        self.current_phase = 0
        self.switch_cooldown = 0

    def _spawn_vehicles(self) -> None:
        for lane in LANE_GROUPS:
            if self.rng.random() < self.rates[lane]:
                queue_before = self.queues[lane]
                self.queues[lane] += 1
                if (
                    self.enable_emergencies
                    and self.emergency_positions[lane] is None
                    and self.rng.random() < EMERGENCY_SPAWN_RATE
                ):
                    self.emergency_positions[lane] = queue_before

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
            self.wait_times[lane] += queue
            queue_cost += 0.018 * queue + 0.0035 * (queue * queue)

            has_emergency = self.emergency_positions[lane] is not None
            if has_emergency:
                self.emergency_wait_steps[lane] += 1
                emergency_cost += 0.08 * \
                    (1.35 ** (self.emergency_wait_steps[lane] - 1))
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

    def __init__(self) -> None:
        self.rng = random.Random()
        self.task_id = DEFAULT_TASK_ID
        self.step_count = 0
        self.max_steps = TASK_MAX_STEPS[DEFAULT_TASK_ID]
        self.intersections: List[_Intersection] = []
        self.history: List[Dict[str, Any]] = []
        self.reset(DEFAULT_TASK_ID)

    def _make_intersection(
        self,
        intersection_id: int,
        rates: Dict[str, float],
        enable_emergencies: bool = False,
    ) -> _Intersection:
        return _Intersection(intersection_id, rates, self.rng, enable_emergencies)

    def reset(self, task_id: str = DEFAULT_TASK_ID) -> TrafficObservation:
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task: {task_id}")

        self.task_id = task_id
        self.step_count = 0
        self.max_steps = TASK_MAX_STEPS[task_id]
        self.history = []
        self.intersections = []
        self.rng.seed(TASK_SEEDS[task_id])

        if task_id == "easy_4_phase":
            rates = {lane: 0.16 for lane in LANE_GROUPS}
            self.intersections.append(self._make_intersection(0, rates, False))

        elif task_id == "medium_asymmetric":
            rates = {
                "N_S_Straight": 0.62,
                "N_S_Left": 0.08,
                "E_W_Straight": 0.18,
                "E_W_Left": 0.07,
            }
            self.intersections.append(self._make_intersection(0, rates, False))

        elif task_id == "hard_corridor_emergency":
            self.intersections.append(
                self._make_intersection(
                    0,
                    {
                        "N_S_Straight": 0.40,
                        "N_S_Left": 0.12,
                        "E_W_Straight": 0.15,
                        "E_W_Left": 0.08,
                    },
                    True,
                )
            )
            self.intersections.append(
                self._make_intersection(
                    1,
                    {
                        "N_S_Straight": 0.05,
                        "N_S_Left": 0.10,
                        "E_W_Straight": 0.14,
                        "E_W_Left": 0.08,
                    },
                    True,
                )
            )
            self.intersections.append(
                self._make_intersection(
                    2,
                    {
                        "N_S_Straight": 0.04,
                        "N_S_Left": 0.09,
                        "E_W_Straight": 0.13,
                        "E_W_Left": 0.08,
                    },
                    True,
                )
            )

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
        if self.step_count >= self.max_steps:
            observation = self.state()
            return StepResult(observation=observation, reward=0.0, done=True, info={})

        action_map = {
            item.intersection_id: item.phase for item in action.actions}
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

        observation = self.state()
        observation.reward_breakdown = reward_breakdown
        info = {
            "task_id": self.task_id,
            "reward_breakdown": reward_breakdown.model_dump(),
        }

        self.history.append(
            {
                "step": self.step_count,
                "reward": reward_breakdown.total,
                "queue_component": reward_breakdown.queue_component,
                "switching_component": reward_breakdown.switching_component,
                "emergency_component": reward_breakdown.emergency_component,
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
        return TrafficObservation(
            task_id=self.task_id,
            step=self.step_count,
            max_steps=self.max_steps,
            intersections=[inter.to_observation()
                           for inter in self.intersections],
            reward_breakdown=None,
        )


env = TrafficCorridorEnv()


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request) -> ResetResponse:
    task_id = request.query_params.get("task_id", DEFAULT_TASK_ID)
    try:
        body = await request.json()
    except Exception:
        body = None

    if isinstance(body, dict):
        task_id = body.get("task_id", task_id)

    try:
        state = env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ResetResponse(
        status="success",
        message=f"Reset to {task_id}",
        state=state,
    )


@app.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    return StateResponse(**env.state().model_dump())


@app.get("/history", response_model=HistoryResponse)
def get_history() -> HistoryResponse:
    return HistoryResponse(history=env.history)


@app.post("/step", response_model=StepResponse)
def step(payload: TrafficAction) -> StepResponse:
    result = env.step(payload)
    return StepResponse(
        reward=round(result.reward, 4),
        done=result.done,
        info=result.info,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Traffic Corridor Pro UI</title>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; max-width: 1080px; }}
        h1, h2 {{ margin-bottom: 8px; }}
        .row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
        .card {{ border: 1px solid #d7d7d7; border-radius: 8px; padding: 12px; margin: 12px 0; }}
        input, select, textarea, button {{ font-size: 14px; padding: 6px; }}
        textarea {{ width: 100%; min-height: 120px; font-family: Consolas, monospace; }}
        pre {{ background: #f6f6f6; border: 1px solid #e3e3e3; border-radius: 8px; padding: 10px; overflow: auto; }}
        .links a {{ margin-right: 10px; }}
        .muted {{ color: #666; font-size: 13px; }}
    </style>
</head>
<body>
    <h1>Traffic Corridor Pro</h1>
    <p class="muted">Interactive controls + direct route links for Hugging Face Space endpoints.</p>

    <div class="card">
        <h2>Target Base URL</h2>
        <div class="row">
            <input id="baseUrl" size="70" />
            <button onclick="useCurrentOrigin()">Use Current Origin</button>
            <button onclick="applyBaseUrl()">Apply</button>
        </div>
        <p class="muted">Set this to your HF Space URL for direct routing, e.g. https://your-space.hf.space</p>
        <div class="links" id="routeLinks"></div>
    </div>

    <div class="card">
        <h2>Reset</h2>
        <div class="row">
            <select id="taskId">
                <option value="easy_4_phase">easy_4_phase</option>
                <option value="medium_asymmetric">medium_asymmetric</option>
                <option value="hard_corridor_emergency">hard_corridor_emergency</option>
            </select>
            <button onclick="resetTask()">POST /reset</button>
        </div>
    </div>

    <div class="card">
        <h2>State / History</h2>
        <div class="row">
            <button onclick="getState()">GET /state</button>
            <button onclick="getHistory()">GET /history</button>
        </div>
    </div>

    <div class="card">
        <h2>Step</h2>
        <p class="muted">Send action payload JSON for POST /step</p>
        <textarea id="stepPayload">{{
    "actions": [
        {{ "intersection_id": 0, "phase": 0 }}
    ]
}}</textarea>
        <div class="row">
            <button onclick="postStep()">POST /step</button>
        </div>
    </div>

    <div class="card">
        <h2>Response</h2>
        <pre id="output">Ready.</pre>
    </div>

<script>
    const defaultSpaceUrl = {json.dumps(DEFAULT_HF_SPACE_URL)};

    function getBaseUrl() {{
        const value = document.getElementById('baseUrl').value.trim();
        return value || window.location.origin;
    }}

    function applyBaseUrl() {{
        const base = getBaseUrl();
        const links = [
            ['Open /reset route', base + '/reset'],
            ['Open /state route', base + '/state'],
            ['Open /history route', base + '/history'],
            ['Open /step route', base + '/step'],
        ];
        document.getElementById('routeLinks').innerHTML = links
            .map(([label, url]) => `<a href="${{url}}" target="_blank" rel="noreferrer">${{label}}</a>`)
            .join('');
    }}

    function useCurrentOrigin() {{
        document.getElementById('baseUrl').value = window.location.origin;
        applyBaseUrl();
    }}

    async function requestJson(path, options = {{}}) {{
        const base = getBaseUrl();
        const res = await fetch(base + path, options);
        const text = await res.text();
        let payload = text;
        try {{ payload = JSON.parse(text); }} catch (_) {{}}
        document.getElementById('output').textContent = JSON.stringify(payload, null, 2);
    }}

    async function resetTask() {{
        const task = document.getElementById('taskId').value;
        await requestJson('/reset', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ task_id: task }})
        }});
    }}

    async function getState() {{
        await requestJson('/state');
    }}

    async function getHistory() {{
        await requestJson('/history');
    }}

    async function postStep() {{
        const raw = document.getElementById('stepPayload').value;
        let payload;
        try {{
            payload = JSON.parse(raw);
        }} catch (e) {{
            document.getElementById('output').textContent = 'Invalid JSON payload: ' + e.message;
            return;
        }}
        await requestJson('/step', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(payload),
        }});
    }}

    (function init() {{
        document.getElementById('baseUrl').value = defaultSpaceUrl || window.location.origin;
        applyBaseUrl();
    }})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
