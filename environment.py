import logging
import random
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Traffic Corridor Pro")
logging.basicConfig(level=logging.INFO)


class ActionConfig(BaseModel):
    intersection_id: int
    phase: Literal[0, 1, 2, 3]


class StepRequest(BaseModel):
    actions: List[ActionConfig]


class TaskRequest(BaseModel):
    task_id: str = "easy_4_phase"


class LaneState(BaseModel):
    queue: int
    wait_time: float
    emergency: bool


class IntersectionState(BaseModel):
    id: int
    lanes: Dict[str, LaneState]
    current_phase: Literal[0, 1, 2, 3]
    in_transition: bool
    switch_cooldown_remaining: int


class ResetResponse(BaseModel):
    status: str
    message: str
    state: Dict[str, Any]


class StateResponse(BaseModel):
    task_id: str
    step: int
    max_steps: int
    intersections: List[IntersectionState]


class StepResponse(BaseModel):
    reward: float
    done: bool
    info: Dict[str, Any] | str


class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]


LANE_GROUPS = ["N_S_Straight", "N_S_Left", "E_W_Straight", "E_W_Left"]
SWITCH_COOLDOWN_STEPS = 2
DISCHARGE_RATE = 3
EMERGENCY_SPAWN_RATE = 0.02
TASK_SEEDS = {
    "easy_4_phase": 17,
    "medium_asymmetric": 29,
    "hard_corridor_emergency": 43,
}
DEFAULT_TASK_ID = "easy_4_phase"


class Intersection:
    def __init__(
        self,
        i_id: int,
        rates: Dict[str, float],
        rng: random.Random,
        has_emergencies: bool = False,
    ):
        self.id = i_id
        self.rates = rates
        self.rng = rng
        self.has_emergencies = has_emergencies

        self.queues = {lane: 0 for lane in LANE_GROUPS}
        self.wait_times = {lane: 0.0 for lane in LANE_GROUPS}
        self.emergencies = {lane: False for lane in LANE_GROUPS}
        self.departures = {lane: 0 for lane in LANE_GROUPS}

        self.current_phase = 0
        self.switch_cooldown = 0
        self.total_arrivals = 0

    def step(self, target_phase: int) -> float:
        penalty = 0.0

        if target_phase != self.current_phase and self.switch_cooldown == 0:
            self.switch_cooldown = SWITCH_COOLDOWN_STEPS
            self.current_phase = target_phase
            penalty -= 5.0

        for lane in LANE_GROUPS:
            if self.rng.random() < self.rates[lane]:
                self.queues[lane] += 1
                self.total_arrivals += 1
                if self.has_emergencies and self.rng.random() < EMERGENCY_SPAWN_RATE:
                    self.emergencies[lane] = True

        self.departures = {lane: 0 for lane in LANE_GROUPS}
        if self.switch_cooldown == 0:
            active_lane = LANE_GROUPS[self.current_phase]
            cleared = min(self.queues[active_lane], DISCHARGE_RATE)
            self.queues[active_lane] -= cleared
            self.departures[active_lane] = cleared
            if self.queues[active_lane] == 0:
                self.emergencies[active_lane] = False

        for lane, count in self.queues.items():
            self.wait_times[lane] += count
            penalty -= count * 0.1
            if self.emergencies[lane]:
                penalty -= 20.0

        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        return penalty

    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "lanes": {
                lane: {
                    "queue": self.queues[lane],
                    "wait_time": round(self.wait_times[lane], 2),
                    "emergency": self.emergencies[lane],
                }
                for lane in LANE_GROUPS
            },
            "current_phase": self.current_phase,
            "in_transition": self.switch_cooldown > 0,
            "switch_cooldown_remaining": self.switch_cooldown,
        }


class TrafficSimulation:
    def __init__(self):
        self.intersections: List[Intersection] = []
        self.step_count = 0
        self.max_steps = 150
        self.task_id = ""
        self.history: List[Dict[str, Any]] = []
        self.rng = random.Random()

    def _make_intersection(
        self, i_id: int, rates: Dict[str, float], has_emergencies: bool = False
    ) -> Intersection:
        return Intersection(i_id, rates, self.rng, has_emergencies)

    def reset(self, task_id: str):
        self.step_count = 0
        self.task_id = task_id
        self.intersections = []
        self.history = []

        if task_id not in TASK_SEEDS:
            raise ValueError(f"Unknown task: {task_id}")

        self.rng.seed(TASK_SEEDS[task_id])

        if task_id == "easy_4_phase":
            rates = {lane: 0.15 for lane in LANE_GROUPS}
            self.intersections.append(self._make_intersection(0, rates))
        elif task_id == "medium_asymmetric":
            rates = {
                "N_S_Straight": 0.6,
                "N_S_Left": 0.05,
                "E_W_Straight": 0.1,
                "E_W_Left": 0.05,
            }
            self.intersections.append(self._make_intersection(0, rates))
        elif task_id == "hard_corridor_emergency":
            rates = {
                "N_S_Straight": 0.3,
                "N_S_Left": 0.1,
                "E_W_Straight": 0.1,
                "E_W_Left": 0.1,
            }
            self.intersections.append(self._make_intersection(0, rates, True))
            self.intersections.append(
                self._make_intersection(
                    1,
                    {
                        "N_S_Straight": 0.0,
                        "N_S_Left": 0.1,
                        "E_W_Straight": 0.1,
                        "E_W_Left": 0.1,
                    },
                    True,
                )
            )
            self.intersections.append(
                self._make_intersection(
                    2,
                    {
                        "N_S_Straight": 0.0,
                        "N_S_Left": 0.1,
                        "E_W_Straight": 0.1,
                        "E_W_Left": 0.1,
                    },
                    True,
                )
            )

    def step(self, actions: List[ActionConfig]):
        action_map = {
            action.intersection_id: action.phase for action in actions}
        reward = 0.0

        for inter in self.intersections:
            phase = action_map.get(inter.id, inter.current_phase)
            reward += inter.step(phase)

        if self.task_id == "hard_corridor_emergency":
            corridor_flow = [
                inter.departures["N_S_Straight"] for inter in self.intersections
            ]
            self.intersections[1].queues["N_S_Straight"] += corridor_flow[0]
            self.intersections[2].queues["N_S_Straight"] += corridor_flow[1]

        self.step_count += 1
        done = self.step_count >= self.max_steps

        self.history.append(
            {
                "step": self.step_count,
                "reward": reward,
                "state": [inter.get_state() for inter in self.intersections],
            }
        )

        return reward, done


sim = TrafficSimulation()
sim.reset("easy_4_phase")


def build_state_response() -> Dict[str, Any]:
    return {
        "task_id": sim.task_id,
        "step": sim.step_count,
        "max_steps": sim.max_steps,
        "intersections": [inter.get_state() for inter in sim.intersections],
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(req: Dict[str, Any] = None):
    # This handles empty bodies, missing fields, and query params manually
    task_id = DEFAULT_TASK_ID
    if req and isinstance(req, dict):
        task_id = req.get("task_id", DEFAULT_TASK_ID)

    try:
        sim.reset(task_id)
        return {
            "status": "success",
            "message": f"Reset to {task_id}",
            "state": build_state_response(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def get_state():
    return build_state_response()


@app.get("/history", response_model=HistoryResponse)
def get_history():
    return {"history": sim.history}


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if sim.step_count >= sim.max_steps:
        return {"reward": 0.0, "done": True, "info": "Episode finished"}

    reward, done = sim.step(req.actions)
    return {"reward": round(reward, 2), "done": done, "info": {}}
