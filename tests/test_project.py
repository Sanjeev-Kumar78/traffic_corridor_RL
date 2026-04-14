import unittest

from fastapi.testclient import TestClient

from environment import TrafficAction, TrafficCorridorEnv, app
from graders import grade_hard_corridor_emergency
from inference import build_eval_seeds


def _rollout_rewards(task_id: str, seed: int, steps: int = 8) -> list[float]:
    env = TrafficCorridorEnv()
    env.reset(task_id, seed=seed)
    rewards: list[float] = []
    for _ in range(steps):
        actions = [
            {"intersection_id": inter.id, "phase": inter.current_phase}
            for inter in env.intersections
        ]
        result = env.step(TrafficAction.model_validate({"actions": actions}))
        rewards.append(result.reward)
    return rewards


class TrafficCorridorProjectTests(unittest.TestCase):
    def test_seeded_reset_is_reproducible_and_varies(self) -> None:
        rewards_a = _rollout_rewards("medium_asymmetric", seed=123)
        rewards_b = _rollout_rewards("medium_asymmetric", seed=123)
        rewards_c = _rollout_rewards("medium_asymmetric", seed=124)

        self.assertEqual(rewards_a, rewards_b)
        self.assertNotEqual(rewards_a, rewards_c)

    def test_step_response_includes_observation_and_seed(self) -> None:
        client = TestClient(app)
        reset = client.post("/reset", json={"task_id": "easy_4_phase", "seed": 77})
        self.assertEqual(reset.status_code, 200)
        session_id = reset.json()["session_id"]

        step = client.post(
            f"/step?session_id={session_id}",
            json={"actions": [{"intersection_id": 0, "phase": 0}]},
        )
        self.assertEqual(step.status_code, 200)
        payload = step.json()
        self.assertEqual(payload["session_id"], session_id)
        self.assertIn("observation", payload)
        self.assertEqual(payload["observation"]["seed"], 77)

    def test_sessions_are_isolated(self) -> None:
        client = TestClient(app)
        session_a = client.post("/reset", json={"task_id": "easy_4_phase", "seed": 50}).json()["session_id"]
        session_b = client.post("/reset", json={"task_id": "easy_4_phase", "seed": 50}).json()["session_id"]

        client.post(
            f"/step?session_id={session_a}",
            json={"actions": [{"intersection_id": 0, "phase": 0}]},
        )

        state_a = client.get(f"/state?session_id={session_a}").json()
        state_b = client.get(f"/state?session_id={session_b}").json()
        self.assertEqual(state_a["step"], 1)
        self.assertEqual(state_b["step"], 0)

    def test_hard_grader_penalizes_non_ns_emergency_backlog(self) -> None:
        healthy_history = [
            {
                "reward": 0.9,
                "queue_component": 0.9,
                "state": [
                    {
                        "id": 0,
                        "current_phase": 0,
                        "lanes": {
                            "N_S_Straight": {"queue": 0, "emergency": False},
                            "N_S_Left": {"queue": 0, "emergency": False},
                            "E_W_Straight": {"queue": 0, "emergency": False},
                            "E_W_Left": {"queue": 0, "emergency": False},
                        },
                    }
                ],
                "metrics": {
                    "total_queue": 0,
                    "emergency_active_count": 0,
                    "emergency_wait_sum": 0,
                    "emergency_spawned": 0,
                    "emergency_cleared": 0,
                },
            }
        ]
        emergency_history = [
            {
                "reward": 0.9,
                "queue_component": 0.9,
                "state": [
                    {
                        "id": 0,
                        "current_phase": 0,
                        "lanes": {
                            "N_S_Straight": {"queue": 0, "emergency": False},
                            "N_S_Left": {"queue": 0, "emergency": False},
                            "E_W_Straight": {"queue": 0, "emergency": False},
                            "E_W_Left": {"queue": 2, "emergency": True},
                        },
                    }
                ],
                "metrics": {
                    "total_queue": 2,
                    "emergency_active_count": 1,
                    "emergency_wait_sum": 5,
                    "emergency_spawned": 1,
                    "emergency_cleared": 0,
                },
            }
        ]

        self.assertLess(
            grade_hard_corridor_emergency(emergency_history),
            grade_hard_corridor_emergency(healthy_history),
        )

    def test_default_eval_seed_schedule_has_multiple_runs(self) -> None:
        seeds = build_eval_seeds("hard_corridor_emergency")
        self.assertEqual(len(seeds), 3)
        self.assertEqual(seeds[0], 43)


if __name__ == "__main__":
    unittest.main()
