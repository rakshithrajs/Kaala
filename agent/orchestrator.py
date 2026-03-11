"""Runtime orchestrator to wire personas into a proactive loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.agent_factory import AgentFactory
from storage.task_store import TaskStore


@dataclass
class MessageResult:
    reply: str
    plans_created: int = 0


class ProactiveOrchestrator:
    """Coordinates Niyati -> Iccha -> Karya -> Karma flows."""

    def __init__(self, model: str = "GEMINI-1.5-PRO", db_path: str = "data/kaal.db"):
        self.niyati = AgentFactory.create("niyati", model)
        self.iccha = AgentFactory.create("iccha", model)
        self.karya = AgentFactory.create("karya", model)
        self.karma = AgentFactory.create("karma", model)
        self.store = TaskStore(db_path)

    async def process_user_message(self, user_message: str) -> MessageResult:
        routed = await self.niyati.generate_async_json(user_message)
        payload = routed if isinstance(routed, dict) else {"route_to": "Iccha", "user_prompt": user_message}

        iccha_input = json.dumps(payload, ensure_ascii=False)
        extracted = await self.iccha.generate_async_json(iccha_input)

        if isinstance(extracted, dict) and not extracted.get("goal_detected", False):
            return MessageResult(reply=extracted.get("response", "Got it."), plans_created=0)

        goals = extracted if isinstance(extracted, list) else []
        if not goals:
            return MessageResult(reply="I couldn't detect a clear long-term goal yet.", plans_created=0)

        plans = await self.karya.generate_async_json(json.dumps(goals, ensure_ascii=False))
        valid_plans = self._validate_plans(plans)
        inserted = self.store.add_plans(valid_plans)
        return MessageResult(
            reply=f"Got it — I scheduled {inserted} follow-up prompt(s) to proactively check in.",
            plans_created=inserted,
        )

    async def run_due_tasks_once(self) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for task in self.store.due_tasks():
            action_payload = {
                "goal": task.goal,
                "prompt": task.prompt,
                "timestamp": task.timestamp,
                "signature": "Karya",
            }
            result = await self.karma.generate_async_json(json.dumps(action_payload, ensure_ascii=False))
            status = "Completed" if result else "Failed"
            output = {"task_id": task.id, "status": status, "result": result}
            outputs.append(output)
            self.store.mark_executed(task.id, status, str(result), result)
        return outputs

    @staticmethod
    def _validate_plans(raw: Any) -> list[dict]:
        if not isinstance(raw, list):
            return []

        valid = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if all(key in item for key in ("goal", "prompt", "timestamp")):
                valid.append(
                    {
                        "goal": str(item["goal"]),
                        "prompt": str(item["prompt"]),
                        "timestamp": str(item["timestamp"]),
                    }
                )
        return valid
