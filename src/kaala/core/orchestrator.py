"""Orchestrator for the Kaala agent pipeline.

This module coordinates the flow between agents:
    User Input → Niyati (route) → [Iccha OR Karma] → [Karya if goals] → Store scheduled prompts
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from kaala.agent.factory import AgentFactory
from kaala.core.tools import ToolRegistry
from kaala.tools.reminder import ReminderTool
from kaala.tools.message import MessageTool
from kaala.storage.database import get_db
from kaala.storage.repositories import GoalRepository, ScheduleRepository, HistoryRepository
from kaala.storage.models import Goal

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the agent pipeline and manages state."""

    def __init__(self, model: str = "GLM-5-CLOUD"):
        """Initialize the orchestrator with all agents.

        Args:
            model: The model to use for all agents.
        """
        self.niyati = AgentFactory.create(name="Niyati", model=model)
        self.iccha = AgentFactory.create(name="Iccha", model=model)
        self.karya = AgentFactory.create(name="Karya", model=model)
        self.karma = AgentFactory.create(name="Karma", model=model)
        self.tools = ToolRegistry()
        self.tools.register(ReminderTool())
        self.tools.register(MessageTool())
        self._db = None

    async def _get_db(self):
        """Get the database instance, initializing if needed."""
        if self._db is None:
            self._db = await get_db()
        return self._db

    async def _get_session(self) -> AsyncSession:
        """Get a new database session."""
        db = await self._get_db()
        return db.async_session()

    async def _get_context(self, agent_name: str, limit: int = 10) -> list[dict] | None:
        """Fetch recent conversation history for an agent.

        Args:
            agent_name: The agent to fetch context for.
            limit: Number of recent entries to include.

        Returns:
            List of {role, content} dicts, or None if no history.
        """
        try:
            async with await self._get_session() as session:
                history_repo = HistoryRepository(session)
                context = await history_repo.get_context_for_agent(agent_name, limit=limit)
            return context if context else None
        except Exception:
            logger.exception("Failed to fetch context for %s", agent_name)
            return None

    async def process_user_input(self, user_input: str) -> dict[str, Any]:
        """Process user input through the agent pipeline.

        Args:
            user_input: Raw user message.

        Returns:
            dict: Result containing the final response and any actions taken.
        """
        # Save user input to history
        try:
            async with await self._get_session() as session:
                history_repo = HistoryRepository(session)
                await history_repo.save(agent_name="User", role="user", content=user_input)
        except Exception:
            logger.exception("Failed to save user input to history")

        # Step 1: Niyati routes the input
        niyati_context = await self._get_context("Niyati")
        niyati_response = await self.niyati.chat_async(user_input, context=niyati_context)
        if not niyati_response:
            return {"error": "Empty Niyati response", "raw": niyati_response}

        route_data = self._as_dict(self._parse_json(niyati_response))

        if route_data is None:
            return {"error": "Failed to parse Niyati response", "raw": niyati_response}

        route_to = route_data.get("route_to")
        prompt = route_data.get("user_prompt")
        prompt_text = str(prompt) if prompt is not None else user_input

        # Step 2: Route to appropriate agent
        if route_to == "Iccha":
            result = await self._handle_iccha_flow(prompt_text, user_input)
        elif route_to == "Karma":
            result = await self._handle_karma_flow(prompt_text)
        elif route_to == "Karya":
            # Karya output should go to Karma for execution
            result = await self._handle_karma_flow(prompt_text)
        else:
            result = {"error": f"Unknown route: {route_to}"}

        return result

    async def _handle_iccha_flow(
        self, prompt: str, original_input: str
    ) -> dict[str, Any]:
        """Handle the Iccha → Karya/Karma flow for goal extraction and scheduling.

        Routes goals based on urgency and context:
        - needs_clarification=True → return clarification response to user
        - urgency=immediate → route to Karma for immediate execution
        - urgency=soon/later with sufficient context → route to Karya for scheduling

        Args:
            prompt: The user prompt (formatted for Iccha).
            original_input: Original user input for history.

        Returns:
            dict: Result containing the final response and any actions taken.
        """
        # Iccha extracts goals
        iccha_context = await self._get_context("Iccha")
        iccha_response = await self.iccha.chat_async(prompt, context=iccha_context)
        if not iccha_response:
            return {"error": "Empty Iccha response", "raw": iccha_response}

        iccha_data = self._parse_json(iccha_response)

        if iccha_data is None:
            return {"error": "Failed to parse Iccha response", "raw": iccha_response}

        # Single-object conversational response (no goals detected).
        if isinstance(iccha_data, dict) and not iccha_data.get("goal_detected", False):
            response_text = iccha_data.get("response", "")
            try:
                async with await self._get_session() as session:
                    history_repo = HistoryRepository(session)
                    await history_repo.save(
                        agent_name="Iccha", role="assistant", content=response_text
                    )
            except Exception:
                logger.exception("Failed to save Iccha conversation response")
            return {
                "type": "conversation",
                "response": response_text,
                "signature": "Iccha",
            }

        # List-based responses are common because the response schema is list[...].
        raw_items = self._as_list(iccha_data)
        goals_list = [
            item
            for item in raw_items
            if isinstance(item, dict) and item.get("goal_detected")
        ]

        if not goals_list:
            response_text = ""
            for item in raw_items:
                if isinstance(item, dict) and item.get("response"):
                    response_text = str(item["response"])
                    break

            try:
                async with await self._get_session() as session:
                    history_repo = HistoryRepository(session)
                    await history_repo.save(
                        agent_name="Iccha", role="assistant", content=response_text
                    )
            except Exception:
                logger.exception("Failed to save Iccha response")

            return {
                "type": "conversation",
                "response": response_text,
                "signature": "Iccha",
            }

        # Classify goals by urgency and clarification needs.
        clarification_goals = []
        immediate_goals = []
        scheduled_goals = []

        for goal_data in goals_list:
            if not isinstance(goal_data, dict):
                continue
            if not goal_data.get("goal_detected"):
                continue

            if goal_data.get("needs_clarification"):
                clarification_goals.append(goal_data)
            elif goal_data.get("urgency") == "immediate":
                immediate_goals.append(goal_data)
            else:
                scheduled_goals.append(goal_data)

        # Handle clarification goals: return the clarifying question to the user.
        combined_response = ""
        if clarification_goals:
            clarification_texts = []
            for g in clarification_goals:
                text = g.get("response", g.get("details", ""))
                if text:
                    clarification_texts.append(text)

            combined_response = " ".join(clarification_texts) if clarification_texts else ""

            try:
                async with await self._get_session() as session:
                    history_repo = HistoryRepository(session)
                    await history_repo.save(
                        agent_name="Iccha", role="assistant", content=combined_response
                    )
            except Exception:
                logger.exception("Failed to save clarification response")

            # If all goals need clarification, return early.
            if not immediate_goals and not scheduled_goals:
                return {
                    "type": "clarification",
                    "response": combined_response,
                    "goals": [g.get("goal", "") for g in clarification_goals],
                    "signature": "Iccha",
                }

        # Handle immediate goals: route directly to Karma.
        immediate_results = []
        for goal_data in immediate_goals:
            goal_text = goal_data.get("goal", "")
            result = await self._handle_karma_flow(
                f"Execute immediately: {goal_text}"
            )
            immediate_results.append({"goal": goal_text, "result": result})

        # Store immediate goals in database.
        for goal_data in immediate_goals:
            try:
                async with await self._get_session() as session:
                    goal_repo = GoalRepository(session)
                    await goal_repo.create(
                        goal_text=goal_data.get("goal", ""),
                        details=goal_data.get("details"),
                        status="in_progress",
                    )
            except Exception:
                logger.exception("Failed to store immediate goal")

        # If no goals need scheduling, return immediate results.
        if not scheduled_goals:
            response_parts = []
            if clarification_goals:
                response_parts.append(combined_response)
            for ir in immediate_results:
                response_parts.append(f"Executed: {ir['goal']}")

            return {
                "type": "immediate",
                "goals": [g.get("goal", "") for g in immediate_goals],
                "clarification": combined_response if clarification_goals else None,
                "results": immediate_results,
                "message": "Immediate action(s) taken.",
            }

        # Store scheduled goals in database and pass to Karya.
        stored_goals: list[Goal] = []
        try:
            async with await self._get_session() as session:
                goal_repo = GoalRepository(session)
                history_repo = HistoryRepository(session)

                for goal_data in scheduled_goals:
                    if not isinstance(goal_data, dict):
                        continue
                    if goal_data.get("goal_detected"):
                        goal = await goal_repo.create(
                            goal_text=goal_data.get("goal", ""),
                            details=goal_data.get("details"),
                            status="pending",
                        )
                        stored_goals.append(goal)

                await history_repo.save(
                    agent_name="Iccha", role="assistant", content=json.dumps(goals_list)
                )
        except Exception:
            logger.exception("Failed to store scheduled goals")

        # Pass to Karya for scheduling (only scheduled goals, not immediate ones)
        karya_input = json.dumps(scheduled_goals)
        karya_context = await self._get_context("Karya")
        karya_response = await self.karya.chat_async(karya_input, context=karya_context)
        if not karya_response:
            return {
                "type": "goals_extracted",
                "goals": [g.goal_text for g in stored_goals],
                "error": "Empty Karya response",
                "raw": karya_response,
            }

        karya_data = self._parse_json(karya_response)

        if karya_data is None:
            return {
                "type": "goals_extracted",
                "goals": [g.goal_text for g in stored_goals],
                "error": "Failed to parse Karya response",
                "raw": karya_response,
            }

        # Store scheduled prompts
        prompts_list = self._as_list(karya_data)
        scheduled_count = 0

        try:
            async with await self._get_session() as session:
                schedule_repo = ScheduleRepository(session)
                history_repo = HistoryRepository(session)

                for prompt_data in prompts_list:
                    if not isinstance(prompt_data, dict):
                        continue
                    timestamp_str = prompt_data.get("timestamp")
                    if timestamp_str:
                        try:
                            scheduled_for = datetime.fromisoformat(
                                timestamp_str.replace(" ", "T")
                            )
                        except ValueError:
                            scheduled_for = datetime.fromisoformat(timestamp_str)

                        # Find matching goal
                        goal_id = None
                        goal_text = prompt_data.get("goal", "")
                        for g in stored_goals:
                            if g.goal_text == goal_text:
                                goal_id = g.id
                                break

                        await schedule_repo.create(
                            prompt_text=prompt_data.get("prompt", ""),
                            scheduled_for=scheduled_for,
                            goal_id=goal_id,
                            prompt_type=prompt_data.get("prompt_type", "check_in"),
                            status="pending",
                        )
                        scheduled_count += 1

                await history_repo.save(
                    agent_name="Karya", role="assistant", content=json.dumps(prompts_list)
                )
        except Exception:
            logger.exception("Failed to store scheduled prompts")

        result = {
            "type": "goals_scheduled",
            "goals": [g.goal_text for g in stored_goals],
            "scheduled_prompts": scheduled_count,
            "message": f"Detected {len(stored_goals)} goal(s) and scheduled {scheduled_count} prompt(s).",
        }
        if immediate_results:
            result["immediate_results"] = immediate_results
        if clarification_goals:
            result["clarification"] = combined_response
        return result

    async def _handle_karma_flow(self, prompt: Any) -> dict[str, Any]:
        """Handle immediate execution via Karma.

        Args:
            prompt: The prompt for Karma (can be str or dict).

        Returns:
            dict: Result of the execution.
        """
        prompt_str = (
            json.dumps(prompt) if isinstance(prompt, (dict, list)) else str(prompt)
        )
        karma_context = await self._get_context("Karma")
        karma_response = await self.karma.chat_async(prompt_str, context=karma_context)
        if not karma_response:
            return {
                "type": "executed",
                "action": None,
                "result": "",
                "error": "Empty Karma response",
            }

        karma_data = self._as_dict(self._parse_json(karma_response))

        # Try to execute the requested tool
        tool_result = None
        tool_name = karma_data.get("tool") if karma_data else None
        if tool_name:
            tool = self.tools.get(tool_name)
            if tool:
                parameters = karma_data.get("parameters", {})
                tool_result = await tool.execute(parameters)

        try:
            async with await self._get_session() as session:
                history_repo = HistoryRepository(session)
                await history_repo.save(
                    agent_name="Karma", role="assistant", content=karma_response
                )
        except Exception:
            logger.exception("Failed to save Karma response to history")

        result = {
            "type": "executed",
            "action": karma_data.get("action") if karma_data else None,
            "tool": tool_name,
            "parameters": karma_data.get("parameters", {}) if karma_data else {},
            "result": karma_response,
        }
        if tool_result:
            result["tool_result"] = tool_result

        return result

    @staticmethod
    def _as_list(data: Any) -> list[Any]:
        """Normalize parsed payload into a list for uniform processing."""
        if isinstance(data, list):
            return data
        if data is None:
            return []
        return [data]

    @staticmethod
    def _as_dict(data: Any) -> dict[str, Any] | None:
        """Normalize parsed payload into a dict.

        Some model calls return a one-item list due to list-based response schema.
        """
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first
        return None

    def _parse_json(self, response: str | None) -> Any:
        """Parse JSON from LLM response.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed JSON or None if parsing fails.
        """
        try:
            # Handle potential markdown code blocks
            if response is None:
                return None

            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove markdown code block
                lines = cleaned.split("\n")
                if len(lines) > 2:
                    cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", response[:200] if response else "None")
            return None

    async def get_pending_goals(self) -> list[dict]:
        """Get all pending goals.

        Returns:
            list: List of pending goals.
        """
        try:
            async with await self._get_session() as session:
                goal_repo = GoalRepository(session)
                goals = await goal_repo.get_pending()
                return [
                    {"id": g.id, "goal": g.goal_text, "details": g.details} for g in goals
                ]
        except Exception:
            logger.exception("Failed to fetch pending goals")
            return []

    async def get_scheduled_prompts(self) -> list[dict]:
        """Get all pending scheduled prompts.

        Returns:
            list: List of pending scheduled prompts.
        """
        try:
            async with await self._get_session() as session:
                schedule_repo = ScheduleRepository(session)
                prompts = await schedule_repo.get_pending()
                return [
                    {
                        "id": p.id,
                        "prompt": p.prompt_text,
                        "scheduled_for": p.scheduled_for.isoformat(),
                        "goal_id": p.goal_id,
                    }
                    for p in prompts
                ]
        except Exception:
            logger.exception("Failed to fetch scheduled prompts")
            return []

    async def get_conversation_history(
        self, agent_name: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Get conversation history.

        Args:
            agent_name: Optional filter by agent name.
            limit: Maximum number of entries to return.

        Returns:
            list: List of history entries.
        """
        try:
            async with await self._get_session() as session:
                history_repo = HistoryRepository(session)
                entries = await history_repo.get_recent(agent_name=agent_name, limit=limit)
                return [
                    {
                        "agent": e.agent_name,
                        "role": e.role,
                        "content": e.content,
                        "timestamp": e.timestamp.isoformat(),
                    }
                    for e in entries
                ]
        except Exception:
            logger.exception("Failed to fetch conversation history")
            return []