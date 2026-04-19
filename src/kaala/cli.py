"""CLI entrypoints for Kaala."""

import asyncio
import sys
from typing import Any

from kaala.config.settings import get_settings
from kaala.core.orchestrator import Orchestrator
from kaala.core.scheduler import PromptScheduler


async def spinner(message: str = "Thinking") -> None:
    """Render a small non-blocking spinner while work is in progress."""
    frames = "|/-\\"
    index = 0
    while True:
        print(f"\r{message} {frames[index % len(frames)]}", end="", flush=True)
        index += 1
        await asyncio.sleep(0.12)


def print_result(result: dict[str, Any]) -> None:
    """Print orchestrator output in a CLI-friendly format."""
    if "error" in result:
        print(f"\nKaala: Error -> {result['error']}")
        raw = result.get("raw")
        if raw:
            print(f"Raw response: {raw}")
        return

    result_type = result.get("type")

    if result_type == "conversation":
        signature = result.get("signature", "Assistant")
        print(f"\n{signature}: {result.get('response', '')}")
        return

    if result_type == "clarification":
        print(f"\nIccha: {result.get('response', '')}")
        return

    if result_type == "immediate":
        goals = result.get("goals", [])
        print(f"\nKaala: Immediate action taken for: {', '.join(goals) if goals else 'N/A'}")
        clarification = result.get("clarification")
        if clarification:
            print(f"Iccha: {clarification}")
        return

    if result_type == "goals_scheduled":
        goals = result.get("goals", [])
        goals_text = ", ".join(goals) if goals else "None"
        print(f"\nKaala: {result.get('message', 'Goals scheduled.')}")
        print(f"Goals: {goals_text}")
        clarification = result.get("clarification")
        if clarification:
            print(f"Iccha: {clarification}")
        immediate_results = result.get("immediate_results")
        if immediate_results:
            for ir in immediate_results:
                print(f"Immediate: {ir['goal']}")
        return

    if result_type == "executed":
        action = result.get("action") or "None"
        tool = result.get("tool") or "None"
        parameters = result.get("parameters", {})
        tool_result = result.get("tool_result")
        print(f"\nKarma: {result.get('result', '')}")
        print(f"Action: {action} | Tool: {tool}")
        if parameters:
            print(f"Parameters: {parameters}")
        if tool_result:
            print(f"Tool result: {tool_result}")
        return

    if result_type == "reminder":
        print(f"\n[Kaala Reminder] {result.get('message', '')}")
        return

    print(f"\nKaala: {result}")


def print_help() -> None:
    """Print available CLI commands."""
    print("\nCommands:")
    print("  /help                 Show this help")
    print("  /goals                Show pending goals")
    print("  /prompts              Show pending scheduled prompts")
    print("  /history [agent] [n]  Show recent history entries")
    print("  exit                  Quit")


async def run_command(orchestrator: Orchestrator, command: str) -> None:
    """Handle slash commands for inspecting orchestrator state."""
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd == "/help":
        print_help()
        return

    if cmd == "/goals":
        goals = await orchestrator.get_pending_goals()
        if not goals:
            print("\nNo pending goals.")
            return

        print("\nPending goals:")
        for goal in goals:
            details = f" - {goal['details']}" if goal.get("details") else ""
            print(f"  [{goal['id']}] {goal['goal']}{details}")
        return

    if cmd == "/prompts":
        prompts = await orchestrator.get_scheduled_prompts()
        if not prompts:
            print("\nNo pending scheduled prompts.")
            return

        print("\nScheduled prompts:")
        for prompt in prompts:
            print(f"  [{prompt['id']}] {prompt['scheduled_for']} -> {prompt['prompt']}")
        return

    if cmd == "/history":
        agent_name = parts[1] if len(parts) > 1 else None
        limit = 20
        if len(parts) > 2:
            try:
                limit = max(1, int(parts[2]))
            except ValueError:
                print("\nInvalid history limit. Use an integer.")
                return

        entries = await orchestrator.get_conversation_history(
            agent_name=agent_name, limit=limit
        )
        if not entries:
            print("\nNo history entries found.")
            return

        print("\nConversation history:")
        for entry in entries:
            print(
                f"  [{entry['timestamp']}] {entry['agent']} ({entry['role']}): "
                f"{entry['content']}"
            )
        return

    print("\nUnknown command. Use /help to see available commands.")


def cli_main() -> None:
    """Run the Kaala CLI loop with background prompt scheduler."""
    settings = get_settings()
    orchestrator = Orchestrator(model=settings.default_model)

    scheduler = PromptScheduler(
        orchestrator=orchestrator,
        poll_interval=settings.poll_interval,
    )

    print("Welcome to Kaala")
    print(f"Model: {settings.default_model}")
    print(f"Scheduler poll interval: {settings.poll_interval}s")
    print("Type /help for commands, or 'exit' to quit.")

    async def _async_main():
        await scheduler.start()
        try:
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "\nYou: ")
                    user_input = user_input.strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input.startswith("/"):
                    await run_command(orchestrator, user_input)
                    continue

                spin_task = asyncio.create_task(spinner())
                try:
                    result = await orchestrator.process_user_input(user_input)
                finally:
                    spin_task.cancel()
                    try:
                        await spin_task
                    except asyncio.CancelledError:
                        pass
                    print("\r", end="")

                print_result(result)
        finally:
            await scheduler.stop()

    asyncio.run(_async_main())


def web_main() -> None:
    """Run the Kaala web server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kaala.web.app:app",
        host=settings.host,
        port=settings.port,
    )


def main() -> None:
    """Dispatch to CLI or web based on sys.argv."""
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        web_main()
    else:
        cli_main()