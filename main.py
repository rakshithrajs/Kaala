"""CLI entrypoint for Kaal proactive assistant."""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from agent.orchestrator import ProactiveOrchestrator


async def scheduler_loop(orchestrator: ProactiveOrchestrator, interval_s: int = 15):
    while True:
        due_results = await orchestrator.run_due_tasks_once()
        for item in due_results:
            print(f"\n[Proactive Action][task={item['task_id']}] {item['result']}")
        await asyncio.sleep(interval_s)


async def chat_loop(model: str, interval_s: int):
    orchestrator = ProactiveOrchestrator(model=model)
    print("Welcome to Kaal (proactive mode). Type 'exit' to quit.")
    print("Scheduler is running in background for due prompts.")

    scheduler_task = asyncio.create_task(scheduler_loop(orchestrator, interval_s))
    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                break

            result = await orchestrator.process_user_message(user_input)
            print(f"Kaal: {result.reply}")
            print(f"Pending scheduled prompts: {orchestrator.store.pending_count()}")
    finally:
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaal proactive assistant")
    parser.add_argument("--model", default="GEMINI-1.5-PRO", help="model alias from config/models.json")
    parser.add_argument("--poll-interval", type=int, default=15, help="scheduler polling interval (seconds)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(chat_loop(args.model, args.poll_interval))
