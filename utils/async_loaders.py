"""Asynchronous loaders and UI helpers."""

import asyncio


async def thinking_animation(text: str = "Thinking..."):
    """Simple async spinner/text animation for CLI usage."""
    while True:
        for char in text:
            print(char, end="\b", flush=True)
            await asyncio.sleep(0.03)
        print("", end="\r", flush=True)
        await asyncio.sleep(0.1)
