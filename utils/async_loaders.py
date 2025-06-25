"""Asynchronous loaders for various tasks."""

import asyncio
import time


async def thinking_animation():
    """A simple asynchronous function to simulate thinking animation"""
    while True:
        for i in "Thinking...":
            print(i, end=" \b", flush=True)
            time.sleep(0.05)
        print("", end="\r")
        await asyncio.sleep(0.2)
        for _ in range(10):
            print("", end="\b", flush=True)
            time.sleep(0.01)
        await asyncio.sleep(0.2)
