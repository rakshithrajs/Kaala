"""just the main file"""

import asyncio

from agent.agent_factory import AgentFactory
from utils.async_loaders import thinking_animation

agent = AgentFactory.create()

# write a sinple chatbot loop
async def main():
    """Main function to run the chatbot loop"""
    print("Welcome to the chatbot!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        thinking_task = asyncio.create_task(thinking_animation("Aaagu ra lucche..."))
        response = await agent.generate_async(user_input)
        thinking_task.cancel()
        print(f"Bot: {response}", end="")


if __name__ == "__main__":
    print(agent.info())
    asyncio.run(main())
