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
        think = asyncio.create_task(thinking_animation())
        response = await agent.generate_async(user_input)
        think.cancel()
        print(f"{agent.name()}: {response['response']}")


if __name__ == "__main__":
    print(agent.info())
    asyncio.run(main())
