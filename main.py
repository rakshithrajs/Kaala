"""just the main file"""

import asyncio

from agent.agent_factory import AgentFactory

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
        response = agent.chat(user_input, tools=True)
        print(f"{agent.name()}: {response['message']['content']}")


if __name__ == "__main__":
    print(agent.info())
    asyncio.run(main())
