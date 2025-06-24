"""just the main file"""

from agent.agent_factory import AgentFactory

agent = AgentFactory.create()

# write a sinple chatbot loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = agent.chat_stream(user_input)
    print(f"{agent.name()}: ", end="", flush=True)
    for chunk in response:
        print(f"{str(chunk).strip()}", end=" ", flush=True)
    print()
