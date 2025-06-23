import ollama
from ollama import GenerateResponse

from utils.load_prompt import load_prompt
from utils.model_parser import model_select

from datetime import datetime

MODEL_NAME = model_select("LLAMA")
SYSTEM_PROMPT = load_prompt("plan")

st = ""
while st != "exit":

    prompt = SYSTEM_PROMPT.replace("{{Iccha_goals}}", input("Enter prompt: ")).replace(
        "{{current_date}}", str(datetime.now())
    )
    if prompt == "exit":
        break
    res: GenerateResponse = ollama.generate(model=MODEL_NAME, prompt=prompt)
    print(res["response"])
