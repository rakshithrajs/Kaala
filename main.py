import ollama
from ollama import GenerateResponse

from dotenv import load_dotenv
import os

from utils.load_prompt import load_prompt

from datetime import datetime

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
SYSTEM_PROMPT = load_prompt("plan")

if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set.")

st = ""
while st != "exit":

    prompt = SYSTEM_PROMPT.replace("{{user_prompt}}", input("Enter prompt: ")).replace(
        "{{current_date}}", str(datetime.now())
    )
    if prompt == "exit":
        break
    res: GenerateResponse = ollama.generate(model=MODEL_NAME, prompt=prompt)
    print(res["response"])
