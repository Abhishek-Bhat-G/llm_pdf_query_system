import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def call_llm(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text