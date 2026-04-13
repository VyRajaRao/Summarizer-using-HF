from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/t5-small"
headers = {"Authorization": "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"}

class DialogueInput(BaseModel):
    dialogue: str

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/summarize")
def summarize(data: DialogueInput):
    payload = {
        "inputs": "summarize: " + data.dialogue,
        "parameters": {"max_length": 150}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    return {"summary": result[0]["summary_text"]}
