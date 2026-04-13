from fastapi import FastAPI
from pydantic import BaseModel
import re
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Text Summarizer API")

# Enable CORS (for Vercel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/t5-small"
headers = {
    "Authorization": "Bearer YOUR_HF_API_KEY"   # 🔴 Replace this
}

# Input schema
class DialogueInput(BaseModel):
    dialogue: str

# Clean text
def clean_data(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# Summarization using API
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    payload = {
        "inputs": "summarize: " + dialogue,
        "parameters": {
            "max_length": 150,
            "min_length": 30
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    # Handle errors safely
    if isinstance(result, dict) and "error" in result:
        return "Error: " + result["error"]

    return result[0]["summary_text"]

# Routes
@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/summarize")
def summarize(data: DialogueInput):
    summary = summarize_dialogue(data.dialogue)
    return {"summary": summary}
