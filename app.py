from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Text Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (relative path)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    input_text = "summarize: " + dialogue

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/summarize")
def summarize(data: DialogueInput):
    summary = summarize_dialogue(data.dialogue)
    return {"summary": summary}