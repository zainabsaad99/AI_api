from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API for sentiment analysis using Hugging Face Transformers",
    version="1.0.0"
)

# Model configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_PATH = "saved_model"

def load_or_save_model():
    if os.path.exists(MODEL_PATH):
        # Load the saved model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        # Download and save the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
    
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = load_or_save_model()

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    label: str  # Adding label to show the raw model output

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(text_input: TextInput):
    try:
        # Get sentiment analysis result
        result = sentiment_analyzer(text_input.text)[0]
        
        # Map the model's output to our response
        sentiment_mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }
        
        return SentimentResponse(
            sentiment=sentiment_mapping[result['label']],
            confidence=result['score'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 