import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pydantic import BaseModel
from typing import List

class SpoilerRequest(BaseModel):
    sentences: List[str]
    titles: List[str] = []


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    This is used to perform startup and shutdown tasks.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ml_models['tokenizer'] = tokenizer

    model_name = "models/bert_180len_10epochs"
    ml_models['general'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['general'].eval()

    model_name = "models/bert_review_200len_0.7corpus_fine_tuned_fight_club"
    ml_models['fight_club'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['fight_club'].eval()

    model_name = "models/bert_review_200len_0.7corpus_fine_tuned_the_dark_knight"
    ml_models['the_dark_knight'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['the_dark_knight'].eval()

    model_name = "models/bert_review_200len_0.7corpus_fine_tuned_the_godfather"
    ml_models['the_godfather'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['the_godfather'].eval()

    model_name = "models/bert_review_200len_0.7corpus_fine_tuned_the_shawshank_redemption"
    ml_models['the_shawshank_redemption'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['the_shawshank_redemption'].eval()

    nodel_name = "models/bert_review_200len_0.7corpus_fine_tuned_the_lord_of_the_rings_the_return_of_the_king"
    ml_models['lotr'] = AutoModelForSequenceClassification.from_pretrained(model_name)
    ml_models['lotr'].eval()

    print("Starting Spoiler Prediction API...")

    yield
    print("Shutting down Spoiler Prediction API...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Spoiler Prediction API"}


@app.post("/spoilers/predict")
async def predict_spoiler(data: SpoilerRequest):
    """
    Predicts whether pieces of text contain spoilers or not.
    Args:
        data (dict): Dictionary with text data and, optionally, specific titles
    Returns:
        dict: Dictionary with predictions.
    """
    sentences = data.sentences
    titles = data.titles

    predictions = []
    tokenizer = ml_models['tokenizer']

    try:
        for sentence in sentences:
            inputs = tokenizer(sentence, max_length=200, padding="max_length", return_tensors="pt", truncation=True)
            outputs = ml_models['general'](**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

            for title in titles:
                if title == "Fight Club":
                    model = ml_models['fight_club']
                elif title == "The Dark Knight":
                    model = ml_models['the_dark_knight']
                elif title == "The Godfather":
                    model = ml_models['the_godfather']
                elif title == "The Shawshank Redemption":
                    model = ml_models['the_shawshank_redemption']
                elif title == "The Lord of the Rings":
                    model = ml_models['lotr']
                else:
                    continue

                outputs = model(**inputs)
                # predicted = torch.argmax(outputs.logits, dim=1)

                probs = torch.softmax(outputs.logits, dim=1)
                spoiler_prob = probs[0][1]
                predicted = torch.tensor(1 if spoiler_prob > 0.95 else 0) # Adjust threshold as needed

                prediction = prediction or predicted

            predictions.append(int(prediction.item()))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

    return {"predictions": predictions}
