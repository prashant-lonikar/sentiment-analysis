from fastapi import FastAPI
from pydantic import BaseModel
# import the pipeline command of the transformers library
from transformers import pipeline

# choose the model
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# create the FastAPI app
app = FastAPI()

# Define that input should be a string
class RequestModel(BaseModel):
    input: str

# Define that we accept post requests
@app.post("/sentiment")
def get_response(request: RequestModel):
    prompt = request.input
    response = pipe(prompt)
    label = response[0]['label']
    score = response[0]['score']
    return f"The input '{prompt}' is {label} with a score of {score}"