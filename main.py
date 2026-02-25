import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from data import generate_mock_data
from compression import train_autoencoder
from predictor import train_predictor
from scheduler import find_best_posting_time

# Generate data
data = generate_mock_data()

X = data.drop(columns=["engagement"]).values
y = data["engagement"].values

# Train compression
autoencoder, scaler, compressed_features = train_autoencoder(X)

# Train engagement predictor
engagement_model = train_predictor(compressed_features, y)

# API
app = FastAPI()


class PostInput(BaseModel):
    likes: int
    comments: int
    shares: int
    hour: int
    content_type: int
    sentiment: float


@app.post("/optimize_post")
def optimize_post(post: PostInput):
    input_array = np.array([
        post.likes,
        post.comments,
        post.shares,
        post.hour,
        post.content_type,
        post.sentiment
    ])

    scaled = scaler.transform([input_array])
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        compressed = autoencoder.encoder(tensor).numpy()[0]

    best_hour, predicted_engagement = find_best_posting_time(
        compressed,
        engagement_model
    )

    return {
        "recommended_hour": int(best_hour),
        "predicted_engagement_score": float(predicted_engagement)
    }


@app.get("/")
def root():
    return {"status": "Autonomous Social Media Manager Running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)