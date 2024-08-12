from fastapi import FastAPI
from pydantic import BaseModel
from model import load_and_preprocess_data, preprocess_data, build_model, train_model, predict
import numpy as np

app = FastAPI()

# Load and preprocess data
features, target = load_and_preprocess_data()
X_train, X_test, y_train, y_test, normalizer = preprocess_data(features, target)

# Build and train the model
model = build_model(normalizer)
model, history = train_model(model, X_train, y_train)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bitcoin Price Prediction API"}

@app.get("/predict")
def predict_price():
    latest_data = fetch_data().iloc[-1][['RSI', 'SMA_50', 'EMA_50']].values.reshape(1, -1)
    latest_data_tensor = tf.convert_to_tensor(latest_data, dtype=tf.float32)
    prediction = model.predict(latest_data_tensor)
    predicted_price = prediction[0][0]
    confidence = prediction[0][1]
    
    action = "buy" if predicted_price > latest_data_tensor[0, 0] else "sell"
    return {
        "predicted_price": float(predicted_price),
        "confidence": float(confidence),
        "action": action
    }
