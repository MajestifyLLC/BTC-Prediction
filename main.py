from fastapi import FastAPI
from pydantic import BaseModel
import threading
from model import fetch_live_data, get_or_train_model, predict, evaluate_model

app = FastAPI()

def train_model_in_background():
    # Start model training in a background thread
    predictor = get_or_train_model(days=365)
    app.state.predictor = predictor  # Store the trained model in app state

# Start the training thread
training_thread = threading.Thread(target=train_model_in_background)
training_thread.start()

@app.on_event("startup")
async def startup_event():
    app.state.predictor = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bitcoin Price Prediction API powered by AutoGluon with live data from CoinGecko"}

@app.get("/predict")
def predict_price():
    if app.state.predictor is None:
        return {"message": "Model is still training. Please try again later."}
    
    # Fetch and preprocess the latest live data
    data = fetch_live_data(days=1)  # Fetch data from the last day for prediction
    
    # Make predictions on the latest data
    predictions = predict(app.state.predictor, data)
    
    # Assuming confidence is derived from prediction consistency or other metrics
    confidence = "High"  # Placeholder for simplicity
    
    # Determine action based on comparison with last known price
    last_price = data['price'].iloc[-1]
    predicted_price = predictions.iloc[-1]
    action = "buy" if predicted_price > last_price else "sell"
    
    return {
        "predicted_price": float(predicted_price),
        "confidence": confidence,
        "action": action
    }

@app.get("/evaluate")
def evaluate():
    if app.state.predictor is None:
        return {"message": "Model is still training. Please try again later."}
    
    data = fetch_live_data(days=365)  # Fetch data from the last 365 days for evaluation
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)
    performance = evaluate_model(app.state.predictor, test_data)
    return performance



