import requests
import pandas as pd
import tensorflow as tf
from datetime import datetime

def fetch_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365"
    response = requests.get(url)
    data = response.json()
    
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    df['EMA_50'] = df['price'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['price'])
    
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_and_preprocess_data():
    df = fetch_data()
    features = df[['RSI', 'SMA_50', 'EMA_50']]
    target = df['price']
    return features, target

def build_model(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units=1),
        tf.keras.layers.Dense(units=1, activation='linear', name='confidence')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)
    return model, history

def predict(model, X):
    return model.predict(X)
