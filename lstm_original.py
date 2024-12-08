import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from google.colab import drive
import os

#mount google drive
drive.mount('/content/drive')

dir_path = '/content/drive/MyDrive/data/stock-market-dataset/stocks'

stocks_to_analyze_file = '/content/drive/MyDrive/data/Stocktest.csv'

#list of stock tickers
tickers_df = pd.read_csv(stocks_to_analyze_file)
#extract from first column
stock_tickers = tickers_df.iloc[:, 0].tolist() 

print(f"Stock tickers to analyze: {stock_tickers}")

#---preprocess data for each stock
def preprocess_data(file_path, time_steps=50):
    data = pd.read_csv(file_path)

    #ensure the dataset contains required columns
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset {file_path} is missing required columns: {required_columns}")
    
    #convert date to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    #extract close prices
    close_prices = data['Close'].values

    #normalize the data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices.reshape(-1, 1))

    #create sequences for LSTM
    def create_sequences(data, time_steps):
        sequences, labels = [], []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i+time_steps])
            labels.append(data[i+time_steps])
        return np.array(sequences), np.array(labels)

    X, y = create_sequences(scaled_prices, time_steps)
    return X, y, scaler

#---train and evaluate model for each stock
for ticker in stock_tickers:
    print(f"\nAnalyzing stock: {ticker}")
    file_path = os.path.join(dir_path, f"{ticker}.csv")

    #check if the file exists
    if not os.path.exists(file_path):
        print(f"Data file for {ticker} not found. Skipping...")
        continue

    #preprocess data
    X, y, scaler = preprocess_data(file_path)

    #split into training, validation, testing
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    #build the LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    model.summary()

    #train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, verbose=1)

    #evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"{ticker} - Test Loss: {test_loss}, Test MAE: {test_mae}")

    #make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #reverse scaling to get actual prices
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    #results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label="Actual Price")
    plt.plot(y_test_pred, label="Predicted Price")
    plt.title(f"Actual vs Predicted Prices for {ticker}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
