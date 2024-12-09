import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from google.colab import drive
import os
import datetime

curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(f"plots/{curr_datetime}")

#mount google drive
# drive.mount('/content/drive')

dir_path = 'data/stock-market-dataset/stocks'

@keras.saving.register_keras_serializable()
class Log1p(keras.layers.Layer):
    def call(self, inputs):
        return keras.ops.log1p(inputs)

class Sampling(keras.layers.Layer):
    def __init__(self, samples=1, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed=seed)
        self.samples = samples
    def call(self, inputs):
        z_mean, z_log_var = inputs[:, 0], inputs[:, 1]
        batch = keras.ops.shape(z_mean)[0]
        dim = keras.ops.shape(z_mean)[1:]
        samples = keras.random.normal(shape=(self.samples, batch, *dim), seed=self.seed_generator)
        epsilon = keras.ops.average(samples, axis=0)
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

encoder = keras.saving.load_model("encoder.keras")

#list of stock tickers
stock_tickers = np.load("tickers.npy")

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
    
    #create sequences for LSTM
    def create_sequences(data, time_steps):
        sequences, labels = [], []
        for i in range(len(data) - time_steps):
            mean, log_var = encoder(data[i:i+time_steps])
            sequences.append([mean, log_var])
            labels.append(data[i+time_steps, 0])
        return np.array(sequences), np.array(labels)

    X, y = create_sequences(np.asarray(data), time_steps)
    return X, y

#---train and evaluate model for each stock
for ticker in stock_tickers:
    print(f"\nAnalyzing stock: {ticker}")
    file_path = os.path.join(dir_path, f"{ticker}.csv")

    #check if the file exists
    if not os.path.exists(file_path):
        print(f"Data file for {ticker} not found. Skipping...")
        continue

    #preprocess data
    X, y = preprocess_data(file_path)

    #split into training, validation, testing
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    #build the LSTM model
    model = Sequential([
        keras.Input(shape=X_train.shape[1:]),
        Sampling(latent_dims=2, samples=5),
        LSTM(64, return_sequences=True),
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

    #results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Price")
    plt.plot(y_test_pred, label="Predicted Price")
    plt.title(f"Actual vs Predicted Prices for {ticker}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"plots/{curr_datetime}/{ticker}.png")
