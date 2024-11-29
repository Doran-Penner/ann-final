import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import pathlib
from keras import utils

data_root = utils.get_file(
    origin="https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset",
    cache_dir=".",
    cache_subdir="data",
    extract=True,
)

tickers = np.load("tickers.npy")

data_root_path = pathlib.Path(data_root)
stock_dir = data_root_path.joinpath("stocks")
data_strs = [str(x) for x in stock_dir.iterdir() if x.stem in tickers]
data_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
input_stacker = lambda x: tf.stack(
    list(x.values()), axis=1
)  # this is maybe inefficient but easy
target_adder = lambda x: (x, x)

vae_dataloader = (
    tf.data.experimental.make_csv_dataset(
        file_pattern=data_strs,
        batch_size=128,
        column_defaults=["float32" for _ in range(6)],
        num_epochs=1,
        select_columns=data_columns,
    )
    .ignore_errors()
    .map(input_stacker)
    .map(target_adder)
)

__all__ = ["vae_dataloader"]
