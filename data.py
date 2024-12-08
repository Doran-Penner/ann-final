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

# load the ticker ids from our pre-made list
tickers = np.load("tickers.npy")
data_root_path = pathlib.Path(data_root)
stock_dir = data_root_path.joinpath("stocks")
all_data_strs = [str(x) for x in stock_dir.iterdir() if x.stem in tickers]

# randomly split into train/valid/test with 75/15/10 ratio
rng_seed = None
rng = np.random.default_rng(seed=rng_seed)
rng.shuffle(all_data_strs)  # obs: in-place modification!
train_valid_split = int(np.floor(len(all_data_strs) * 0.75))
valid_test_split = int(np.floor(len(all_data_strs) * 0.9))
train_strs = all_data_strs[:train_valid_split]
valid_strs = all_data_strs[train_valid_split:valid_test_split]
test_strs = all_data_strs[valid_test_split:]


# simple convenience to avoid repeat code
def make_vae_dataloader(data_strs, seed=None):
    data_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    # this is maybe inefficient but easy
    input_stacker = lambda x: tf.stack(list(x.values()), axis=1)  # noqa: E731
    target_adder = lambda x: (x, x)  # noqa: E731

    return (
        tf.data.experimental.make_csv_dataset(
            file_pattern=data_strs,
            batch_size=128,
            column_defaults=["float64" for _ in range(6)],
            num_epochs=1,
            shuffle=True,
            shuffle_seed=seed,
            select_columns=data_columns,
        )
        .ignore_errors()
        .map(input_stacker)
        .map(target_adder)
    )


vae_data = (
    make_vae_dataloader(train_strs, rng_seed),
    make_vae_dataloader(valid_strs, rng_seed),
    make_vae_dataloader(test_strs, rng_seed),
)


__all__ = ["vae_data"]
