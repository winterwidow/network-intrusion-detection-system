"""loads the raw NSL-KDD text files into pandas DataFrames"""

from pathlib import Path

import pandas as pd

from .constants import NSL_KDD_COLUMNS, TEST_DATA_PATH, TRAIN_DATA_PATH


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load an NSL-KDD CSV-style text file with canonical column names."""
    return pd.read_csv(path, names=NSL_KDD_COLUMNS)


def load_data(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_dataset(train_path), load_dataset(test_path)
