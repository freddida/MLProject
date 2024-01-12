import pandas as pd
from pandas.core.frame import DataFrame


def load_data(data_path: str = "../../data/training/train_data.csv") -> DataFrame:
    """
    Load a dataset and display basic information.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        DataFrame: The loaded dataset.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Display basic information about the dataset
    print("Raw data")
    data.info()

    return data
