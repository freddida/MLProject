from sklearn.preprocessing import LabelEncoder
from pandas.core.frame import DataFrame


def label_encode_column(df: DataFrame, column_name: str) -> None:
    """
    Encode the values in the specified column of the DataFrame using LabelEncoder.

    Parameters:
    - df (DataFrame): The pandas DataFrame.
    - column_name (str): The name of the column to be label-encoded.

    Returns:
    None
    """
    if column_name in df.columns:
        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
