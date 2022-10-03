import pandas as pd
from sklearn.model_selection import train_test_split
import params
import os

TARGET = "churn"


def get_data_splits(df, test_size=0.2, stratify=None):
    """Generate balanced data splits."""
    df_train, df_test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=stratify
        )

    return df_train, df_test


def save_parquet(path, frames: dict):
    for name, frame in frames.items():
        frame.to_parquet(os.path.join(path, f"{name}.parquet"))

def main():
    data = pd.read_csv(params.data_local_path)
    df_train, df_test = get_data_splits(data, stratify=data[TARGET])

    # Save the data
    save_parquet(
        path = "data",
        frames = {
            "df_train":df_train, 
            "df_test": df_test
                }
                )


if __name__ ==  "__main__":
    main()

