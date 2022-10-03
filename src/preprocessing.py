import pandas as pd
from util import setup_binning
import params
import time
import os

TARGET: str = "churn"
SAVE_BINNING_OBJ = True


def load_data(path):
    df = pd.read_parquet(path)
    #  TODO: run some pandera data verification
    return df

def formating(df):

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    return df


def main():
    start_time = time.perf_counter()

    # Get raw data and split into X and y
    df_train = pd.read_parquet(os.path.join(params.data_base_path, "df_train.parquet"))

    columns_to_drop = [col for col in df_train.columns if df_train[col].nunique() == 1]
    print(f"Columns with no vatiance: {columns_to_drop}")
    X = df_train.drop(columns = columns_to_drop+[TARGET, "customerid", "Unnamed: 0"])
    y = df_train[TARGET].astype("int8").values
    
    binning_process = setup_binning(X)
    binning_process.fit(X, y)

    # Save Tranform data and binning_process
    X_transformed = binning_process.transform(X)
    X_transformed[TARGET] = y
    print(X_transformed.head())
    X_transformed.to_parquet(os.path.join(params.data_base_path, params.TRANSFORM_DATA_PATH))
    
    if SAVE_BINNING_OBJ:
        binning_process.save(os.path.join(params.data_base_path, params.BINNING_TRANSFORM_PATH) )

    print(f"Time taken : {round(time.perf_counter() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
    