"""
Script to ingest data from Source.
---------------------------------
Here our data is coming from the ml zoom camp page
Works in env: general38
"""
import pandas as pd
import params
import os

OVERWRITE = True

# def ingest_data_from_url(url):
#     data = pd.read_csv(params.data_source)
#     data.to_csv(params.data_local_path)

def formating(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.churn = (df.churn == 'yes').astype(int)
    return df 

def main():
    raw_data = pd.read_csv(params.data_source)
    cleaned_data = formating(raw_data)
    cleaned_data.to_csv(params.data_local_path)

if __name__ == "__main__":

    if OVERWRITE:
        print(f"Ingesting data from {params.data_source} with overwrite")
        main()
    else:
        if not os.path.exists(params.data_local_path):
            print(f"Ingesting data from {params.data_source} with no overwrite")
            main()
    
