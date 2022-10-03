"""
Feature selection step 1:
Here we look at correlated features and treat them as clusters. We can then select one of two features
per clusters. This step removes multi-collinearity

Approach:
=========
We useds k means clustering to select a certain number of features. The features are converted to rows 
so the the data becomes num_features*num_instances
"""

import pandas as pd
from varclushi import VarClusHi
import json
import os
import time
import personal_config as config
import features as params
from preprocess import load_data
from util import get_iv_from_binning_obj, get_best_feature_from_each_cluster, filter_iv_table


MAX_EIGEN_SPLIT = 0.5
SEGMENTATION = None #"alnc-vs-non-alnc"

def load_transformed_data(path):
    # Load transformed data and return only cols with non singular value
    transformed_data = pd.read_parquet(path)
    select_columns = transformed_data.columns[transformed_data.nunique() != 1]
    return transformed_data[select_columns]

def main():
    start_time = time.perf_counter()
    # transformed_data = load_transformed_data(config.TRANSFORM_DATA_PATH)
    # use ivs to ensure that we don't pass useless features to the clustering

    # Define segment to run:
    segment = "ALNC"
    # Define storage for data
    os.makedirs(path:=config.BASE_PATH, exist_ok=True)
    if segment:
        os.makedirs(path:=os.path.join(config.BASE_PATH, segment), exist_ok=True)

    print(f"Working dir is:  {path}")

    iv_table = get_iv_from_binning_obj(os.path.join(path, config.BINNING_TRANSFORM_PATH))
    transformed_data_all = load_transformed_data(os.path.join(path, config.TRANSFORM_DATA_PATH))

    # Filter by segment
    # alnc_mask = transformed_data_all['B1_SEG_BUS_DIV_CD'] == 'ALNC'
    # non_alnc_mask = transformed_data_all['B1_SEG_BUS_DIV_CD'] == "NON_ALNC"
    # transformed_data_all = transformed_data_all[non_alnc_mask]

    # Non-rsps variables and other features to drop
    # filter_variables = list(
    #     set(
    #         params.rsps_features + 
    #         params.code_features
    #         ).intersection(set(transformed_data_all.columns)))
    # if filter_variables:
    #     # non_rsps_variables = list(set(params.features).intersection(set(transformed_data_all.columns.values)))
    #     print("=============Dropping features==========")
    #     print(*filter_variables)
    #     print("========================================")
    #     transformed_data_all.drop(columns=filter_variables, inplace=True)
    #     iv_table = iv_table[~iv_table.name.isin(filter_variables)]

    modelling_features = list(
            set(filter_iv_table(iv_table, iv_cutoff=0.02, min_n_bins=2)).intersection(set(params.alnc_model_features))
        )
    # modelling_features = list(
    #         set(filter_iv_table(iv_table, iv_cutoff=0.02, min_n_bins=2)).intersection(params.alnc_model_features)
    #     )


    transformed_data = transformed_data_all[modelling_features]

    print([col for col in transformed_data.columns if transformed_data[col].nunique() == 1])


    # model_data = raw_data[transformed_data.columns]

    # Using the transform data to get features and clusters
    # TODO: Do we want to use a cutoff of 1.0 or 0.7 for max eigenvalue?
    clusters = VarClusHi(
        transformed_data, maxeigval2=MAX_EIGEN_SPLIT, maxclus=None
    )
    clusters.varclus()

    # Select best feature from each cluster
    r_square_iv_table = pd.merge(
        clusters.rsquare, iv_table[iv_table.name.isin(modelling_features)], how="left", left_on="Variable", right_on="name"
    ).round(3)

    r_square_iv_table.to_csv(os.path.join(path, "r_square_iv_table.csv"))
    # breakpoint()

    # Selected features by variable clustering
    selected_features_varclushi = get_best_feature_from_each_cluster(r_square_iv_table)
    with open(os.path.join(path, "selected-features-varclushi.json"), mode='w') as f:
        json.dump({f"selected-features-varclushi": selected_features_varclushi}, f, indent=6)

    print(f"Time taken : {round(time.perf_counter() - start_time, 2)} seconds")


if __name__ == "__main__":
    main()
