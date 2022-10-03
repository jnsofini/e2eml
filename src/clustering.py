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
import numpy as np
import json
import os
import time
import params
from util import get_iv_from_binning_obj, get_best_feature_from_each_cluster, filter_iv_table

from sklearn.cluster import KMeans


def load_transformed_data(path):
    # Load transformed data and return only cols with non singular value
    transformed_data = pd.read_parquet(path)
    select_columns = transformed_data.columns[transformed_data.nunique() != 1]
    return transformed_data[select_columns]

def generate_cluster_metrics(extimator, cluster_class, X):
    r_square_ratio = []
    r_square_own = []
    r_square_nc = []
    for i, l in enumerate(extimator.labels_):
        centroid = extimator.cluster_centers_[l]
        # print(centroid.shape, X.values[0].shape)
        rsq_own = np.corrcoef(X.values[i], centroid)[0, 1]**2 
        rsq_nc = np.max(
            [np.corrcoef(X.values[i], extimator.cluster_centers_[j])[0, 1]**2 for j in set(extimator.labels_) if j!=l]
        )
        r_square_own.append(rsq_own)
        r_square_nc.append(rsq_nc)
        r_square_ratio.append((1-rsq_own)/(1-rsq_nc))

    rsq_table = cluster_class.assign(
    rsq_ratio=r_square_ratio,
    r_square_own=r_square_own,
    r_square_nc=r_square_nc
    ).sort_values(by=["cluster", "rsq_ratio"]).round(2)

    return rsq_table

# def get_cluster_tables(estimator, features):
#     cluster_class = pd.DataFrame(
#     {
#         "feature": features, 
#         "cluster": estimator.predict(X)
#         }
#     ).sort_values(by="cluster")

#     return cluster_class

def main():
    start_time = time.perf_counter()
    iv_table = get_iv_from_binning_obj(os.path.join(params.data_base_path, params.BINNING_TRANSFORM_PATH))
    transformed_data_all = load_transformed_data(os.path.join(params.data_base_path, params.TRANSFORM_DATA_PATH))

    modelling_features = filter_iv_table(iv_table, iv_cutoff=0.02, min_n_bins=2)


    transformed_data = transformed_data_all[modelling_features]

    print([col for col in transformed_data.columns if transformed_data[col].nunique() == 1])

    # K = range(1, 15)

    X = transformed_data.T
    kmeans = KMeans(n_clusters = 8, init='k-means++')
    kmeans.fit(X)

    cluster_class = pd.DataFrame(
    {
        "feature": modelling_features, 
        "cluster": kmeans.predict(X)
        }
    ).sort_values(by="cluster")
    rsq_table = generate_cluster_metrics(kmeans, cluster_class, X)

    rsq_iv_table = pd.merge(
        rsq_table,
        iv_table.rename(columns={"name":"feature"}),
        on="feature"
    )

    rsq_iv_table.to_csv(os.path.join(params.data_base_path, "r_square_iv_table.csv"))
    print(rsq_iv_table)
    # breakpoint()

    # Selected features by variable clustering
    selected_features_varclushi = get_best_feature_from_each_cluster(rsq_iv_table)
    with open(os.path.join(params.data_base_path, "selected-features-clustering.json"), mode='w') as f:
        json.dump({f"selected-features-varclushi": selected_features_varclushi}, f, indent=6)

    print(f"Time taken : {round(time.perf_counter() - start_time, 2)} seconds")


if __name__ == "__main__":
    main()
