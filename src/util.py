import pandas as pd
from optbinning import Scorecard, BinningProcess
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

SPECIAL_CODES = [-9]
MISSING = [-99_000_000]

def setup_binning(
    df, 
    *, 
    target=None,
    features=None, 
    binning_fit_params=None
):
    """
    Setup the binning process for optbinning.

    Args:
        binning_fit_params: fit parameters object, including splits
        features: the list of features that we are interested in
        target: the target variable
        df (DataFrame): Dataframe containing features and a target column called 'target'

    Returns: Optbinning functional to bin the data BinningProcess()

    """
    # TODO: Use features as None to assume all features are used to fit model
    # Remove target if present in data
    if target:
        df = df[list(set(df.columns.values) - {target})]

    # Subset only the features provided by user
    if features:
        binning_features = features
    else:
        binning_features = df.columns.values
    
    categorical_variables = df[binning_features].select_dtypes(
            include=["object", "category", "string"]
        ).columns.values


    return BinningProcess(
        categorical_variables=categorical_variables,
        variable_names=binning_features,
        # Uncomment the below line and pass a binning fit parameter to stop doing automatic binning 
        # binning_fit_params=binning_fit_params,
        # This is the prebin size that should make the feature set usable 
        min_prebin_size=10e-5,
        special_codes=SPECIAL_CODES
    )


def get_best_feature_from_each_cluster(clusters):
    # The best feature from each cluster is the one with the min RS Ratio from that cluster
    # If the feature with the highest IV is different than the one with the highest RS Ratio, it is included as well. 
    highest_iv = clusters.loc[clusters.groupby(["cluster"])["iv"].idxmax()][
        "feature"
    ].tolist()
    lowest_rs_ratio = clusters.loc[clusters.groupby(["cluster"])["rsq_ratio"].idxmin()][
        "feature"
    ].tolist()

    return list(set(highest_iv + lowest_rs_ratio))


def get_iv_from_binning_obj(path):
    iv_table = BinningProcess.load(path).summary()
    iv_table["iv"] = iv_table["iv"].astype("float").round(3)
    return iv_table[["iv", "name", "n_bins"]]

def filter_iv_table(iv_table, iv_cutoff=0.02, min_n_bins=2):
    # Filter based on IV and min_number of bins
    return iv_table.query(f"n_bins >= {min_n_bins} and iv >= {iv_cutoff}").name.values

def calculate_vif(data, features, target):
    data = data[list(features) + [target]]
    _, X = dmatrices(f"{target} ~" + "+".join(features), data, return_type="dataframe")
    X = data[features].values

    vif_info = pd.DataFrame()
    vif_info["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_info["feature"] = features
    vif_info.sort_values("VIF", ascending=False)
    return vif_info


def scorecard(process, *, method=None):
    """
    Model estimator to be used for fitting.

    Args:
        process: Optbinning binning process operator
        method: Scikit learn estimator for fitting

    Returns:
        A Scorecard object
    """
    if method is None:
        method = LogisticRegression()

    scaling_method: str = "min_max"
    scaling_method_data = {
        "min": 350,
        "max": 850,
    }
    return Scorecard(
        binning_process=process,
        estimator=method,
        scaling_method=scaling_method,
        scaling_method_params=scaling_method_data,
        intercept_based=False,
        reverse_scorecard=False,
    )
