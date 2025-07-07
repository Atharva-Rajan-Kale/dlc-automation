from autogluon.core.utils import show_versions

def test_ts():
    show_versions()
    import numpy as np
    import pandas as pd
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    df = pd.read_csv("https://autogluon.s3-us-west-2.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")

    data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    # Select the first 50 time series for faster training
    data = data.loc[data.item_ids[:50]]

    prediction_length = 5
    target_column = "target"

    # known covariates
    data["weekend"] = data.index.get_level_values("timestamp").weekday.isin([5, 6]).astype(float)
    data["log_target"] = np.log(data[target_column])
    # use item id as static features
    static_features = data.index.unique("item_id").to_frame()
    static_features.columns = ["extra_id_column"]
    data.static_features = static_features

    test_data = data.copy()
    train_data = data.slice_by_timestep(None, -prediction_length)

    predictor = TimeSeriesPredictor(
        path="autogluon-m4_hourly",
        target=target_column,
        prediction_length=prediction_length,
        eval_metric="sMAPE",
        known_covariates_names=["weekend"],
    )
    predictor.fit(
        train_data,
        hyperparameters={
            "Naive": {},
            "SeasonalNaive": {},
            "NPTS": {},
            "CrostonSBA": {},
            "ETS": {},
            "DirectTabular": {},
            "RecursiveTabular": {},
            "DeepAR": {"epochs": 2, "num_batches_per_epoch": 2},
            "TemporalFusionTransformer": {"epochs": 2, "num_batches_per_epoch": 2},
        },
    )

    predictor.leaderboard()
    predictor.leaderboard(test_data)

    known_covariates = test_data.drop(target_column, axis=1)
    print(predictor.predict(train_data, known_covariates=known_covariates).head(5))
