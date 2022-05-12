import pandas as pd
import actableai.timeseries.params
import actableai.timeseries.models
from gluonts.evaluation import Evaluator
import ray
import mxnet as mx

mx.random.seed(1)

import torch

ray.init()
df = pd.read_csv("data/retail_sales.csv")
df.index = pd.to_datetime(df["date"], format="%Y/%m/%d")
df.sort_index(inplace=True)

prediction_length = 10
m = actableai.timeseries.models.AAITimeseriesForecaster(
    prediction_length,
    mx.cpu(),
    torch.device("cpu"),
    #     actableai.timeseries.params.DeepARParams(epochs=1),
    feed_forward_params=actableai.timeseries.params.FeedForwardParams(
        learning_rate=1e-6, context_length=20, epochs=2
    ),
    #     actableai.timeseries.params.ProphetParams(),10
)

m.fit(
    df[
        [
            "numer_retail_sales",
        ]
    ]
    / 1000,
    trials=1,
)
