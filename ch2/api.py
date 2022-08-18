#%%
import os
import io

import grpc
import numpy as np
import pandas as pd
from gen.py.proto import inference_pb2_grpc, input_pb2
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

from ch2.ch2_util import BasicPreprocessing, RecordizeDataframe

server_host = os.environ.get("CH2_HOST", "localhost:9527")


def preprocess(csv_data):
    preprocess = load("ch2/model/proprocess.joblib")
    df = pd.read_csv(io.BytesIO(csv_data))
    df = df.loc[:, df.columns != "MSRP"]
    return preprocess.transform(df)


def predict(X):
    channel = grpc.insecure_channel(server_host)
    stub = inference_pb2_grpc.InferenceStub(channel)
    req = input_pb2.Input(shape=X.shape, data=np.asarray(X.todense()).ravel())
    resp = stub.predict(req)
    prices = np.expm1(np.array(resp.result).reshape(resp.shape))
    res = []
    for i, price in enumerate(prices):
        res.append(dict(seq=i, price=price))
    return res
