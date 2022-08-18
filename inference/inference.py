#%%
import sys
import logging
import os
from concurrent import futures

import grpc
import numpy as np
import pandas as pd
from gen.py.proto import inference_pb2_grpc, output_pb2
from joblib import load
from sklearn.feature_extraction import DictVectorizer
import pathlib

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

listen_on = os.environ.get("LISTEN_ON", "0.0.0.0:9527")
model_path = os.environ.get("MODEL_PATH")

#%%
class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self):
        self.model = load(model_path)

    def predict(self, request, context):
        X = np.array(request.data).reshape(request.shape)
        yhat = self.model.predict(X)
        return output_pb2.Output(shape=yhat.shape, result=yhat.ravel())

    def predict_proba(self, request, context):
        X = np.array(request.data).reshape(request.shape)
        yhat = self.model.predict_proba(X)
        return output_pb2.Output(shape=yhat.shape, result=yhat.ravel())


#%%
def serve():
    if not model_path or not pathlib.Path(model_path).exists():
        logger.error(f"model file not found.")
        return

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(listen_on)
    server.start()
    logger.info(f"Listen on: {listen_on}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt as e:
        logger.info("Bye")
        sys.exit(0)


#%%
if __name__ == "__main__":
    serve()
