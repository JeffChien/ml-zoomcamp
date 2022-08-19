#%%
from concurrent import futures
import logging
import os
import pathlib
import signal
import sys
import threading

import grpc
from joblib import load
import numpy as np

from gen.py.proto import inference_pb2_grpc
from gen.py.proto import output_pb2

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

listen_on = os.environ.get("LISTEN_ON", "0.0.0.0:9527")
model_path = os.environ.get("MODEL_PATH")
model_framework = os.environ.get("MODEL_FRAMEWORK")

#%%
class SklearnInferenceServicer(inference_pb2_grpc.InferenceServicer):
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
class GracefulExit:
    def __init__(self, server: grpc.Server, timeout=10):
        self.server = server
        self.timeout = timeout
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        evt: threading.Event = self.server.stop(self.timeout)
        evt.wait(self.timeout)
        logger.info("Bye")


#%%
def serve():
    if not model_path or not pathlib.Path(model_path).exists():
        logger.error("model file not found.")
        return 1

    servicer = {"sklearn": SklearnInferenceServicer}.get(model_framework)
    if not servicer:
        logger.error(f"unknown model framework: {model_framework}")
        return 1

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServicer_to_server(servicer(), server)
    server.add_insecure_port(listen_on)
    server.start()
    logger.info(f"Listen on: {listen_on}")
    GracefulExit(server)
    server.wait_for_termination()
    return 0


#%%
if __name__ == "__main__":
    sys.exit(serve())
