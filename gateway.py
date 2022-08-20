from flask import Flask
from flask import jsonify
from flask import request

from ch2 import api as ch2_api

app = Flask(__name__)


@app.route("/ch2_car_price:predict", methods=["POST"])
def ch2_car_price_predict():
    X = ch2_api.preprocess(request.data)
    resp = ch2_api.predict(X)
    return jsonify(resp)
