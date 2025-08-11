# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 23:03:57 2025

@author: dell
"""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite que React (u otro origen) acceda a la API

# Variable global que se actualizará desde CamaraIp.py
last_prediction = {"label": ""}

@app.route('/prediction')
def get_prediction():
    return jsonify(last_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


