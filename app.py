import flask
from flask import Flask, request
import os
import pickle
import numpy as np
import torch

app = Flask(__name__)
app.env = 'development'

print('flask app launched')

@app.route('/predict', methods=['POST'])
def predict():
    print('Request.method: ', request.method)
    print('Request.TYPE: ', type(request))
    print('making prediction ...')
    if request.method == 'POST':
        input = request.json['input']
        input = torch.tensor(np.random.random(10), dtype=torch.float32)
        model = torch.load('test_model.pt')
        prediction = model(input)
        print('prediction made, predicted result: ', prediction.detach().numpy().tolist())
        return {
            "output": prediction.detach().numpy().tolist()
        }
    return {
        "output": "wrong request type"
    }

app.run(host='0.0.0.0', port=5001, debug=False)