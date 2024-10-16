
from flask import Flask, request, jsonify
import joblib
import numpy as np
from waitress import serve


import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# function to load the saved model
def load_model():
    model = joblib.load('Model/pkl/rf_model.pkl')
    return model

# function to make predictions
def predict(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction

app = Flask(__name__)

# load the model
model = joblib.load('Model/pkl/rf_model.pkl')

# define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from POST request
    input_data = np.array(data['input']).reshape(1, -1)  # Reshape input
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)