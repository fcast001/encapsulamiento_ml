from flask import Flask, request, jsonify
import joblib
import numpy as np
from waitress import serve
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Function to load the saved model
def load_model():
    model = joblib.load('Model/pkl/rf_model.pkl')
    return model

# Initialize Flask app
app = Flask(__name__)

# Load the model globally for efficiency
model = load_model()

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()  # Get input data from POST request
    input_data = np.array(data['input']).reshape(1, -1)  # Reshape input
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Serve the app using waitress
    serve(app, host='0.0.0.0', port=8000)