import streamlit as st
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model here
model = joblib.load('Model/pkl/rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()  # Get input data from POST request
    input_data = np.array(data['input']).reshape(1, -1)  # Reshape input
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

# Start Flask server in a separate thread
def run_app():
    app.run(port=5000)  # Use a different port to avoid conflicts

if __name__ == '__main__':
    import threading
    threading.Thread(target=run_app).start()
    st.title("Mi aplicación de predicción")
    # Aquí puedes agregar tu interfaz de Streamlit
