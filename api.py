from flask import Flask, request, jsonify
import joblib
import numpy as np
from threading import Thread
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from flask_cors import CORS


# Ignorar advertencias de versiones inconsistentes
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Función para cargar el modelo guardado
def load_model():
    model = joblib.load('Model/pkl/rf_model.pkl')
    return model

# Inicializar la aplicación Flask
app = Flask(__name__)

CORS(app)

# Cargar el modelo globalmente para eficiencia
model = load_model()

# Definir una ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()  # Obtener datos de entrada de la solicitud POST
    if 'input' not in data:
        return jsonify({'error': 'No input data provided'}), 400  # Manejo de errores

    input_data = np.array(data['input']).reshape(1, -1)  # Dar forma a la entrada
    prediction = model.predict(input_data)

    return jsonify({'prediction': int(prediction[0])})  # Devolver la predicción

def run_flask():
    app.run(host='0.0.0.0', port=8000)

# Correr Flask en un hilo separado
Thread(target=run_flask).start()

# Código de Streamlit
import streamlit as st
import requests

st.title("Predicción con Modelo de Machine Learning")

# Inputs para la predicción
input_data = st.text_input("Ingresa tus datos de entrada (separados por comas)")

if st.button("Predecir"):
    try:
        # Convertir los datos de entrada a una lista de flotantes
        input_array = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        
        # Realizar la solicitud a la API Flask
        response = requests.post("http://localhost:8000/predict", json={"input": input_array.tolist()})
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"La predicción es: {prediction}")
        else:
            st.error("Error al obtener la predicción.")
    except ValueError:
        st.error("Por favor ingresa datos válidos.")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")