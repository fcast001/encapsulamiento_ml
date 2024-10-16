from flask import Flask, request, jsonify
import joblib
import numpy as np
from waitress import serve
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignorar advertencias de versiones inconsistentes
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Función para cargar el modelo guardado
def load_model():
    model = joblib.load('Model/pkl/rf_model.pkl')
    return model

# Inicializar la aplicación Flask
app = Flask(__name__)

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

if __name__ == '__main__':
    # Servir la aplicación usando waitress
    serve(app, host='0.0.0.0', port=8000)
