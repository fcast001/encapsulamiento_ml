from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Inicializa Flask app
app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Carga el modelo
model = joblib.load('Model/pkl/rf_model.pkl')

# Define la ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()  # Obtener datos del cuerpo de la solicitud
    input_data = np.array(data['input']).reshape(1, -1)  # Reformatear los datos
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
