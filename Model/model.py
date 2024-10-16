from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

# load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
import pickle

# Supongamos que tienes tus datos en X_train e y_train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el nuevo modelo
with open('Model/pkl/rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)