# Download de dataset m.b.v. load_iris() uit sklearn.datasets.
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()

# Vul je featurematrix X op basis van de data.
X = dataset.data
feature_names = dataset.features_names
print(feature_names)
# De uitkomstvector y ga je vullen op basis van target. Standaard bevat deze array de waardes 0, 1 en 2
# (resp. 'setosa', 'versicolor', 'virginica'). Maak deze binair door 0 en 1 allebei 0 te maken (niet-virginica) en van
# elke 2 een 1 te maken (wel-virginica). Denk erom dat y het juiste datatype en de juiste shape krijgt.
y = dataset.target
temp = []

for i in y:
    if i == 0 or i == 1:
        temp.append(0)
    else:
        temp.append(1)

y = np.array(temp)
print(y.shape)

# Definieer een functie sigmoid() die de sigmoïde-functie implementeert.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))