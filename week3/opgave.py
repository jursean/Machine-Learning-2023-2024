# Download de dataset m.b.v. load_iris() uit sklearn.datasets.
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

dataset = load_iris()

p = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
target = dataset.target
target = np.where(target < 2, 0, 1)
df = pd.concat([p, pd.DataFrame({'target': target})], axis=1)
print(df.head())

# Vul je featurematrix X op basis van de data
X = dataset.data

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

# Definieer een functie sigmoid() die de sigmoïde-functie implementeert.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Aantal eigenschappen en initialiseren van theta
# Initialiseer een vector theta met 1.0'en in de juiste shape
n = X.shape[1]
theta = np.ones(n)

# Nu kun je beginnen aan de loop waarin je in 1500 iteraties
# De voorspellingen (denk aan sigmoid!) en de errors berekent.
# De gradient berekent en theta aanpast. Werk in eerste instantie met een learning rate van 0.01.
# De kosten bereken
alpha = 0.01
for i in range(1500):
    voorspelling = sigmoid(np.dot(X, theta))

    kosten = np.mean(-y * np.log(voorspelling) - (1 - y) * np.log(1 - voorspelling))
    theta -= alpha * (np.dot(X.T, (voorspelling - y)) / len(y))

    print(f"{i}: kosten: {kosten}")
