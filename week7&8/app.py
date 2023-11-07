import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Load the data
test_data = pd.read_csv('sign_mnist_test.csv')
train_data = pd.read_csv('sign_mnist_train.csv')

m = len(test_data)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

index = np.random.randint(0, m)

labels = test_data['label'].values
images = test_data.iloc[:, 1:].values

def plot_images(label, image):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colormap = ['cividis', 'inferno', 'gray']
    image = np.reshape(image, (28, 28))

    for i in range(3):
        axes[i].matshow(image, cmap=colormap[i])
        axes[i].set_xlabel(alphabet[label])
        axes[i].text(0, -10, alphabet[label], ha='center')

    plt.show()

plot_images(labels[index], images[index])

# Prepare data
X = images / 255.0
y = labels
epoch = 3
batch_size = 32

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(activation, hidden_layer, learning_rate):
    model = keras.Sequential([
        keras.layers.Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation=activation),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_layer, activation=activation),
        keras.layers.Dense(26, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

activations = ['relu', 'tanh']
hidden_layers = [32,64,128]
learning_rates = [0.001, 0.01, 0.1]

param_grid = {
    'activation': activations,
    'hidden_layer': hidden_layers,
    'learning_rate': learning_rates
}

model = KerasClassifier(build_fn=create_model, verbose=1, activation='adam', hidden_layer=128, learning_rate=0.001)

X_train = X_train.reshape(-1, 28, 28, 1)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=6, cv=2, n_jobs=1, verbose=2)
random_search.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

best_hyper_grid = grid_search.best_params_
best_accuracy_grid = grid_search.best_score_

best_hyper_random = random_search.best_params_
best_accuracy_random = random_search.best_score_

print("Beste Hyperparameters GridSearch: ", best_hyper_grid)
print("Beste Test Accuracy GridSearch:", best_accuracy_grid)

print("Beste Hyperparameters RandomSearch: ", best_hyper_random)
print("Beste Test Accuracy RandomSearch:", best_accuracy_random)


