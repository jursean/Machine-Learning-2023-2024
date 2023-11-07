import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    # YOUR CODE HERE
    img = np.reshape(img, (28,28), order="F")
    plt.matshow(img, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


# OPGAVE 1b
# Dit wordt gedaan om de waarden te schalen naar een lager bereik (0 tot 1). Neurale netwerken gebruiken vaak activatiefuncties die werken met waarden
# in het interval [0, 1] of [-1, 1].  Hierdoor verloopt het trainingsproces soepeler
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.
    # YOUR CODE HERE
    X = X.astype(float)

    # Berekenen van de maximale waarde per rij. (keepdims) zorgt ervoor dat de vorm gelijk blijft als X
    max_val = np.amax(X, axis=1, keepdims=True)

    # Voorzorgen dat de maximale waarde 0 vervangen wordt door 1 zodat de items niet gedeeld worden door 0
    max_val[max_val == 0] = 1


    return X / max_val




# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    # YOUR CODE HERE
    # Creëren van een sequantieel model (achteréén volgend).
    model = Sequential()

    # Input layer: Alleen 28x28 shape als input layer. Vervolgens omzetten naar een 1D > 784
    model.add(Flatten(input_shape=(28, 28)))

    # Hidden layer: Creëren van 128 neurons en toepassen van de activatiefunctie ReLU (wordt gebruikt in neurale netwerken)
    # ReLU vervangt alle negatieve weaarden door nul en laat positieve waarden ongewijzigd.
    model.add(Dense(128, activation=relu))

    # Output layer: Creëren van 10 neuronen/klassen (er zijn 10 soorten kleding stukken) en toepassen van softmax-activatiefunctie.
    # Die wordt gebruikt voor het berekekenen van de kansverdeling over de verschillende klassen
    model.add(Dense(10, activation=softmax))

    # Optimizer Adam wordt gebruikt, het is geschikt voor het aanpassen van de gewichten tijdens het trainen
    # SparseCategoricalCrossentropy bepaalt de loss-functie. Dit wordt gebruikt om de onnauwkeurigheid van het model te meten tijdens te training. Het meet dus de afwijking tussen de voorspelde waarde en de werkelijke waarde
    # Qua metrics wordt alleen de accuracy bij gehouden, dit geeft aan hoevaak het model correcte voorspelling doet.
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    
    # YOUR CODE HERE
    return tf.math.confusion_matrix(labels,pred)

    

# OPGAVE 2b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
 
    # YOUR CODE HERE
    tp = np.diagonal(conf)
    # Verticale som
    fp = np.sum(conf, axis=0) - tp
    # Horizontale som
    fn = np.sum(conf, axis=1) - tp
    tn = np.sum(conf) - (tp + fp + fn)
    result = list(zip(labels, tp, fp, fn, tn))

    return result

# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE 
    # ALS EEN DICTIONARY
    for a in metrics:
        tp += a[1]
        fp += a[2]
        fn += a[3]
        tn += a[4]

    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    rv = {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}

    return rv
