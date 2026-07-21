"""Fonctions d'affichage : courbes d'entraînement et prédictions individuelles.

Séparées du modèle et des données pour respecter le principe de
responsabilité unique (SOLID: S) : le modèle ne devrait pas savoir comment
ses résultats sont affichés.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import NeuralNetwork


def plot_accuracy(history: dict, title: str = "Précision en fonction du nombre d'itérations") -> None:
    plt.plot(history["iteration"], history["accuracy"])
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel("Précision")
    plt.show()


def show_prediction(x: np.ndarray, y: np.ndarray, index: int, model: NeuralNetwork) -> None:
    """Affiche une image du dataset avec la prédiction du modèle et le vrai label."""
    image = x[:, index, None]
    prediction = model.predict_classes(model.forward(image))
    label = y[index]

    print(f"Prédiction : {prediction[0]} — Label réel : {label}")

    pixels = image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(pixels, interpolation="nearest")
    plt.show()
