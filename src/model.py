"""Modèle : orchestre les couches, la fonction de coût et l'optimiseur.

Design principle : Dependency Inversion (SOLID: D).
NeuralNetwork ne dépend que d'abstractions (Activation via les couches,
Optimizer, la fonction de coût), jamais d'une implémentation concrète.
On peut donc changer d'optimiseur, de fonction de coût ou d'activation
sans modifier une seule ligne de cette classe.

Design principle : Open/Closed (SOLID: O).
Ajouter une couche, un optimiseur ou une activation ne demande jamais de
modifier NeuralNetwork, DenseLayer ou Activation existants : seulement d'en
ajouter de nouveaux qui respectent l'interface commune.
"""

from __future__ import annotations

import numpy as np

from .layers import DenseLayer
from .losses import CategoricalCrossEntropy
from .optimizers import Optimizer


class NeuralNetwork:
    def __init__(
        self,
        layers: list[DenseLayer],
        optimizer: Optimizer,
        loss: CategoricalCrossEntropy | None = None,
    ):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss or CategoricalCrossEntropy()
        self.history: dict[str, list] = {"iteration": [], "accuracy": [], "cost": []}

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _backward(self, y: np.ndarray, m: int) -> list[tuple[np.ndarray, np.ndarray]]:
        output_layer = self.layers[-1]
        dz = self.loss.output_gradient(output_layer.output, y)

        gradients: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore
        gradients[-1] = self.layers[-1].compute_gradients(dz, m)

        for i in reversed(range(len(self.layers) - 1)):
            next_layer = self.layers[i + 1]
            dz = self.layers[i].propagate_error_backward(dz, next_layer.W)
            gradients[i] = self.layers[i].compute_gradients(dz, m)

        return gradients

    def _update_params(self, gradients: list[tuple[np.ndarray, np.ndarray]], iteration: int) -> None:
        learning_rate = self.optimizer.get_learning_rate(iteration)
        for layer, (dw, db) in zip(self.layers, gradients):
            layer.update(dw, db, learning_rate)

    @staticmethod
    def predict_classes(a_output: np.ndarray) -> np.ndarray:
        return np.argmax(a_output, axis=0)

    @staticmethod
    def accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(predictions == y))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Précision du modèle sur un jeu de données (ex: données de test)."""
        predictions = self.predict_classes(self.forward(x))
        return self.accuracy(predictions, y)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iterations: int,
        log_every: int = 10,
        verbose: bool = True,
    ) -> dict[str, list]:
        m = x.shape[1]

        for i in range(iterations):
            a_output = self.forward(x)
            gradients = self._backward(y, m)
            self._update_params(gradients, i)

            if i % log_every == 0:
                predictions = self.predict_classes(a_output)
                acc = self.accuracy(predictions, y)
                cost = self.loss.compute(a_output, y)

                self.history["iteration"].append(i)
                self.history["accuracy"].append(acc)
                self.history["cost"].append(cost)

                if verbose:
                    print(f"Itération {i:4d} — coût : {cost:.4f} — précision : {acc * 100:.2f}%")

        return self.history
