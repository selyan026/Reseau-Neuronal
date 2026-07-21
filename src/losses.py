"""Fonction de coût.

Isoler la fonction de coût dans sa propre classe permet de la remplacer
(ex: par une MSE) sans toucher au modèle ni aux couches -> Dependency
Inversion (SOLID: D) : NeuralNetwork dépend d'une abstraction de "loss",
pas d'une implémentation figée en dur dans son propre code.
"""

import numpy as np


class CategoricalCrossEntropy:
    """Fonction de coût pour la classification multi-classes avec sortie Softmax."""

    @staticmethod
    def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Transforme un vecteur de labels (ex: [2, 7, 1]) en matrice one-hot."""
        one_hot_y = np.zeros((y.size, num_classes))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y.T

    def output_gradient(self, a_output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dZ pour la couche de sortie.

        Simplification analytique classique : quand l'activation de sortie
        est Softmax et le coût une cross-entropy, d(coût)/dZ = A - Y_onehot.
        """
        one_hot_y = self.one_hot(y, a_output.shape[0])
        return a_output - one_hot_y

    def compute(self, a_output: np.ndarray, y: np.ndarray) -> float:
        """Valeur scalaire du coût, utile pour suivre l'entraînement."""
        one_hot_y = self.one_hot(y, a_output.shape[0])
        m = y.size
        eps = 1e-9  # évite log(0)
        return float(-np.sum(one_hot_y * np.log(a_output + eps)) / m)
