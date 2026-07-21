"""Fonctions d'activation du réseau.

Design pattern : Strategy.
Chaque activation implémente la même interface (forward / derivative), ce qui
permet d'en ajouter de nouvelles (Sigmoid, Tanh, LeakyReLU...) sans jamais
modifier le code des couches ou du modèle -> principe Ouvert/Fermé (SOLID: O).
"""

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Interface commune à toutes les fonctions d'activation."""

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Calcule la sortie de l'activation à partir de la somme pondérée z."""

    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Calcule la dérivée de l'activation, utilisée en rétropropagation."""


class ReLU(Activation):
    """Rectified Linear Unit : max(0, z)."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)


class Softmax(Activation):
    """Transforme un vecteur de scores en distribution de probabilités.

    Utilisée uniquement en couche de sortie pour la classification
    multi-classes.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        # On soustrait le max par colonne pour la stabilité numérique
        # (évite les débordements de np.exp sur de grandes valeurs).
        shifted = z - np.max(z, axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=0, keepdims=True)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # Non utilisée telle quelle : lorsque Softmax est combinée à une
        # cross-entropy, la dérivée se simplifie analytiquement en
        # (A - Y). C'est cette simplification qu'utilise CategoricalCrossEntropy
        # dans losses.py, donc cette méthode ne devrait jamais être appelée.
        raise NotImplementedError(
            "La dérivée de Softmax est gérée conjointement avec la fonction "
            "de coût dans CategoricalCrossEntropy.output_gradient()."
        )
