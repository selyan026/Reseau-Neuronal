"""Couches du réseau de neurones.

Design principle : Single Responsibility (SOLID: S).
Une DenseLayer ne connaît que ses propres poids/biais et son propre calcul
avant/arrière. Elle ne sait rien de la fonction de coût ni des autres couches.
"""

from __future__ import annotations

import numpy as np

from .activations import Activation


class DenseLayer:
    """Une couche entièrement connectée (fully-connected / dense)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Activation,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        # Initialisation aléatoire centrée sur 0, comme dans la version originale.
        self.W = rng.random((output_size, input_size)) - 0.5
        self.b = rng.random((output_size, 1)) - 0.5
        self.activation = activation

        # Caches nécessaires à la rétropropagation, remplis lors du forward.
        self._input_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None
        self._output_cache: np.ndarray | None = None

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """Propage l'entrée de la couche précédente à travers cette couche."""
        self._input_cache = a_prev
        self._z_cache = self.W.dot(a_prev) + self.b
        self._output_cache = self.activation.forward(self._z_cache)
        return self._output_cache

    @property
    def output(self) -> np.ndarray:
        return self._output_cache

    def compute_gradients(self, dz: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        """À partir de l'erreur dZ de cette couche, calcule dW et db."""
        dw = (1 / m) * dz.dot(self._input_cache.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        return dw, db

    def propagate_error_backward(self, dz_next: np.ndarray, w_next: np.ndarray) -> np.ndarray:
        """Calcule dZ pour cette couche à partir de l'erreur de la couche suivante."""
        return w_next.T.dot(dz_next) * self.activation.derivative(self._z_cache)

    def update(self, dw: np.ndarray, db: np.ndarray, learning_rate: float) -> None:
        self.W -= learning_rate * dw
        self.b -= learning_rate * db
