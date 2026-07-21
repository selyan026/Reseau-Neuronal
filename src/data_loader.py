"""Chargement et préparation des données MNIST.

Design principle : Single Responsibility (SOLID: S).
Cette classe ne fait qu'une chose : lire le CSV, mélanger, normaliser et
séparer en jeu d'entraînement / jeu de test. Elle ne connaît rien du modèle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MnistDataLoader:
    def __init__(self, csv_path: str, dev_size: int = 1000, seed: int = 42):
        self.csv_path = csv_path
        self.dev_size = dev_size
        self.seed = seed

    def load(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Renvoie ((X_train, y_train), (X_dev, y_dev)).

        X a la forme (784, m) : une colonne par image.
        y a la forme (m,) : un label par image.
        """
        data = pd.read_csv(self.csv_path).to_numpy()

        rng = np.random.default_rng(self.seed)
        rng.shuffle(data)

        dev = data[: self.dev_size].T
        y_dev, x_dev = dev[0].astype(int), dev[1:] / 255.0

        train = data[self.dev_size :].T
        y_train, x_train = train[0].astype(int), train[1:] / 255.0

        return (x_train, y_train), (x_dev, y_dev)
