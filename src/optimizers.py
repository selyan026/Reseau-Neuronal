"""Optimiseurs (stratégies de mise à jour du taux d'apprentissage).

Design patterns :
- Strategy : Optimizer définit une interface commune (get_learning_rate),
  interchangeable dans NeuralNetwork sans changer son code.
- Decorator : StepDecayOptimizer enrichit un optimiseur existant (ici SGD)
  avec un comportement supplémentaire (baisse du taux après un certain
  nombre d'itérations) sans modifier ni dupliquer la classe SGD.
  C'est la traduction directe de la stratégie "alpha=1 puis alpha=0.1"
  qui a donné les meilleurs résultats dans les expérimentations du TIPE.
"""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def get_learning_rate(self, iteration: int) -> float:
        """Renvoie le taux d'apprentissage à utiliser pour cette itération."""


class SGD(Optimizer):
    """Descente de gradient à taux d'apprentissage constant."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def get_learning_rate(self, iteration: int) -> float:
        return self.alpha


class StepDecayOptimizer(Optimizer):
    """Décore un optimiseur existant pour réduire le taux d'apprentissage
    une fois un certain nombre d'itérations atteint.
    """

    def __init__(self, base_optimizer: Optimizer, decay_at: int, decayed_alpha: float):
        self._base_optimizer = base_optimizer
        self._decay_at = decay_at
        self._decayed_alpha = decayed_alpha

    def get_learning_rate(self, iteration: int) -> float:
        if iteration >= self._decay_at:
            return self._decayed_alpha
        return self._base_optimizer.get_learning_rate(iteration)
