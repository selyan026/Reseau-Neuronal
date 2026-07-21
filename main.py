"""Point d'entrée : entraîne le réseau et affiche les résultats.

Ce fichier "compose" l'application : c'est ici, et seulement ici, que l'on
choisit les implémentations concrètes (ReLU, Softmax, SGD + StepDecay...).
Le reste du code (model.py, layers.py...) ne connaît que des abstractions.
C'est le principe d'injection de dépendances en pratique.
"""

from src.activations import ReLU, Softmax
from src.data_loader import MnistDataLoader
from src.layers import DenseLayer
from src.model import NeuralNetwork
from src.optimizers import SGD, StepDecayOptimizer
from src.visualization import plot_accuracy, show_prediction

CSV_PATH = "data/train.csv"
ITERATIONS = 500
ALPHA_INITIAL = 1.0
ALPHA_DECAYED = 0.1
DECAY_AT_ITERATION = 360


def build_model() -> NeuralNetwork:
    layers = [
        DenseLayer(input_size=784, output_size=10, activation=ReLU(), seed=1),
        DenseLayer(input_size=10, output_size=10, activation=Softmax(), seed=2),
    ]
    optimizer = StepDecayOptimizer(
        base_optimizer=SGD(alpha=ALPHA_INITIAL),
        decay_at=DECAY_AT_ITERATION,
        decayed_alpha=ALPHA_DECAYED,
    )
    return NeuralNetwork(layers=layers, optimizer=optimizer)


def main() -> None:
    loader = MnistDataLoader(csv_path=CSV_PATH)
    (x_train, y_train), (x_dev, y_dev) = loader.load()

    model = build_model()
    history = model.train(x_train, y_train, iterations=ITERATIONS, log_every=10)

    plot_accuracy(history)

    test_accuracy = model.evaluate(x_dev, y_dev)
    print(f"\nPrécision finale sur les données de test : {test_accuracy * 100:.2f}%")

    show_prediction(x_train, y_train, index=0, model=model)


if __name__ == "__main__":
    main()
