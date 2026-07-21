import numpy as np

from src.losses import CategoricalCrossEntropy


def test_one_hot_encoding():
    loss = CategoricalCrossEntropy()
    y = np.array([2, 0])
    one_hot = loss.one_hot(y, num_classes=3)

    expected = np.array(
        [
            [0, 1],  # classe 0
            [0, 0],  # classe 1
            [1, 0],  # classe 2
        ]
    )
    assert np.array_equal(one_hot, expected)


def test_output_gradient_is_zero_for_perfect_prediction():
    loss = CategoricalCrossEntropy()
    y = np.array([1])
    perfect_prediction = np.array([[0.0], [1.0], [0.0]])

    gradient = loss.output_gradient(perfect_prediction, y)

    assert np.allclose(gradient, np.zeros_like(gradient))
