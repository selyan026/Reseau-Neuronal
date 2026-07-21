import numpy as np

from src.activations import ReLU, Softmax


def test_relu_forward_clips_negative_values():
    relu = ReLU()
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu.forward(z), np.array([[0.0, 0.0, 3.0]]))


def test_relu_derivative_is_zero_for_negative_input():
    relu = ReLU()
    z = np.array([[-1.0, 2.0]])
    assert np.array_equal(relu.derivative(z), np.array([[0.0, 1.0]]))


def test_softmax_output_sums_to_one():
    softmax = Softmax()
    z = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]])
    output = softmax.forward(z)
    column_sums = output.sum(axis=0)
    assert np.allclose(column_sums, np.ones(2))
