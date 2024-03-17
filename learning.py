from math import tanh
from random import random
from typing import Callable, Sequence, Literal, Union


def activation_function(x: float) -> float:
    return tanh(x)


def validate_learning_input(inputs: Sequence[Sequence[float]], desired_outputs: Sequence[float],
                            tolerance_threshold: float, starting_weights_type: Literal['random', 'external'],
                            starting_weights: Union[Sequence[float], None]):
    input_lens = [len(input) for input in inputs]

    if any([input_len != input_lens[0] for input_len in input_lens]):
        raise Exception('Inputs have different lengths')

    if len(inputs) != len(desired_outputs):
        raise Exception(
            'The number of input samples and desired outputs should be equal')

    if tolerance_threshold < 0:
        raise Exception('Tolerance threshold should be non-negative number')

    if starting_weights_type == 'external' and starting_weights is None:
        raise Exception('No starting weights were provided')

    if starting_weights is not None and len(starting_weights) != len(inputs[0]) + 1:
        raise Exception('Provided weights have inappropriate length')


def generate_random_weights(N: int) -> Sequence[float]:
    return [random() for _ in range(N + 1)]


def get_weighted_sum(x: Sequence[float], weights: Sequence[float]) -> float:
    return weights[0] + sum([x[i] * weights[i + 1] for i in range(len(x))])


def get_neuron_output(input: Sequence[float], weights: Sequence[float],
                      activation_function: Callable[[float], float] = activation_function) -> float:
    return activation_function(get_weighted_sum(input, weights))


def get_neuron_error(inputs: Sequence[Sequence[float]], desired_outputs: Sequence[float],
                     weights: Sequence[float],
                     activation_function: Callable[[float], float] = activation_function) -> Sequence[float]:
    N = len(inputs)

    return [desired_outputs[j] - get_neuron_output(inputs[j], weights, activation_function) for j in range(N)]


def get_MSE(inputs: Sequence[Sequence[float]], desired_outputs: Sequence[float], weights: Sequence[float],
            activation_function: Callable[[float],
                                          float] = activation_function,
            neuron_error: Union[Sequence[float], None] = None) -> float:
    N = len(inputs)

    deltas = neuron_error if neuron_error is not None else get_neuron_error(
        inputs, desired_outputs, weights, activation_function)

    return sum(delta ** 2 for delta in deltas) / N


def get_RMSE(inputs: Sequence[Sequence[float]], desired_outputs: Sequence[float], weights: Sequence[float],
             activation_function: Callable[[
                 float], float] = activation_function,
             neuron_error: Union[Sequence[float], None] = None, MSE: Union[float, None] = None) -> float:
    if MSE is not None:
        return MSE ** 0.5
    else:
        return get_MSE(inputs, desired_outputs, weights, activation_function=activation_function,
                       neuron_error=neuron_error)


def error_correction_learning_with_RMSE_stopping(inputs: Sequence[Sequence[float]],
                                                 desired_outputs: Sequence[float], tolerance_threshold: float,
                                                 starting_weights_type: Literal['random',
                                                                                'external'] = 'random',
                                                 starting_weights: Union[Sequence[float], None] = None):
    validate_learning_input(inputs, desired_outputs, tolerance_threshold,
                            starting_weights_type, starting_weights)

    alpha = 1 / (len(inputs[0]) + 1)

    N = len(inputs)

    iterations = 1

    RMSEs = []

    weights = starting_weights if starting_weights is not None else generate_random_weights(
        N)

    while True:
        delta = get_neuron_error(inputs, desired_outputs, weights)

        RMSEs.append(get_RMSE(inputs, desired_outputs,
                     weights, neuron_error=delta))

        if (RMSEs[-1] <= tolerance_threshold):
            break

        for j in range(N):
            if delta[j] != 0:
                weights = [weights[i] + alpha * delta[j] *
                           (1 if i == 0 else 1 / inputs[j][i - 1]) for i in range(len(weights))]

        iterations += 1

    real_outputs = [get_neuron_output(
        input, weights, activation_function) for input in inputs]

    return {'tolerance_threshold': tolerance_threshold, 'weights': weights, 'iterations': iterations,
            'RMSEs': RMSEs, 'desired_outputs': desired_outputs, 'real_outputs': real_outputs}
