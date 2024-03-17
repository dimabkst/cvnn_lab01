from traceback import print_exc
from typing import Literal, Sequence, Union

from learning import error_correction_learning_with_RMSE_stopping
from services import discrete_to_range, print_results, range_to_discrete


def lab_task(discrete_inputs: Sequence[Sequence[int]], discrete_desired_outputs: Sequence[int],
             tolerance_threshold: float,
             starting_weights_type: Literal['random',
                                            'external'] = 'random',
             starting_weights: Union[Sequence[float], None] = None):
    discrete_elements = []

    for discrete_input in discrete_inputs:
        discrete_elements += discrete_input

    discrete_elements += discrete_desired_outputs

    k = max(discrete_elements) + 1

    discrete_to_range_mapper = discrete_to_range(k)

    inputs = [[discrete_to_range_mapper(
        el) for el in discrete_input] for discrete_input in discrete_inputs]

    desired_outputs = [discrete_to_range_mapper(
        el) for el in discrete_desired_outputs]

    result = error_correction_learning_with_RMSE_stopping(
        inputs, desired_outputs, tolerance_threshold, starting_weights_type, starting_weights)

    range_to_discrete_mapper = range_to_discrete(k)

    result['discrete_desired_outputs'] = discrete_desired_outputs

    result['discrete_real_outputs'] = [range_to_discrete_mapper(
        el) for el in result['real_outputs']]

    return result


if __name__ == '__main__':
    try:
        inputs = [[0, 2, 1], [1, 2, 3], [2, 1, 3]]

        desired_outputs = [0, 1, 2]

        tolerance_thresholds = [0.1, 0.01, 0.0001]

        results = [lab_task(inputs, desired_outputs, tolerance_threshold=t, starting_weights_type='random')
                   for t in tolerance_thresholds]

        print_results(results)
    except Exception as e:
        print('Error occurred:')

        print_exc()
