from typing import Any, Callable, Dict, Sequence


def discrete_range_mappers_validation(k: int, a: float = -1, b: float = 1):
    if k < 1:
        raise Exception('K should be integer greater than or equal to 1')
    if a >= b:
        raise Exception('Left end of range should be less that the right one')


def discrete_to_range(k: int, a: float = -1, b: float = 1) -> Callable[[int], float]:
    discrete_range_mappers_validation(k, a, b)

    step = (b - a) / (2 * k)

    intervals_centers = [a + (2 * i + 1) * step for i in range(k)]

    def mapper(discrete_value: int) -> float:
        if discrete_value < 0 or discrete_value >= k:
            raise Exception(f'Discrete value should be in range 0, {k - 1}')

        return intervals_centers[discrete_value]

    return mapper


def range_to_discrete(k: int, a: float = -1, b: float = 1) -> Callable[[float], int]:
    discrete_range_mappers_validation(k, a, b)

    step = (b - a) / (2 * k)

    intervals_right_ends = [a + 2 * (i + 1) * step for i in range(k)]

    def mapper(value_from_range: float) -> int:
        if value_from_range < a or value_from_range >= b:
            raise Exception(f'Value from range should be in range [{a}, {b})')

        for i in range(k):
            if value_from_range < intervals_right_ends[i]:
                return i

        return k - 1

    return mapper


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    print_strs = []

    for i in range(len(results)):
        result = results[i]

        print_str = ''

        for key in result:
            key_split = key.split('_')

            formatted_key_name = ' '.join([key_split[i] if i > 0 else key_split[i][0].upper(
            ) + key_split[i][1:] for i in range(len(key_split))])

            print_str += f'{formatted_key_name}: {result[key]}\n'

        print_strs.append(print_str)

    print("\n".join(print_strs))
