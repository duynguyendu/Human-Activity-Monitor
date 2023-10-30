from typing import Union, Tuple, List
import os


__all__ = [
    "workers_handler",
    "tuple_handler"
]



def workers_handler(value: Union[int, float]) -> int:
    """
    Calculate the number of workers based on an input value.

    Args:
        value (int | float): The input value to determine the number of workers.

    Returns:
        int: The computed number of workers for parallel processing.
    """
    max_workers = os.cpu_count()
    match value:
        case int():
            workers = value
        case float():
            workers = int(max_workers * value)
        case _:
            workers = 0
    if not (-1 < workers < max_workers):
        raise ValueError(f"Number of workers is out of bounds. Min: 0 | Max: {max_workers}")
    return workers


def tuple_handler(value: Union[int, List[int], Tuple[int]], max_dim: int) -> Tuple:
    """
    Create a tuple with specified dimensions and values.

    Args:
        value (Union[int, List[int], Tuple[int]]): The value(s) to populate the tuple with.
            - If an integer is provided, a tuple with 'max_dim' elements, each set to this integer, is created.
            - If a tuple or list of integers is provided, it should have 'max_dim' elements.
        max_dim (int): The desired dimension (length) of the resulting tuple.

    Returns:
        Tuple: A tuple containing the specified values.

    Raises:
        TypeError: If 'max_dim' is not an integer or is less than or equal to 1.
        TypeError: If 'value' is not an integer, tuple, or list.
        ValueError: If the length of 'value' is not equal to 'max_dim'.
    """

    # Check max_dim
    if not isinstance(max_dim, int) and max_dim > 1:
        raise TypeError(f"The 'max_dim' parameter must be an int. Got {type(max_dim)} instead.")
    # Check value
    if isinstance(value, int):
        output = tuple([value] * max_dim)
    else:
        try:
            output = tuple(value)
        except:
            raise TypeError(f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead.")
    if len(output) != max_dim:
        raise ValueError(f"The lenght of 'value' parameter must be equal to {max_dim}. Got {len(output)} instead.")
    return output
