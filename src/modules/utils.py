import os


__all__ = [
    "workers_handler"
]

def workers_handler(value: int | float) -> int:
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
