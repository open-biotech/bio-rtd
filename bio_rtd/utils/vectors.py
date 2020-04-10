"""Helper functions for vector operations."""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as _np
import typing as _typing


def true_start(boolean_vector: _np.ndarray) -> int:
    """Accepts array of booleans and returns index of first `True`.

    Parameters
    ----------
    boolean_vector
        Boolean vector. It must contain at least one `True` value.

    Returns
    -------
    int
        Index for first `True` value in `boolean_vector`.

    Raises
    ------
    ValueError
        If all values in `boolean_vector` are `False`.

    """
    r = boolean_vector.flatten().argmax()
    if r == 0 and not boolean_vector.flatten()[0]:
        raise ValueError("all values are False")
    else:
        return r


def true_end(boolean_vector: _np.ndarray) -> int:
    """Accepts array of booleans and returns index AFTER last `True`.

    Parameters
    ----------
    boolean_vector
        Boolean vector. It must contain at least one `True` value.

    Returns
    -------
    int
        Index AFTER last `True` value in `boolean_vector`.

    Raises
    ------
    ValueError
        If all values in `boolean_vector` are `False`.

    """
    r = boolean_vector.size - boolean_vector[::-1].argmax()
    if r == boolean_vector.size and not boolean_vector[-1]:
        raise ValueError("all values are False")
    else:
        return r


def true_start_and_end(boolean_vector: _np.ndarray) -> _typing.Tuple[int, int]:
    """Returns indexes of first True value and AFTER last `True` value.

    Parameters
    ----------
    boolean_vector
        Boolean vector. It must contain at least one `True` value.

    Returns
    -------
    index_of_first_true
        Index of first `True` value in `boolean_vector`.
    index_after_last_true
        Index AFTER last `True` value in `boolean_vector`.

    Raises
    ------
    ValueError
        If all values in `boolean_vector` are `False`.

    """
    return true_start(boolean_vector), true_end(boolean_vector)
