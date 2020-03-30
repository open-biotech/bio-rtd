__version__ = '0.1'
__author__ = 'Jure Sencar'

import numpy as _np


def true_start(boolean_vector: _np.ndarray) -> int:
    """accepts array of booleans and returns index of first true"""
    r = boolean_vector.flatten().argmax()
    if r == 0 and not boolean_vector.flatten()[0]:
        raise ValueError("all values are False")
    else:
        return r


def true_end(boolean_vector: _np.ndarray) -> int:
    """accepts array of booleans and returns index AFTER last true"""
    r = boolean_vector.size - boolean_vector[::-1].argmax()
    if r == boolean_vector.size and not boolean_vector[-1]:
        raise ValueError("all values are False")
    else:
        return r


def true_start_and_end(boolean_vector: _np.ndarray) -> (int, int):
    """accepts array of booleans and returns indexes of first and AFTER last true"""
    return true_start(boolean_vector), true_end(boolean_vector)
