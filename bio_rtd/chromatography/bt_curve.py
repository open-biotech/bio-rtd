"""Breakthrough curves.

Breakthrough curves provide information on what ratio of inlet does not
bind to the column after certain amount of material was already loaded
(not all of loaded material is bound) onto the column.

"""

__all__ = ['btc_constant_pattern_solution']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.logger as _logger


def btc_constant_pattern_solution(
        m_load: _np.ndarray,
        dbc_100: float,
        k: float,
        cv: float,
        logger: _typing.Union[_logger.RtdLogger, None] = None) -> _np.ndarray:
    """Breakthrough curve - Constant Pattern Solution.

    r = 1 / (1 + exp(`k` * (`dbc_100` - `m_load` / `cv`)))

    Parameters
    ----------
    m_load
        Amount of load material already sent onto the column.
    dbc_100
        Dynamic binding capacity if the load would last indefinitely.
    k
        Steepness of the breakthrough profile.
    cv
        Column volume.
    logger
        Logger for messaging about potential suspicious profiles.

    Returns
    -------
    r: ndarray
        Share of unbound material for given `m_load`.

        `r`.shape == `m_load`.shape

    """
    result = k * (dbc_100 - m_load / cv)
    # prevent overflow in exp (argument over 100 and under -100)
    result[result > 100] = 100
    result[result < -100] = -100
    result = 1 / (1 + _np.exp(result))

    if logger is not None and k * dbc_100 < 4.6:
        logger.w("Breakthrough profile is suspiciously broad")

    return result
