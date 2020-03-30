__all__ = ['btc_constant_pattern_solution']
__version__ = '0.2'
__author__ = 'Jure Sencar'


import typing as _typing
import numpy as _np

import bio_rtd.logger as _logger


def btc_constant_pattern_solution(m_load: _np.ndarray,
                                  dbc_100: float,
                                  k: float,
                                  cv: float,
                                  logger: _typing.Union[_logger.RtdLogger, None] = None) -> _np.ndarray:

    result = k * (dbc_100 - m_load / cv)
    # prevent overflow in exp (argument over 100 and under -100)
    result[result > 100] = 100
    result[result < -100] = -100
    result = 1 / (1 + _np.exp(result))

    if logger is not None and k * dbc_100 < 4.6:
        logger.w("Breakthrough profile is suspiciously broad")

    return result
