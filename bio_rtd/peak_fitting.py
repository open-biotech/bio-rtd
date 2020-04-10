"""Helper module for obtaining peak shape parameters from data points.
"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as _np
import typing as _typing
from scipy import optimize as _optimize
from scipy import special as _special

import bio_rtd.peak_shapes as _peak_shapes


def calc_emg_parameters_from_peak_shape(
        t_peak_start: float,
        t_peak_max: float,
        t_peak_end: float,
        relative_threshold: float) -> _typing.Tuple[float, float, float]:
    """Calculate EMG parameters from characteristic data points.

    Parameters
    ----------
    t_peak_start: float
        Peak start position.
    t_peak_max: float
        Peak max position.
    t_peak_end: float
        Peak end position.
    relative_threshold: float
        Relative signal (compared to max) at start and end positions.

    Returns
    -------
    rt_mean
        Mean residence time (== first momentum).
    sigma
        Standard deviation of Gaussian part.
    skew
        The rate of exponential part.

    """

    sigma_estimate = (t_peak_max - t_peak_start) \
        / (-2 * _np.log(relative_threshold)) ** 0.5
    sigma_joined_estimate = (t_peak_end - t_peak_max) \
        / (-2 * _np.log(relative_threshold)) ** 0.5
    sigma_exp_estimate = _np.sqrt(sigma_joined_estimate ** 2 -
                                  sigma_estimate ** 2)
    skew_estimate = 1 / sigma_exp_estimate

    _t = _np.array([t_peak_start, t_peak_max, t_peak_end])

    def get_rt_mean(_sigma, _skew):
        _k = _np.sqrt(_np.pi / 2) * _sigma * _skew

        def df(x):
            return _k * _special.erfc(x) - _np.exp(- x ** 2)

        _x_min = _k / _np.sqrt(_np.pi)
        _x_max = (- 1 / _skew + _skew * _sigma ** 2) / (2 ** 0.5 * _sigma)
        _x = _optimize.root_scalar(df,
                                   bracket=[_x_min, _x_max],
                                   x0=(_x_min + _x_max) / 2)

        _mu = t_peak_max + _x.root * _np.sqrt(2) * _sigma - _skew * _sigma ** 2

        _rt_mean = _mu + 1 / _skew

        return _rt_mean

    def score_func(x):
        # x == [sigma, skew]
        _sigma, _skew = x
        _rt_mean = get_rt_mean(_sigma, _skew)
        _p = _peak_shapes.emg(_t, _rt_mean, _sigma, _skew)
        scr = (relative_threshold - _p[0] / _p[1]) ** 2 * 10 + \
              (relative_threshold - _p[2] / _p[1]) ** 2
        # print(_rt_mean, _sigma, _skew, scr)
        return scr

    v = _optimize.minimize(score_func,
                           x0=_np.array([sigma_estimate, skew_estimate]),
                           bounds=((sigma_estimate / 10, sigma_estimate * 10),
                                   (0.0001, 10)),
                           tol=1e-11,
                           method="TNC"
                           )

    assert v.success, "Peak fit did not converge"
    # print("------")

    # return mean residence time
    # noinspection PyTypeChecker
    return (get_rt_mean(v.x[0], v.x[1]), *v.x)
