"""
Peak shapes based on mean residence time (`rt_mean`).

Notes
-----
Functions are evaluated for given time vectors. Peaks are considered
clipped if they do not fully fit on the time vector.

For un-clipped peak, the integral over the peak over time == 1.

For un-clipped peak, first momentum == `rt_mean`.

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

from scipy import special as _special

from bio_rtd.logger import RtdLogger as _RtdLogger


def gaussian(t: _np.ndarray, rt_mean: float, sigma: float,
             logger: _typing.Union[_RtdLogger, None] = None) -> _np.ndarray:
    """Gaussian distribution.

    p = exp(- ((t - rt_mean) / sigma) ** 2 / 2) / (sigma * sqrt(2 * pi))

    Parameters
    ----------
    t
        Time vector.
    rt_mean
        Mean residence time (== first momentum of un-clipped peak)
    sigma
        Standard deviation.
    logger
        Logger for logging suspicious parameters or peak shapes.

    Returns
    -------
    p: ndarray
        Evaluated pdf for specified time vector (`t`).

    """
    if logger:  # warnings for unsuitable peak parameters
        if sigma < 4 * (t[1] - t[0]):
            logger.w(f"Gaussian peak: sigma < 4 * dt")
        if rt_mean + 3 * sigma > t[-1]:
            logger.w(f"Gaussian peak: rt_mean + 3 * sigma > t[-1]")
        if rt_mean - 3 * sigma < t[0]:
            logger.w(f"Gaussian peak: rt_mean - 3 * sigma < t[0]")

    p = - (t - rt_mean) ** 2 / 2 / sigma ** 2
    p[p > 700] = 700
    p[p < -700] = -700
    p = _np.exp(p) / sigma / _np.sqrt(2 * _np.pi)

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


# noinspection DuplicatedCode
def emg(t, rt_mean, sigma, skew,
        logger: _typing.Union[_RtdLogger, None] = None):
    """Exponentially modified Gaussian distribution.

    Parameters
    ----------
    t
        Time vector.
    rt_mean
        Mean residence time (== first momentum of un-clipped peak).
    sigma
        Standard deviation of Gaussian part.
    skew
        The rate of exponential part. Recommended: 1/40 < `skew` < 10.
    logger
        Logger for logging suspicious parameters or peak shapes.

    Returns
    -------
    p: ndarray
        Evaluated pdf for specified time vector (`t`).

    """
    # first momentum of emg = rt_mean = t0 + 1 / skew
    t0 = rt_mean - 1 / skew
    # exp argument
    p = skew / 2 * (2 * (t0 - t) + skew * sigma ** 2)

    if logger:  # warnings for unsuitable peak parameters
        if sigma + 1 / skew < 4 * (t[1] - t[0]):
            logger.w(f"EMG peak: sigma + 1 / skew < 4 * dt")
        if rt_mean + 2 * sigma + 1 / skew > t[-1]:
            logger.w(f"EMG peak: rt_mean + 2 * sigma + 2 / skew > t[-1]")
        if t0 < 3 * sigma:
            logger.w(f"EMG peak: t0 < 3 * sigma; t0 = rt_mean - 1 / skew")
        if skew < 1 / 40:
            logger.w(f"EMG peak: skew < 1/40")
        if skew > 10:
            logger.w(f"EMG peak: skew > 10")
        if _np.any(p > 200):
            logger.w(f"EMG peak: exp argument (p) > 200")
        if _np.any(p < -200):  # not that relevant as it results in 0
            logger.i(f"EMG peak: exp argument (p) < -200")

    p[p > 700] = 700
    p[p < -700] = -700
    # curve
    p = _np.exp(p) * _special.erfc(
        (t0 - t + skew * sigma ** 2) / (2 ** 0.5 * sigma)
    ) * skew / 2

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


# noinspection DuplicatedCode
def skewed_normal(t, rt_mean, sigma, skew,
                  logger: _typing.Union[_RtdLogger, None] = None):
    """Skewed normal distribution.

    For `skew` == 0, the distribution becomes Gaussian distribution.

    Parameters
    ----------
    t
        Time vector.
    rt_mean
        Mean residence time (== first momentum of un-clipped peak).
    sigma
        Standard deviation of Gaussian part.
    skew
        Skewness of the peak. Recommended: -20 < `skew` < 20.
    logger
        Logger for logging suspicious parameters or peak shapes.

    Returns
    -------
    p: ndarray
        Evaluated pdf for specified time vector (`t`).

    """
    if logger:  # warnings for unsuitable peak parameters
        if sigma < 4 * (t[1] - t[0]):
            logger.w(f"Skewed normal peak: sigma < 4 * dt")
        if rt_mean + 3 * sigma > t[-1]:
            logger.w(f"Skewed normal peak: rt_mean + 3 * sigma > t[-1]")
        if rt_mean < 3 * sigma:
            logger.w(f"Skewed normal peak: rt_mean < 3 * sigma")
        if skew < -20:
            logger.w(f"Skewed normal peak: skew < -20")
        if skew > 20:
            logger.w(f"Skewed normal peak: skew > 20")

    # rt_mean = t0 + sigma * np.sqrt(2 / np.pi) * skew / (1 + skew**2)
    t0 = rt_mean - sigma * skew * _np.sqrt(2 / _np.pi / (1 + skew ** 2))
    # skew
    x = (t - t0) / sigma
    p = gaussian(t, t0, sigma) * (1 + _special.erf(skew * x / _np.sqrt(2)))

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


def tanks_in_series(t: _np.ndarray, rt_mean: float, n_tanks: int,
                    logger: _typing.Union[_RtdLogger, None] = None,
                    allow_open_end=False,
                    ) -> _np.ndarray:
    """N tanks in series distribution.

    `rt_mean` is for entire unit operation (all tanks together).

    For `n_tanks` == 1, the distribution results in exponential decay.

    Parameters
    ----------
    t
        Time vector.
    rt_mean
        Mean residence time (== first momentum of un-clipped peak).
    n_tanks
        Number of tanks. Recommended: 1 <= `n_tanks` < 50
    logger
        Logger for logging suspicious parameters or peak shapes.

    Returns
    -------
    p: ndarray
        Evaluated pdf for specified time vector (`t`).

    """

    if logger:  # warnings for unsuitable peak parameters
        if rt_mean > t[-1] / 4 and not allow_open_end:
            logger.w(f"Tanks in series peak: rt_mean > t[-1] / 4")
        if t[0] > 0:
            logger.e(f"Tanks in series peak: Initial time point > 0")
        if n_tanks < 1:
            logger.e(f"Tanks in series peak: n_tanks {n_tanks} (< 1)")
        if n_tanks > 50:
            logger.w(f"Tanks in series peak: n_tanks {n_tanks} (> 50)")

    if n_tanks == 1:
        p = _np.exp(_np.clip(-t / rt_mean, -100, 0)) / rt_mean
    else:
        p = t ** (n_tanks - 1) / _np.math.factorial(n_tanks - 1) \
            / (rt_mean / n_tanks) ** n_tanks \
            * _np.exp(_np.clip(-t / rt_mean * n_tanks, -100, 0))

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger,
                                      ignore_min_start=True,
                                      open_end=allow_open_end)
    return p


def _report_suspicious_peak_shape(
        t: _np.ndarray, p: _np.ndarray, rt_mean: float,
        logger: _typing.Union[_RtdLogger, None],
        ignore_min_start=False,
        open_end=False):
    dt = t[1] - t[0]
    # check values at edges
    if not ignore_min_start:
        rel_start = p[0] / p.max()
        if rel_start > 0.05:
            logger.e(f"Peak shape: relative value at start: {rel_start}")
        elif rel_start > 0.001:
            logger.w(f"Peak shape: relative value at start: {rel_start}")
    if not open_end:
        rel_end = p[-1] / p.max()
        if rel_end > 0.05:
            logger.e(f"Peak shape: relative value at end: {rel_end}")
        elif rel_end > 0.001:
            logger.w(f"Peak shape: relative value at end: {rel_end}")
        # check rt_mean
        p_rt_mean = _np.sum(t * p) * dt
        rel_diff = abs(rt_mean - p_rt_mean) / rt_mean
        if rel_diff > 0.1:
            logger.e(f"Peak shape: relative difference in rt_mean: {rel_diff}")
        elif rel_diff > 0.01:
            logger.w(f"Peak shape: relative difference in rt_mean: {rel_diff}")
        # check normalization
        s = p.sum() * dt
        if abs(s - 1) > 0.1:
            logger.e(f"Peak shape: integral: {s}")
        elif abs(s - 1) > 0.01:
            logger.w(f"Peak shape: integral: {s}")
