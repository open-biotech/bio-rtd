"""
Peak shapes based on mean residence time (`rt_mean` == first momentum)

For un-clipped peak, the integral over the peak over time == 1.
For un-clipped peak, first momentum == `rt_mean`.
"""

__version__ = '0.2'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

from scipy import special as _special

from bio_rtd.logger import RtdLogger as _RtdLogger


def gaussian(t: _np.ndarray, rt_mean: float, sigma: float,
             logger: _typing.Union[_RtdLogger, None] = None) -> _np.ndarray:
    if logger:  # warnings for unsuitable peak parameters
        if sigma < 4 * (t[1] - t[0]):
            logger.w("Gaussian peak: sigma < 4 * dt")
        if rt_mean + 3 * sigma > t[-1]:
            logger.w("Gaussian peak: rt_mean + 3 * sigma > t[-1]")
        if rt_mean - 3 * sigma < t[0]:
            logger.w("Gaussian peak: rt_mean - 3 * sigma < t[0]")
    p = - (t - rt_mean) ** 2 / 2 / sigma ** 2
    p[p > 700] = 700
    p[p < -700] = -700
    p = _np.exp(p) / sigma / _np.sqrt(2 * _np.pi)

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


def emg(t, rt_mean, sigma, skew, logger: _typing.Union[_RtdLogger, None] = None):
    # first momentum of emg = rt_mean = t0 + 1 / skew
    t0 = rt_mean - 1 / skew
    # exp argument
    p = skew / 2 * (2 * (t0 - t) + skew * sigma ** 2)

    if logger:  # warnings for unsuitable peak parameters
        if sigma < 4 * (t[1] - t[0]):
            logger.w("EMG peak: sigma < 4 * dt")
        if rt_mean + 3 * sigma > t[-1]:
            logger.w("EMG peak: rt_mean + 3 * sigma > t[-1]")
        if t0 < 3 * sigma:
            logger.w("EMG peak: t0 < 3 * sigma; t0 = rt_mean - 1 / skew")
        if skew < 1 / 40:
            logger.w("EMG peak: skew < 1/40")
        if skew > 10:
            logger.w("EMG peak: skew > 10")
        if _np.any(p > 200):
            logger.w("EMG peak: exp argument (p) > 200")
        if _np.any(p < -200):
            logger.w("EMG peak: exp argument (p) < -200")

    p[p > 700] = 700
    p[p < -700] = -700
    # curve
    p = _np.exp(p) * _special.erfc((t0 - t + skew * sigma ** 2) / (2 ** 0.5 * sigma)) * skew / 2

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


def skew_normal(t, rt_mean, sigma, skew, logger: _typing.Union[_RtdLogger, None] = None):
    if logger:  # warnings for unsuitable peak parameters
        if sigma < 4 * (t[1] - t[0]):
            logger.w("Skewed normal peak: sigma < 4 * dt")
        if rt_mean + 3 * sigma > t[-1]:
            logger.w("Skewed normal peak: rt_mean + 3 * sigma > t[-1]")
        if rt_mean < 3 * sigma:
            logger.w("Skewed normal peak: rt_mean < 3 * sigma")
        if skew < -20:
            logger.w("Skewed normal peak: skew < -20")
        if skew > 20:
            logger.w("Skewed normal peak: skew > 20")

    # rt_mean = t0 + sigma * np.sqrt(2 / np.pi) * skew / (1 + skew**2)
    t0 = rt_mean - sigma * _np.sqrt(2 / _np.pi) * skew / _np.sqrt(1 + skew ** 2)
    # skew
    x = (t - t0) / sigma
    p = 2 * gaussian(t, t0, sigma) * 1 / 2 * (1 + _special.erf(skew * x / _np.sqrt(2)))

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger)
    return p


def tanks_in_series(t: _np.ndarray, rt_mean: float, n_tanks: int,
                    logger: _typing.Union[_RtdLogger, None] = None) -> _np.ndarray:
    """`rt_mean` is for entire unit operation (all tanks together)"""

    if logger:  # warnings for unsuitable peak parameters
        if rt_mean > t[-1] / 4:
            logger.w("Tanks in series peak: rt_mean > t[-1] / 4")
        if t[0] > 0:
            logger.e("Tanks in series peak: Initial time point > 0")
        if n_tanks < 1:
            logger.e("Tanks in series peak: n_tanks " + str(n_tanks) + " (< 1)")
        if n_tanks > 50:
            logger.w("Tanks in series peak: n_tanks " + str(n_tanks) + " (> 50)")

    if n_tanks == 1:
        p = _np.exp(_np.clip(-t / rt_mean, -100, 0)) / rt_mean
    else:
        p = t ** (n_tanks - 1) / _np.math.factorial(n_tanks - 1) / (rt_mean / n_tanks) ** n_tanks \
            * _np.exp(_np.clip(-t / rt_mean * n_tanks, -100, 0))

    if logger:
        _report_suspicious_peak_shape(t, p, rt_mean, logger, ignore_min_start=True)
    return p


def _report_suspicious_peak_shape(
        t: _np.ndarray, p: _np.ndarray, rt_mean: float, logger: _typing.Union[_RtdLogger, None],
        ignore_min_start=False
):
    dt = t[1] - t[0]

    # check normalization
    s = p.sum() * dt
    if abs(s - 1) > 0.1:
        logger.e("Peak shape: integral: " + str(s))
    elif abs(s - 1) > 0.01:
        logger.w("Peak shape: integral: " + str(s))

    # check values at edges
    if not ignore_min_start:
        rel_start = p[0] / p.max()
        if rel_start > 0.05:
            logger.e("Peak shape: relative value at start: " + str(rel_start))
        elif rel_start > 0.001:
            logger.w("Peak shape: relative value at start: " + str(rel_start))

    rel_end = p[-1] / p.max()
    if rel_end > 0.05:
        logger.e("Peak shape: relative value at end: " + str(rel_end))
    elif rel_end > 0.001:
        logger.w("Peak shape: relative value at end: " + str(rel_end))

    # check rt_mean
    p_rt_mean = _np.sum(t * p) * dt
    rel_diff = abs(rt_mean - p_rt_mean) / rt_mean
    if rel_diff > 0.1:
        logger.e("Peak shape: relative difference in rt_mean: " + str(rel_diff))
    elif rel_diff > 0.01:
        logger.w("Peak shape: relative difference in rt_mean: " + str(rel_diff))
