__version__ = '0.2'
__author__ = 'Jure Sencar'

import numpy as _np
from scipy import optimize as _optimize

import bio_rtd.peak_shapes as _peak_shapes


def calc_emg_parameters_from_peak_shape(t_peak_max: float,
                                        t_peak_start: float,
                                        t_peak_end: float,
                                        relative_threshold: float) -> (float, float, float):
    """
    Calculates mean_rt, sigma and skew based on peak start, peak position and peak end

    Parameters
    ----------
    t_peak_max: float
        Peak max position.
    t_peak_start: float
        Peak start position.
    t_peak_end: float
        Peak end position.
    relative_threshold: float
        Relative signal (compared to peak max) for given start and end positions

    Returns
    -------
    (rt_mean, sigma, skew)
        Calculated `rt_mean`.
    """

    sigma_estimate = (t_peak_max - t_peak_start) / (-2 * _np.log(relative_threshold)) ** 0.5

    # find skew, rt_mean and sigma
    ab_t = _np.linspace(t_peak_start, t_peak_end, 1000)
    peak_i = _np.argmax(ab_t >= t_peak_max)

    def score_func(x):
        # rt_mean, sigma, skew = x
        y = _peak_shapes.emg(ab_t, x[0], x[1], x[2])
        max_y = max(y)
        if max_y == 0:
            return 10
        scr = (1 - y[peak_i] / max_y) ** 2 + \
              (relative_threshold - y[0] / max_y) ** 2 + \
              (relative_threshold - y[-1] / max_y) ** 2
        return scr

    v = _optimize.minimize(score_func,
                           x0=_np.array([t_peak_max, sigma_estimate, 0.1]),
                           bounds=((t_peak_start, t_peak_end),
                                   (sigma_estimate / 10, sigma_estimate * 10),
                                   (0.0001, 1)),
                           )

    assert v.success, "Peak fit did not converge"

    # return mean residence time
    return v.x
