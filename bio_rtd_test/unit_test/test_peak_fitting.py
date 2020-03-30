import unittest
import numpy as np

from bio_rtd import peak_shapes
from bio_rtd.peak_fitting import calc_emg_parameters_from_peak_shape


class FitEmgTest(unittest.TestCase):

    def run_emg_peak_fit(self,
                         t_peak_max: float,
                         t_peak_start: float,
                         t_peak_end: float,
                         relative_threshold: float):
        rt_mean, sigma, skew = calc_emg_parameters_from_peak_shape(t_peak_max,
                                                                   t_peak_start,
                                                                   t_peak_end,
                                                                   relative_threshold)
        v = peak_shapes.emg(np.array([t_peak_start, t_peak_max, t_peak_end]), rt_mean, sigma, skew)

        self.assertAlmostEqual(v[0], relative_threshold * v[1], 4)
        self.assertAlmostEqual(v[2], relative_threshold * v[1], 4)
        t = np.linspace(0, t_peak_end * 5, 10000)
        p = peak_shapes.emg(t, rt_mean, sigma, skew)
        self.assertAlmostEqual(p.max(), v[1], 3)

    def test_emg_fit(self):
        self.run_emg_peak_fit(10.4, 6.4, 18.9, 0.1)
        self.run_emg_peak_fit(16.4, 15.4, 20.9, 0.1)
        self.run_emg_peak_fit(150.4, 143, 160.1, 0.05)
        self.run_emg_peak_fit(20.4, 6.6, 45.7, 0.0001)
        self.run_emg_peak_fit(18.3, 1.2, 48.9, 0.1)
        self.run_emg_peak_fit(10000.2, 1.2, 48.9, 0.1)
