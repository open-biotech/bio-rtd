import unittest
import numpy as np

from bio_rtd import peak_shapes
from bio_rtd.peak_fitting import calc_emg_parameters_from_peak_shape


class FitEmgTest(unittest.TestCase):

    def run_emg_peak_fit(self, t_peak_start: float, t_peak_max: float, t_peak_end: float, relative_threshold: float):
        rt_mean, sigma, skew = calc_emg_parameters_from_peak_shape(t_peak_start, t_peak_max, t_peak_end,
                                                                   relative_threshold)
        v = peak_shapes.emg(np.array([t_peak_start, t_peak_max, t_peak_end]), rt_mean, sigma, skew)

        # try:
        self.assertAlmostEqual(v[0], relative_threshold * v[1], 4)
        self.assertAlmostEqual(v[2], relative_threshold * v[1], 4)
        # except AssertionError as ex:
        #     print(ex)
        #     t = np.linspace(0, t_peak_end * 5, 10000)
        #     p = peak_shapes.emg(t, rt_mean, sigma, skew)
        #     from bokeh.plotting import figure, show
        #     plt_1 = figure(plot_width=690, plot_height=350)
        #     plt_1.line(t, p, line_width=2, color='green')
        #     plt_1.line(t_peak_start * np.ones(2), np.array([0, p.max()]), line_width=2, color='black')
        #     plt_1.line(t_peak_max * np.ones(2), np.array([0, p.max()]), line_width=2, color='green')
        #     plt_1.line(t_peak_end * np.ones(2), np.array([0, p.max()]), line_width=2, color='green')
        #     show(plt_1)

        t = np.linspace(0, t_peak_end * 5, 10000)
        p = peak_shapes.emg(t, rt_mean, sigma, skew)
        self.assertAlmostEqual(p.max(), v[1], 3)

    def test_emg_fit(self):
        self.run_emg_peak_fit(6.4, 10.4, 18.9, 0.1)
        self.run_emg_peak_fit(15.4, 16.4, 20.9, 0.1)
        self.run_emg_peak_fit(143, 150.4, 160.1, 0.05)
        self.run_emg_peak_fit(6.6, 20.4, 45.7, 0.0001)
        self.run_emg_peak_fit(1.2, 18.3, 48.9, 0.1)
        self.run_emg_peak_fit(1.2, 10.2, 48.9, 0.1)
        self.run_emg_peak_fit(18, 24, 65, 0.1)
        self.run_emg_peak_fit(17, 28, 50, 0.1)
        self.run_emg_peak_fit(9, 12, 32.5, 0.1)
