import unittest
import numpy as np

from bio_rtd.utils.vectors import true_start
from bio_rtd import peak_shapes
from bio_rtd_test.aux_bio_rtd_test import TestLogger


class PeakShapeTest(unittest.TestCase):

    def setUp(self):
        self.log = TestLogger()

    def validate_normalization(self, t: np.ndarray, p: np.ndarray):
        dt = t[1] - t[0]
        self.assertAlmostEqual(p.sum() * dt, 1, 2)

    def validate_rt_mean(self, t: np.ndarray, p: np.ndarray, rt_mean: float):
        dt = t[1] - t[0]
        self.assertAlmostEqual((p * t).sum() * dt, rt_mean, 3)

    def validate_point_on_slope(self, t: np.ndarray, p: np.ndarray, x: float, y: float):
        i = true_start(t >= x)
        # y should be between p[i-1] and p[i]
        self.assertTrue(p[i - 1] < y <= p[i] or p[i - 1] > y >= p[i])

    def validate_max(self, t: np.ndarray, p: np.ndarray, x: float):
        i = p.argmax()
        self.assertTrue(t[i - 1] < x <= t[i + 1])

    # noinspection DuplicatedCode
    def run_gauss(self, t: np.ndarray, rt_mean: float, sigma: float, validate_results=True):
        p = peak_shapes.gaussian(t, rt_mean, sigma, self.log)
        if validate_results:
            # validations
            self.validate_normalization(t, p)
            self.validate_max(t, p, rt_mean)
            self.validate_rt_mean(t, p, rt_mean)
            # check start and end at half max signal
            w_half = sigma * np.sqrt(2 * np.log(2))
            self.validate_point_on_slope(t, p, rt_mean - w_half, 0.5 * p.max())
            self.validate_point_on_slope(t, p, rt_mean + w_half, 0.5 * p.max())

    def test_gauss(self):
        self.run_gauss(np.linspace(0, 100, 100000), rt_mean=22.5, sigma=3.4)
        self.run_gauss(np.linspace(0, 1000, 3000), rt_mean=50.6, sigma=10)
        self.run_gauss(np.linspace(0, 150, 2000), rt_mean=10, sigma=1)
        # warnings
        with self.assertWarns(Warning, msg="Gaussian peak: sigma < 4 * dt"):
            self.run_gauss(np.linspace(0, 1000, 2000), rt_mean=3.5, sigma=1)
        with self.assertRaises(RuntimeError):
            with self.assertWarns(Warning, msg="Gaussian peak: rt_mean + 3 * sigma > t[-1]"):
                self.run_gauss(np.linspace(0, 100, 1000), rt_mean=35, sigma=0.01 + (100 - 35) / 3,
                               validate_results=False)
        with self.assertWarns(Warning, msg="Gaussian peak: rt_mean - 3 * sigma < t[0]"):
            self.run_gauss(np.linspace(0, 100, 1000), rt_mean=35, sigma=0.01 + 35 / 3,
                           validate_results=False)

    def test_emg(self) -> None:
        t = np.linspace(0, 100, 1000)
        rt_mean = 20
        p = peak_shapes.emg(t, rt_mean=rt_mean, sigma=2, skew=1, logger=self.log)

        # validations
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 15.33, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 24.80, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 17.35, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 22.47, 0.5 * p.max())
        self.validate_max(t, p, 19.82)

        # warnings

        with self.assertWarns(Warning, msg="EMG peak: sigma < 4 * dt"):
            peak_shapes.emg(t, rt_mean=rt_mean, sigma=4 * t[1] - 0.01, skew=1, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Peak shape: integral: 0.7723355216030109"):
            with self.assertWarns(Warning, msg="EMG peak: rt_mean + 3 * sigma > t[-1]"):
                peak_shapes.emg(t, rt_mean=rt_mean, sigma=(t[-1] - rt_mean) / 3 + 0.01, skew=1, logger=self.log)

        with self.assertWarns(Warning, msg="EMG peak: t0 < 3 * sigma; t0 = rt_mean - 1 / skew"):
            skew = 0.8
            t0 = rt_mean - 1 / skew
            peak_shapes.emg(t, rt_mean=rt_mean, sigma=t0 / 3 - 0.01, skew=1, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Peak shape: integral: 0.5483504934228841"):
            with self.assertWarns(Warning, msg="EMG peak: skew < 1/40"):
                peak_shapes.emg(t, rt_mean=rt_mean, sigma=2, skew=1 / 41, logger=self.log)

        with self.assertWarns(Warning, msg="EMG peak: skew > 10"):
            peak_shapes.emg(t, rt_mean=rt_mean, sigma=2, skew=10.1, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Peak shape: relative value at start: 0.1359995541938657"):
            with self.assertWarns(Warning, msg="EMG peak: exp argument (p) > 200"):
                skew = 2
                sigma = 10
                t0 = rt_mean - 1 / skew
                assert np.any(skew / 2 * (2 * (t0 - t) + skew * sigma ** 2) > 200)
                peak_shapes.emg(t, rt_mean=rt_mean, sigma=sigma, skew=skew, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Peak shape: integral: 0.0069704155745190025"):
            with self.assertWarns(Warning, msg="EMG peak: exp argument (p) < -200"):
                skew = 40
                sigma = 1
                t0 = rt_mean - 1 / skew
                assert np.any(skew / 2 * (2 * (t0 - t) + skew * sigma ** 2) < -200)
                peak_shapes.emg(t, rt_mean=rt_mean, sigma=sigma, skew=skew, logger=self.log)

        # 2nd test
        t = np.linspace(0, 150, 5000)
        rt_mean = 40
        p = peak_shapes.emg(t, rt_mean, 7.4, 0.26, self.log)
        # validations
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 22.62725, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 57.90722, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 30.12018, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 49.14717, 0.5 * p.max())
        self.validate_max(t, p, 39.41918)

    # noinspection DuplicatedCode
    def test_skew_normal(self) -> None:
        t = np.linspace(0, 150, 2000)
        rt_mean = 83.5
        sigma = 10.5
        p = peak_shapes.skew_normal(t, rt_mean, sigma, 0, self.log)
        # should be same as for gauss
        self.validate_normalization(t, p)
        self.validate_max(t, p, rt_mean)
        self.validate_rt_mean(t, p, rt_mean)
        w_half = sigma * np.sqrt(2 * np.log(2))
        self.validate_point_on_slope(t, p, rt_mean - w_half, 0.5 * p.max())
        self.validate_point_on_slope(t, p, rt_mean + w_half, 0.5 * p.max())

        # warnings

        with self.assertWarns(Warning, msg="Skewed normal peak: sigma < 4 * dt"):
            peak_shapes.skew_normal(t, rt_mean=rt_mean, sigma=4 * t[1] - 0.01, skew=1, logger=self.log)

        with self.assertWarns(Warning, msg="Skewed normal peak: rt_mean + 3 * sigma > t[-1]"):
            peak_shapes.skew_normal(t, rt_mean=rt_mean, sigma=(t[-1] - rt_mean) / 3 + 0.01, skew=1, logger=self.log)

        with self.assertWarns(Warning, msg="Skewed normal peak: rt_mean < 3 * sigma"):
            peak_shapes.skew_normal(t, rt_mean=rt_mean, sigma=rt_mean / 3 + 0.01, skew=1, logger=self.log)

        with self.assertWarns(Warning, msg="Skewed normal peak: skew < -20"):
            peak_shapes.skew_normal(t, rt_mean=rt_mean, sigma=sigma, skew=-20.1, logger=self.log)

        with self.assertWarns(Warning, msg="Skewed normal peak: skew > 20"):
            peak_shapes.skew_normal(t, rt_mean=rt_mean, sigma=sigma, skew=20.1, logger=self.log)

        # test 2
        t = np.linspace(0, 150, 3000)
        rt_mean = 55
        p = peak_shapes.skew_normal(t, rt_mean, 7.4, 3.87, self.log)
        # validations
        self.validate_normalization(t, p)
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 46.74376, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 65.64461, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 48.96744, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 58.84496, 0.5 * p.max())
        self.validate_max(t, p, 52.41927)

        # test 3
        t = np.linspace(0, 50, 2500)
        rt_mean = 35
        p = peak_shapes.skew_normal(t, rt_mean, 3.5, -8, self.log)
        # validations
        self.validate_normalization(t, p)
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 30.17614, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 38.34127, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 33.49889, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 37.79871, 0.5 * p.max())
        self.validate_max(t, p, 36.80115)

    def test_n_tanks_in_series(self) -> None:
        # test 1
        t = np.linspace(0, 150, 2000)
        rt_mean = 10.5
        n_tanks = 1
        p = peak_shapes.tanks_in_series(t, rt_mean, n_tanks, self.log)
        # should be same as exp decay
        self.validate_normalization(t, p)
        self.validate_rt_mean(t, p, rt_mean)
        self.assertTrue(p[0] == p.max())
        self.validate_point_on_slope(t, p, rt_mean * np.log(2), 0.5 * p.max())

        # test 2
        t = np.linspace(0, 100, 350)
        rt_mean = 7.8
        n_tanks = 4
        p = peak_shapes.tanks_in_series(t, rt_mean, n_tanks, self.log)
        # validations
        self.validate_normalization(t, p)
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 1.233363, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 16.35403, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 2.718567, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 10.77443, 0.5 * p.max())
        self.validate_max(t, p, 5.850017)

        # warnings & errors

        with self.assertWarns(Warning, msg="Tanks in series peak: rt_mean > t[-1] / 4"):
            peak_shapes.tanks_in_series(t, rt_mean=t[-1] / 4 + 0.01, n_tanks=4, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Tanks in series peak: Initial time point > 0"):
            peak_shapes.tanks_in_series(t + 0.1, rt_mean=rt_mean, n_tanks=4, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Tanks in series peak: n_tanks " + str(0) + " (< 1)"):
            peak_shapes.tanks_in_series(t, rt_mean=rt_mean, n_tanks=0, logger=self.log)

        with self.assertRaises(RuntimeError, msg="Peak shape: integral: 1.0011351261590138e+34"):
            with self.assertWarns(Warning, msg="Tanks in series peak: n_tanks " + str(51) + " (> 50)"):
                peak_shapes.tanks_in_series(t, rt_mean=rt_mean, n_tanks=51, logger=self.log)

        # test 3
        t = np.linspace(0, 100, 2500)
        rt_mean = 15.2
        n_tanks = 15
        p = peak_shapes.tanks_in_series(t, rt_mean, n_tanks, self.log)
        # validations
        self.validate_normalization(t, p)
        self.validate_rt_mean(t, p, rt_mean)
        self.validate_point_on_slope(t, p, 7.525465, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 23.94761, 0.1 * p.max())
        self.validate_point_on_slope(t, p, 19.13091, 0.5 * p.max())
        self.validate_point_on_slope(t, p, 19.13090, 0.5 * p.max())
        self.validate_max(t, p, 14.18667)

    def test_report_suspicious_peak_shape(self):
        t = np.linspace(0, 15, 200)
        dt = t[1]
        p = np.zeros_like(t)
        p[50] = 1 / dt
        rt_mean = t[50]

        # ok
        peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)

        # normalization
        p[50] = 0.88 / dt
        with self.assertRaises(RuntimeError, msg="Peak shape: integral: 0.88"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        p[50] = 0.98 / dt
        with self.assertWarns(Warning, msg="Peak shape: integral: 0.98"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)

        # value of initial point
        p[0] = 0.06
        p[50] = 1
        p = p / p.sum() / dt
        with self.assertRaises(RuntimeError, msg="Peak shape: relative value at start: 0.06"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        p[0] = 0.0011
        p[50] = 1
        p = p / p.sum() / dt
        with self.assertWarns(Warning, msg="Peak shape: relative value at start: 0.0011"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        p[0] = 0

        # value of end point
        p[-1] = 0.06
        p[50] = 1
        p = p / p.sum() / dt
        with self.assertRaises(RuntimeError, msg="Peak shape: relative value at end: 0.06"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        p[-1] = 0.0011
        p[50] = 1
        p = p / p.sum() / dt
        with self.assertWarns(Warning, msg="Peak shape: relative value at end: 0.0011"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        p[-1] = 0

        # value of rt_mean
        p[50] = 1 / dt
        rt_mean = t[50] * 0.89
        with self.assertRaises(RuntimeError, msg="Peak shape: relative difference in rt_mean: 0.89"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
        rt_mean = t[50] * 0.989
        with self.assertWarns(Warning, msg="Peak shape: relative difference in rt_mean: 0.989"):
            peak_shapes._report_suspicious_peak_shape(t, p, rt_mean, self.log)
