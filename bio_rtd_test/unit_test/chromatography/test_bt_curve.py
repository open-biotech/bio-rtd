import unittest
import numpy as np

from bio_rtd.chromatography import bt_curve
from bio_rtd_test.aux_bio_rtd_test import TestLogger


class TestBtcConstantPatternSolution(unittest.TestCase):

    def test_btc_constant_pattern_solution(self):
        def run_test(cv, dbc_100, k, logger=None):
            m = np.array([0, dbc_100 * cv, dbc_100 * cv * 2, 0.1, dbc_100 * cv * 1.1, dbc_100 * cv * 3])
            r_target = 1 / (1 + np.exp(k * (dbc_100 - m / cv)))
            r_target[0] = 1 / (1 + np.exp(k * dbc_100))
            r_target[1] = 0.5
            r_target[2] = 1 / (1 + np.exp(- k * dbc_100))
            r = bt_curve.btc_constant_pattern_solution(m, dbc_100, k, cv, logger)
            np.testing.assert_array_almost_equal(r_target, r)

        log = TestLogger()
        run_test(14.5, 120, 0.2)
        run_test(14.5, 120, 0.2, log)
        run_test(4.5, 2, 1.2)
        with self.assertWarns(Warning):
            run_test(4.5, 2, 1.2, log)
        run_test(21.5, 20, 0.5)
