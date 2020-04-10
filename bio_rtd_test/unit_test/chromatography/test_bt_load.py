import unittest

import numpy as np

from bio_rtd.chromatography import bt_curve, bt_load
from bio_rtd.core import ChromatographyLoadBreakthrough
from bio_rtd_test.aux_bio_rtd_test import TestLogger


class DummyChromatographyLoadBreakthrough(ChromatographyLoadBreakthrough):
    POSSIBLE_KEY_GROUPS = []
    OPTIONAL_KEYS = []

    def _calc_unbound_to_load_ratio(self, loaded_material: np.ndarray) -> np.ndarray:
        return np.ones_like(loaded_material) * 0.7

    def _update_btc_parameters(self, **kwargs) -> None:  # pragma: no cover
        pass

    def get_total_bc(self) -> float:  # pragma: no cover
        pass


class TestChromatographyLoadBreakthrough(unittest.TestCase):

    def test_calc_c_bound(self):
        t = np.linspace(0, 10, 100)
        f = np.ones_like(t) * 0.2
        c = np.ones([1, t.size])
        bt = DummyChromatographyLoadBreakthrough(t[1])
        # zero in -> zero out
        self.assertTrue(np.all(bt.calc_c_bound(f, c * 0) == 0))
        # normal function
        np.testing.assert_array_equal(
            c * (1 - 0.7),
            bt.calc_c_bound(f, c)
        )


class TestConstantPatternSolution(unittest.TestCase):

    def btc_init(self, dbc_100, k):
        dt = 0.4
        self.btc = bt_load.ConstantPatternSolution(dt, dbc_100, k)
        self.btc.set_logger_from_parent("id", TestLogger())
        self.assertEqual(self.btc.k, k)
        self.assertEqual(self.btc.dbc_100, dbc_100)
        self.assertEqual(self.btc._cv, -1)

    def test_update_btc_parameters(self):
        self.btc_init(120, 0.2)
        cv = 14.5
        with self.assertRaises(KeyError):
            self.btc.update_btc_parameters(cv_not_right=cv)
        self.assertEqual(self.btc._cv, -1)
        self.btc.update_btc_parameters(cv=cv)
        self.assertEqual(self.btc._cv, cv)

    def test_calc_unbound_to_load_ratio(self):
        def run_test(cv, dbc_100, k):
            m = np.array([0, dbc_100 * cv, dbc_100 * cv * 2, 0.1, dbc_100 * cv * 1.1, dbc_100 * cv * 3])
            r_target = bt_curve.btc_constant_pattern_solution(m, dbc_100, k, cv, None)
            self.btc_init(dbc_100, k)
            self.btc.update_btc_parameters(cv=cv)
            r = self.btc._calc_unbound_to_load_ratio(m)
            np.testing.assert_array_almost_equal(r_target, r)

        self.btc_init(120, 0.2)
        with self.assertRaises(AssertionError):  # update_btc_parameters must be called
            self.btc._calc_unbound_to_load_ratio(np.array([]))
        run_test(14.5, 120, 0.2)
        with self.assertWarns(Warning):
            run_test(4.5, 2, 1.2)
        run_test(21.5, 20, 0.5)
        with self.assertWarns(Warning):
            run_test(4.5, 2, 1.2)

    def test_get_total_bc(self):
        self.btc_init(120, 0.2)

        with self.assertRaises(AssertionError):  # cv is undefined
            self.btc.get_total_bc()

        self.btc._cv = 23.3

        self.assertEqual(self.btc._cv * self.btc.dbc_100,
                         self.btc.get_total_bc())
