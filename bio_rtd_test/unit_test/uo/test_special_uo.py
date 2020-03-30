import unittest

import numpy as np

from bio_rtd import core, logger
from bio_rtd.uo import special_uo


class MockUpUnitOperation(core.UnitOperation):
    add_to_c = 0

    def _calculate(self):
        self._c += self.add_to_c


class TestComboUO(unittest.TestCase):

    def setUp(self) -> None:
        self.logger = logger.StrictLogger()
        self.logger.log_data = True
        self.logger.log_level = self.logger.DEBUG
        self.uo_id = "combo_uo"
        self.gui_title = "Parent UO"
        self.t = np.linspace(0, 100, 1000)
        self.child_uo_list = [MockUpUnitOperation(self.t, "uo_1", "Unit Operation 1"),
                              MockUpUnitOperation(self.t, "uo_2", "Unit Operation 2"),
                              MockUpUnitOperation(self.t, "uo_3", "Unit Operation 3")]
        self.child_uo_list[0].add_to_c = 0.1
        self.child_uo_list[1].add_to_c = 0.01
        self.child_uo_list[2].add_to_c = 0.001

        self.uo_combo = special_uo.ComboUO(self.t, self.child_uo_list, self.uo_id, self.gui_title)

    def test_init(self):
        # non-empty list
        with self.assertRaises(AssertionError):
            special_uo.ComboUO(self.t, [], self.uo_id)
        # unique `uo_id`s
        self.child_uo_list[1].uo_id = "uo_1"
        with self.assertRaises(AssertionError):
            special_uo.ComboUO(self.t, self.child_uo_list, self.uo_id)
        self.child_uo_list[1].uo_id = "uo_2"
        self.uo_id = "uo_1"
        with self.assertRaises(AssertionError):
            special_uo.ComboUO(self.t, self.child_uo_list, self.uo_id)
        self.uo_id = "combo_uo"
        # same time vector
        self.child_uo_list[1]._t = self.child_uo_list[1]._t[:-10]
        with self.assertRaises(AssertionError):
            special_uo.ComboUO(self.t, self.child_uo_list, self.uo_id)
        self.child_uo_list[1]._t = self.child_uo_list[0]._t.copy()

        # asset id and title
        self.assertEqual(self.uo_id, self.uo_combo.uo_id)
        self.assertEqual(self.gui_title, self.uo_combo.gui_title)

        # asset child uo list
        self.assertEqual(self.child_uo_list, self.uo_combo.sub_uo_list)

    def test_logger(self):
        def assert_shared_logger():
            for uo in self.child_uo_list:
                self.assertEqual(self.uo_combo.log, uo.log)

        # assert default logger
        self.assertTrue(isinstance(self.uo_combo.log, logger.DefaultLogger))
        # assert util logger
        assert_shared_logger()

        # update logger
        self.uo_combo.log = self.logger
        # assert strict logger
        self.assertTrue(isinstance(self.uo_combo.log, logger.StrictLogger))
        assert_shared_logger()
        # assert util logger
        assert_shared_logger()

    def test_calculate(self):
        # make sure the function body is empty

        def empty_func():  # pragma: no cover
            """ This method has no 'flow-processing' logic. """
            pass

        # noinspection PyUnresolvedReferences
        self.assertEqual(
            self.uo_combo._calculate.__code__.co_code,
            empty_func.__code__.co_code
        )

    def test_evaluate(self):
        # at init there are only empty arrays
        self.assertTrue(self.uo_combo.get_result()[1].size == 0)
        self.assertTrue(self.child_uo_list[0].get_result()[1].size == 0)
        self.assertTrue(self.child_uo_list[1].get_result()[1].size == 0)
        self.assertTrue(self.child_uo_list[2].get_result()[1].size == 0)

        f_in = np.ones_like(self.uo_combo._t)
        c_in = np.ones([2, self.uo_combo._t.size])
        c0 = c_in[0][0]

        f_out, c_out = self.uo_combo.evaluate(f_in, c_in)

        # no not touch input vector
        self.assertEqual(c0, c_in[0][0])
        # output is the same as output at last uo
        np.testing.assert_array_equal(c_out, self.child_uo_list[2].get_result()[1])
        np.testing.assert_array_equal(c_out, self.uo_combo.get_result()[1])
        # check proper order and values
        self.assertAlmostEqual(c0 + 0.1, self.child_uo_list[0].get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 0.11, self.child_uo_list[1].get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 0.111, self.child_uo_list[2].get_result()[1][0][0])
















