import sys
import io
import unittest
from collections import OrderedDict
from typing import Sequence

import numpy as np
from bio_rtd import core, peak_shapes, utils
from bio_rtd.logger import DefaultLogger, StrictLogger

from bio_rtd_test.aux_bio_rtd_test import TestLogger, EmptyLogger


class TestDefaultLoggerLogic(unittest.TestCase):

    def test_default_logger_logic(self):
        # init
        parent_id = "Parent"
        dll = core.DefaultLoggerLogic(parent_id)

        # default
        self.assertTrue(isinstance(dll.log, DefaultLogger))

        # custom
        custom_logger = StrictLogger()
        dll.log = custom_logger
        self.assertTrue(parent_id in custom_logger.get_entire_data_tree().keys())
        with self.assertRaises(RuntimeError, msg=(parent_id + ": Hey ya all!")):
            dll.log.w("Hey ya all!")

        # custom with uo id
        uo_id = "MyUO"
        custom_logger = DefaultLogger()
        dll.set_logger_from_parent(uo_id, custom_logger)
        combo_id = uo_id + "/" + parent_id
        self.assertTrue(combo_id in custom_logger.get_entire_data_tree().keys())
        with self.assertRaises(RuntimeError, msg=(combo_id + ": Hey ya all!")):
            dll.log.e("Hey ya all!")


class MockUpInlet(core.Inlet):

    def __init__(self, t, f, c, species_list, inlet_id: str, gui_title: str):
        super().__init__(t, species_list, inlet_id, gui_title)

        self._f_out = f.copy()
        self._c_out = c.copy()

    def refresh(self):
        self._f_out += 1
        self._c_out += 1.5


class TestInlet(unittest.TestCase):

    def test_inlet(self):
        t = np.linspace(0, 10, 100)
        f = np.random.rand(t.size)
        species = ['c1', 'c2']
        c = np.random.rand(len(species), t.size)

        # ensure proper t
        with self.assertRaises(AssertionError, msg="t should start with 0"):
            MockUpInlet(t[3:], f, c, species, "", "")
        with self.assertRaises(AssertionError, msg="t should be a 1D np.ndarray"):
            MockUpInlet(t[:, np.newaxis], f, c, species, "", "")
        with self.assertRaises(AssertionError, msg="t should have a fixed step size"):
            MockUpInlet(t[[0, 2, 3, 4, 5]], f, c, species, "", "")

        # ensure proper assignment
        inlet_id = "DummyInlet"
        gui_title = "ABC Inlet test profile"
        inlet = MockUpInlet(t, f, c, species, inlet_id, gui_title)
        # vectors
        np.testing.assert_array_almost_equal(inlet._t, t)
        # species
        self.assertEqual(inlet.species_list, species)
        self.assertEqual(inlet._n_species, len(species))
        # strings
        self.assertEqual(inlet.uo_id, inlet_id)
        self.assertEqual(inlet.gui_title, gui_title)
        # placeholders
        self.assertTrue(len(inlet.adj_par_list) == 0)

        # getters
        np.testing.assert_array_equal(inlet._t, inlet.get_t())
        self.assertEqual(inlet._n_species, inlet.get_n_species())

        # refresh method
        inlet.refresh()
        np.testing.assert_array_almost_equal(inlet._f_out, f + 1)
        np.testing.assert_array_equal(inlet._f_out, inlet.get_result()[0])
        np.testing.assert_array_almost_equal(inlet._c_out, c + 1.5)
        np.testing.assert_array_equal(inlet._c_out, inlet.get_result()[1])

        # logger
        self.assertTrue(isinstance(inlet.log, DefaultLogger))  # default
        custom_logger = StrictLogger()  # custom
        inlet.log = custom_logger
        self.assertTrue(inlet_id in custom_logger.get_entire_data_tree().keys())
        with self.assertRaises(RuntimeError, msg=(inlet_id + ": Hey ya all!")):
            inlet.log.w("Hey ya all!")


class MockUpUnitOperation(core.UnitOperation):
    set_to_constant_inlet = False
    add_to_c = 0

    def _calculate(self):
        if self.set_to_constant_inlet:
            self._c = self._c * 0 + self._c.max()
            self._f = self._f * 0 + self._f.max()
        self._c += self.add_to_c


class TestUnitOperation(unittest.TestCase):

    def set_periodic_flow(self,
                          f_on=1,
                          n_periods=10,
                          rel_on_time=0.2,
                          clip_last_at_end_of_t=False,
                          delay_start=False) -> (int, int, float):
        dt = self.t[1] - self.t[0]
        t_cycle_duration = self.t[-1] / (n_periods +
                                         clip_last_at_end_of_t * rel_on_time / 2 +
                                         delay_start * (1 - rel_on_time) / 2)
        i_flow_on_duration = int(t_cycle_duration * rel_on_time / dt)
        self.uo._f = np.zeros_like(self.t)
        i_flow_on_start_list = [int(v) for v in np.arange(
            delay_start * (1 - rel_on_time) / 2 * t_cycle_duration,
            self.t[-1],
            t_cycle_duration
        ) / dt]
        for i_start in i_flow_on_start_list:
            self.uo._f[i_start:min(i_start + i_flow_on_duration, self.t.size - 1)] = f_on

        return i_flow_on_start_list, i_flow_on_duration, t_cycle_duration

    def generate_periodic_peaks(self,
                                i_flow_on_start_list,
                                i_flow_on_duration,
                                add_gradient_slope=0.) -> (np.ndarray, Sequence[float]):
        """
        Returns
        -------
        (np.ndarray, Sequence[float])
            c : np.ndarray
                Concentration profile for single specie (1D).
            mean_c_list : Sequence[float]
                Mean concentration peak with max mean concentration (apart from last peak if clipped)..
        """
        dt = self.t[1] - self.t[0]
        rt_mean = i_flow_on_duration / 2 * dt
        sigma = rt_mean / 5
        peak = peak_shapes.emg(self.t[:i_flow_on_duration], rt_mean=rt_mean, sigma=sigma, skew=0.4)

        c = np.zeros_like(self.uo._f)
        mean_c_list = []
        for i_start in i_flow_on_start_list:
            i_duration = min(i_flow_on_duration, self.t.size - 1 - i_start)
            if add_gradient_slope >= 0:
                add_on = add_gradient_slope * (i_start + i_flow_on_duration / 2) * dt
            else:
                add_on = add_gradient_slope * (self.t.size - 1 - i_start + i_flow_on_duration / 2) * dt
            p = peak[:i_duration] + add_on
            p[p < 0] = 0
            c[i_start:i_start + i_duration] = p
            if i_duration == i_flow_on_duration:
                mean_c_list.append(p.mean())

        return c, mean_c_list

    def generate_single_peak(self, rt_mean, sigma, skew) -> np.ndarray:
        """
        Returns
        -------
        c : np.ndarray
            Concentration profile for single specie (1D).
        """
        return peak_shapes.emg(self.t, rt_mean=rt_mean, sigma=sigma, skew=skew)

    def setUp(self) -> None:
        self.t = np.linspace(0, 200, 2000)
        self.dt = self.t[1] - self.t[0]
        self.uo = MockUpUnitOperation(t=self.t, uo_id="TestUO")
        self.log = TestLogger()
        self.uo.log = self.log

    def testLogger(self):
        # normal
        with self.assertWarns(Warning):
            self.uo.log.w("Test Warning")
        # other logger
        self.uo.log = EmptyLogger()
        self.uo.log.w("Do not show this warning")
        # blank logger
        self.uo._logger = None
        with self.assertRaises(RuntimeError):
            self.uo.log.e("Default Logger should raise this error.")
        # normal
        self.uo.log = self.log
        with self.assertWarns(Warning):
            self.uo.log.w("Test Warning")

    def test_assert_valid_species_vector(self):
        self.uo._n_species = 1
        self.uo._assert_valid_species_list([0])
        with self.assertRaises(AssertionError):
            self.uo._assert_valid_species_list([1])
        with self.assertRaises(AssertionError):
            self.uo._assert_valid_species_list([-1])
        with self.assertWarns(Warning):
            self.uo._assert_valid_species_list([])
        self.uo._n_species = 3
        self.uo._assert_valid_species_list([0, 2])
        self.uo._assert_valid_species_list([1])
        with self.assertRaises(AssertionError):
            self.uo._assert_valid_species_list([1, 3])
        with self.assertRaises(AssertionError):
            self.uo._assert_valid_species_list([-1, 0])
        with self.assertRaises(AssertionError):
            self.uo._assert_valid_species_list([2, 0])
        with self.assertWarns(Warning):
            self.uo._assert_valid_species_list([])

    def test_is_flow_box_shaped(self):
        # lin gradient (false)
        self.uo._f = self.t
        self.assertFalse(self.uo._is_flow_box_shaped())
        # ones (true)
        self.uo._f = np.ones_like(self.t)
        self.assertTrue(self.uo._is_flow_box_shaped())
        # zeros (false + warning)
        self.uo._f = np.zeros_like(self.t)
        with self.assertWarns(Warning):
            self.assertFalse(self.uo._is_flow_box_shaped())
        # init delay (true)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:200] = 0
        self.assertTrue(self.uo._is_flow_box_shaped())
        # early stop (true)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[-200:] = 0
        self.assertTrue(self.uo._is_flow_box_shaped())
        # negative values (assertion error)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[-200:] = -1
        with self.assertRaises(AssertionError):
            self.uo._is_flow_box_shaped()
        # positive spike (false)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[200] = 1.5
        self.assertFalse(self.uo._is_flow_box_shaped())
        # negative step (false)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:200] = 1.5
        self.assertFalse(self.uo._is_flow_box_shaped())
        # negative spike (false)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[200] = 0.5
        self.assertFalse(self.uo._is_flow_box_shaped())
        # zero in between (false)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[1:200] = 0
        self.assertFalse(self.uo._is_flow_box_shaped())

    def test_i_flow_on(self):
        # ones -> [0]
        self.uo._f = np.ones_like(self.t)
        self.assertEqual(self.uo._i_flow_on(), [0])
        # zeros -> [] + Warning
        self.uo._f = np.zeros_like(self.t)
        with self.assertWarns(Warning):
            self.assertEqual(self.uo._i_flow_on(), [])
        # ones + first zero -> [1]
        self.uo._f = np.ones_like(self.t)
        self.uo._f[0] = 0
        self.assertEqual(self.uo._i_flow_on(), [1])
        # ones + first 100 zero -> [100]
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:100] = 0
        self.assertEqual(self.uo._i_flow_on(), [100])
        # ones + first 100 zero + 500 + last == zero -> [100, 501, self.uo._f.size - 1]
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:100] = 0
        self.uo._f[500] = 0
        self.uo._f[-1] = 0
        self.assertEqual(self.uo._i_flow_on(), [100, 501])
        # ones + first 100 zero + 500 + one before last == zero -> [100, 501, self.uo._f.size - 1]
        self.uo._f[-1] = 1
        self.uo._f[-2] = 0
        self.assertEqual(self.uo._i_flow_on(), [100, 501, self.uo._f.size - 1])

    def test_assert_periodic_flow(self):
        # ones -> Error
        self.uo._f = np.ones_like(self.t)
        with self.assertRaises(AssertionError):
            self.uo._assert_periodic_flow()
        # zeros -> Error
        self.uo._f = np.zeros_like(self.t)
        with self.assertWarns(Warning):
            with self.assertRaises(AssertionError):
                self.uo._assert_periodic_flow()
        # ones + first zero -> Error
        self.uo._f = np.ones_like(self.t)
        self.uo._f[0] = 0
        with self.assertRaises(AssertionError):
            self.uo._assert_periodic_flow()
        # ones + last 100 zero -> [100]
        self.uo._f = np.ones_like(self.t)
        self.uo._f[-100:] = 0
        with self.assertRaises(AssertionError):
            self.uo._assert_periodic_flow()
        # ones + first 100 zero + 500 + last == zero -> Error
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:100] = 0
        self.uo._f[500] = 0
        self.uo._f[-1] = 0
        with self.assertRaises(AssertionError):
            self.uo._assert_periodic_flow()
        # periodic flow
        pf_target = self.set_periodic_flow()
        pf = self.uo._assert_periodic_flow()
        self.assertTrue(pf[0], pf_target[0])
        self.assertTrue(pf[1], pf_target[1])
        self.assertTrue(pf[2], pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(clip_last_at_end_of_t=True)
        pf = self.uo._assert_periodic_flow()
        self.assertTrue(pf[0], pf_target[0])
        self.assertTrue(pf[1], pf_target[1])
        self.assertTrue(pf[2], pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(delay_start=True)
        pf = self.uo._assert_periodic_flow()
        self.assertTrue(pf[0], pf_target[0])
        self.assertTrue(pf[1], pf_target[1])
        self.assertTrue(pf[2], pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(delay_start=True, clip_last_at_end_of_t=True)
        pf = self.uo._assert_periodic_flow()
        self.assertTrue(pf[0], pf_target[0])
        self.assertTrue(pf[1], pf_target[1])
        self.assertTrue(pf[2], pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(
            delay_start=True, clip_last_at_end_of_t=True,
            n_periods=2, rel_on_time=0.5)
        pf = self.uo._assert_periodic_flow()
        self.assertTrue(pf[0], pf_target[0])
        self.assertTrue(pf[1], pf_target[1])
        self.assertTrue(pf[2], pf_target[2])

    def test_estimate_steady_state_mean_f(self):
        # ones -> Error
        # it relies on `_estimate_steady_state_mean_f`
        self.uo._f = np.ones_like(self.t)
        with self.assertRaises(AssertionError):
            self.uo._estimate_steady_state_mean_f()
        # periodic flow
        rel_on_time = 0.2
        pf_target = self.set_periodic_flow(rel_on_time=rel_on_time)
        f_ss, t_cycle_duration = self.uo._estimate_steady_state_mean_f()
        self.assertTrue(f_ss, self.uo._f.max() * rel_on_time)
        self.assertTrue(t_cycle_duration, pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(clip_last_at_end_of_t=True, rel_on_time=rel_on_time)
        f_ss, t_cycle_duration = self.uo._estimate_steady_state_mean_f()
        self.assertTrue(f_ss, self.uo._f.max() * rel_on_time)
        self.assertTrue(t_cycle_duration, pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(delay_start=True, rel_on_time=rel_on_time)
        f_ss, t_cycle_duration = self.uo._estimate_steady_state_mean_f()
        self.assertTrue(f_ss, self.uo._f.max() * rel_on_time)
        self.assertTrue(t_cycle_duration, pf_target[2])
        # periodic flow
        pf_target = self.set_periodic_flow(delay_start=True,
                                           clip_last_at_end_of_t=True,
                                           rel_on_time=rel_on_time)
        f_ss, t_cycle_duration = self.uo._estimate_steady_state_mean_f()
        self.assertTrue(f_ss, self.uo._f.max() * rel_on_time)
        self.assertTrue(t_cycle_duration, pf_target[2])
        # periodic flow
        rel_on_time = 0.5
        pf_target = self.set_periodic_flow(
            delay_start=True, clip_last_at_end_of_t=True,
            n_periods=2, rel_on_time=rel_on_time)
        f_ss, t_cycle_duration = self.uo._estimate_steady_state_mean_f()
        self.assertTrue(f_ss, self.uo._f.max() * rel_on_time)
        self.assertTrue(t_cycle_duration, pf_target[2])
        with self.assertWarns(Warning):
            self.set_periodic_flow(delay_start=True, clip_last_at_end_of_t=True,
                                   n_periods=400, rel_on_time=rel_on_time)
            self.uo._estimate_steady_state_mean_f()

    def test_estimate_steady_state_mean_c(self):
        # it calls `_assert_valid_species_vector`, so just do 1 case for species
        self.uo._n_species = 1
        with self.assertRaises(AssertionError):
            self.uo._estimate_steady_state_mean_c([1])
        # zero -> warning
        self.uo._c = np.zeros([1, self.t.size])
        with self.assertWarns(Warning):
            np.testing.assert_array_equal(
                self.uo._estimate_steady_state_mean_c([0]),
                np.zeros([1, 1])
            )
        # it calls `_is_flow_box_shaped`, so just do 1 case for wrong flow
        self.uo._c = np.ones([1, self.t.size])
        self.uo._f = np.ones_like(self.t)
        self.uo._f[-200:] = -1
        with self.assertRaises(AssertionError):
            self.uo._estimate_steady_state_mean_c([0])
        # ============ Single Specie ==========

        # single peak
        peak = self.generate_single_peak(self.t[-1] / 2, self.t[-1] / 8, 0.5)
        self.uo._c = peak[np.newaxis, :]
        self.uo._f = np.ones_like(self.t)
        c_mean_target = np.mean(peak[peak >= peak.max() * 0.9])
        with self.assertWarns(Warning):
            c_mean = self.uo._estimate_steady_state_mean_c([0])
        np.testing.assert_array_equal(c_mean, np.array([c_mean_target])[:, np.newaxis])

        # multi peak
        rel_on_time = 0.2
        i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = self.set_periodic_flow(rel_on_time=rel_on_time)
        c, max_c_mean = self.generate_periodic_peaks(
            i_flow_on_start_list=i_flow_on_start_list,
            i_flow_on_duration=i_flow_on_duration
        )
        self.uo._c = c[np.newaxis, :]
        with self.assertWarns(Warning):
            c_mean = self.uo._estimate_steady_state_mean_c()
        np.testing.assert_array_equal(c_mean, np.array([[max(max_c_mean)]]))

        # multi peak # 2
        rel_on_time = 0.2
        i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = self.set_periodic_flow(
            rel_on_time=rel_on_time,
            delay_start=True
        )
        c, max_c_mean = self.generate_periodic_peaks(
            i_flow_on_start_list=i_flow_on_start_list,
            i_flow_on_duration=i_flow_on_duration,
            add_gradient_slope=-0.002,
        )
        self.uo._c = c[np.newaxis, :]
        with self.assertWarns(Warning):
            c_mean = self.uo._estimate_steady_state_mean_c([0])
        np.testing.assert_array_equal(c_mean, np.array([[max(max_c_mean)]]))

        # multi peak # 3
        rel_on_time = 0.5
        i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = self.set_periodic_flow(
            rel_on_time=rel_on_time,
            clip_last_at_end_of_t=True
        )
        c, max_c_mean = self.generate_periodic_peaks(
            i_flow_on_start_list=i_flow_on_start_list,
            i_flow_on_duration=i_flow_on_duration,
            add_gradient_slope=0.2,
        )
        self.uo._c = c[np.newaxis, :]
        with self.assertWarns(Warning):
            c_mean = self.uo._estimate_steady_state_mean_c([0])
        np.testing.assert_array_equal(c_mean, np.array([[max(max_c_mean)]]))

        # ============ Multi Species ==========
        self.uo._n_species = 3

        # single peak
        peak = self.generate_single_peak(self.t[-1] / 2, self.t[-1] / 8, 0.5)
        peak2 = self.generate_single_peak(self.t[-1] * 3 / 4, self.t[-1] / 8, 0.25)
        self.uo._c = np.stack((peak, np.zeros_like(self.t), peak2), axis=0)
        self.uo._f = np.ones_like(self.t)
        self.uo._f[:300] = 0
        self.uo._f[-30:] = 0
        for species in [[0, 1], [0, 2], [0], [2]]:
            c_sum = self.uo._c[species].sum(0)
            c_sum[self.uo._f == 0] = 0
            i_st, i_end = utils.vectors.true_start_and_end(c_sum >= c_sum.max() * 0.9)
            c_mean_target = np.mean(self.uo._c[species, i_st:i_end], 1)[:, np.newaxis]
            with self.assertWarns(Warning):
                c_mean = self.uo._estimate_steady_state_mean_c(species)
            np.testing.assert_array_almost_equal(c_mean, c_mean_target, 10)

        # multi peak
        for delay_start in [True, False]:
            for clip_last_at_end_of_t in [True, False]:
                for rel_on_time in [0.2, 0.8, 0.5]:
                    i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = self.set_periodic_flow(
                        rel_on_time=rel_on_time, delay_start=delay_start, clip_last_at_end_of_t=clip_last_at_end_of_t
                    )
                    c1, max_c_mean_1 = self.generate_periodic_peaks(
                        i_flow_on_start_list=i_flow_on_start_list,
                        i_flow_on_duration=i_flow_on_duration,
                        add_gradient_slope=0.2,
                    )
                    c2, max_c_mean_2 = self.generate_periodic_peaks(
                        i_flow_on_start_list=i_flow_on_start_list,
                        i_flow_on_duration=i_flow_on_duration,
                        add_gradient_slope=-0.002,
                    )
                    self.uo._c = np.stack((c1, np.zeros_like(self.t), c2), axis=0)
                    max_c_mean_all = np.stack((max_c_mean_1, np.zeros_like(max_c_mean_1), max_c_mean_2), axis=0)
                    for species in [[0, 1], [0, 2], [0], [2]]:
                        c_sum = max_c_mean_all[species].sum(0)
                        i_target = np.argmax(c_sum)
                        c_mean_target = max_c_mean_all[species, i_target].reshape(len(species), 1)
                        with self.assertWarns(Warning):
                            c_mean = self.uo._estimate_steady_state_mean_c(species)
                        np.testing.assert_array_almost_equal(c_mean, c_mean_target, 10)

    def test_ensure_single_non_negative_parameter(self):
        with self.assertWarns(Warning):
            self.uo._ensure_single_non_negative_parameter(
                log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
                par1=-1, par2=-1
            )
        with self.assertRaises(RuntimeError):
            self.uo._ensure_single_non_negative_parameter(
                log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
                par1=100, par2=100
            )
        self.uo._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            par1=-1, par2=100
        )
        self.uo._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            par1=-1, par2=0
        )

    def reset_cut_start_data(self):
        # reset peak cut values
        self.uo.discard_inlet_until_t = -1
        self.uo.discard_inlet_n_cycles = -1
        self.uo.discard_inlet_until_min_c = np.array([])
        self.uo.discard_inlet_until_min_c_rel = np.array([])
        self.uo.discard_outlet_until_t = -1
        self.uo.discard_outlet_n_cycles = -1
        self.uo.discard_outlet_until_min_c = np.array([])
        self.uo.discard_outlet_until_min_c_rel = np.array([])
        # reset _c and _f
        self.uo._c = self._peak_cut_source_c.copy()
        self.uo._f = self._peak_cut_source_flow.copy()

    def run_cut_start_box_shaped(self):
        for outlet in [True, False]:
            # reset
            self.reset_cut_start_data()
            peak = self._peak_cut_source_c.copy()
            flow = self._peak_cut_source_flow.copy()
            # nothing should happen
            self.uo._cut_start_of_c_and_f(outlet=outlet)
            np.testing.assert_array_equal(self.uo._c, peak)
            np.testing.assert_array_equal(self.uo._f, flow)
            # nothing should happen
            self.uo.discard_inlet_until_min_c = np.array([0])
            self.uo.discard_outlet_until_min_c = np.array([0])
            self.uo._cut_start_of_c_and_f(outlet=outlet)
            np.testing.assert_array_equal(self.uo._c, peak)
            np.testing.assert_array_equal(self.uo._f, flow)

            # reset
            self.reset_cut_start_data()
            peak = self._peak_cut_source_c.copy()
            flow = self._peak_cut_source_flow.copy()
            # concentration
            c_lim = peak.max(1)[:, np.newaxis] * 0.7
            if outlet:
                self.uo.discard_outlet_until_min_c = np.array([c_lim])
            else:
                self.uo.discard_inlet_until_min_c = np.array([c_lim])
            b_lim = np.all(peak >= c_lim, 0) * (flow > 0)
            empty = not np.any(b_lim)
            if empty:
                flow[:] = 0
                peak[:] = 0
                # test
                with self.assertWarns(Warning):
                    self.uo._cut_start_of_c_and_f(outlet=outlet)
                    np.testing.assert_array_equal(self.uo._f, flow)
                    np.testing.assert_array_equal(self.uo._c, peak)
            else:
                i_lim = utils.vectors.true_start(np.all(peak >= c_lim, 0) * (flow > 0))
                flow[:i_lim] = 0
                peak[:, :i_lim] = 0
                # test
                self.uo._cut_start_of_c_and_f(outlet=outlet)
                np.testing.assert_array_equal(self.uo._f, flow)
                np.testing.assert_array_equal(self.uo._c, peak)
            # reset
            self.reset_cut_start_data()
            # relative concentration
            c_rel_lim = 0.7 * np.ones([self.uo._n_species, 1])
            if outlet:
                self.uo.discard_outlet_until_min_c_rel = c_rel_lim
            else:
                self.uo.discard_inlet_until_min_c_rel = c_rel_lim
            # test (should be the same as above)
            if empty:
                with self.assertWarns(Warning):
                    self.uo._cut_start_of_c_and_f(outlet=outlet)
                    np.testing.assert_array_equal(self.uo._f, flow)
                    np.testing.assert_array_equal(self.uo._c, peak)
            else:
                self.uo._cut_start_of_c_and_f(outlet=outlet)
                np.testing.assert_array_equal(self.uo._f, flow)
                np.testing.assert_array_equal(self.uo._c, peak)

            # reset
            self.reset_cut_start_data()
            peak = self._peak_cut_source_c.copy()
            flow = self._peak_cut_source_flow.copy()
            # time
            t_lim = self.t[-1] / 10
            i_lim = int(t_lim / self.dt)
            flow[:i_lim] = 0
            peak[:, :i_lim] = 0
            if outlet:
                self.uo.discard_outlet_until_t = t_lim
            else:
                self.uo.discard_inlet_until_t = t_lim
            # test
            self.uo._cut_start_of_c_and_f(outlet=outlet)
            np.testing.assert_array_equal(self.uo._c, peak)
            np.testing.assert_array_equal(self.uo._f, flow)

            # reset
            self.reset_cut_start_data()
            peak = self._peak_cut_source_c.copy()
            flow = self._peak_cut_source_flow.copy()
            # cycle
            if outlet:
                self.uo.discard_outlet_n_cycles = 1
            else:
                self.uo.discard_inlet_n_cycles = 1
            peak *= 0
            flow *= 0
            # assert (in box shaped no cycles)
            with self.assertRaises(AssertionError):
                self.uo._cut_start_of_c_and_f(outlet=outlet)

    def test_cut_start_box_shaped(self):
        # -------- single component - box shaped -----------
        # set number of species (n_species) and prepare peak (c) and flow (f) profile
        self.uo._n_species = 1
        self._peak_cut_source_c = self.generate_single_peak(self.t[-1] / 2, self.t[-1] / 8, 0.5)[np.newaxis, :]
        self._peak_cut_source_flow = np.ones_like(self.t)
        self._peak_cut_source_flow[:200] = 0

        # run tests
        self.run_cut_start_box_shaped()

        # -------- multi component - box shaped -----------
        # set number of species (n_species) and prepare peak (c) and flow (f) profile
        self.uo._n_species = 3
        peak = self.generate_single_peak(self.t[-1] / 2, self.t[-1] / 8, 0.5)
        peak2 = self.generate_single_peak(self.t[-1] * 3 / 4, self.t[-1] / 8, 0.25)
        self._peak_cut_source_c = np.stack((peak, np.zeros_like(self.t), peak2), axis=0)
        self._peak_cut_source_flow = np.ones_like(self.t)
        self._peak_cut_source_flow[:300] = 0
        self._peak_cut_source_flow[-30:] = 0

        # run tests
        self.run_cut_start_box_shaped()

        # no longer need to test inlet and outlet interchangeability (it was tested in box shaped)

    def run_cut_start_periodic(self):
        # util methods
        i_start_list = self._peak_cut_source_i_flow_on_start_list
        i_duration = self._peak_cut_source_i_flow_on_duration

        # get start of cycle which includes i
        def get_cycle_start(i):
            for i_start in i_start_list:
                if i <= i_start + i_duration:
                    return i_start

        def validate(_flow, _peak):
            if _flow.max() == 0:
                with self.assertWarns(Warning):
                    self.uo._cut_start_of_c_and_f(outlet=False)
                    np.testing.assert_array_equal(self.uo._f, _flow)
                    np.testing.assert_array_equal(self.uo._c, _peak * 0)
            else:
                self.uo._cut_start_of_c_and_f(outlet=False)
                np.testing.assert_array_equal(self.uo._f, _flow)
                np.testing.assert_array_equal(self.uo._c, _peak)

        # reset
        self.reset_cut_start_data()
        peak = self._peak_cut_source_c.copy()
        flow = self._peak_cut_source_flow.copy()
        # nothing should happen
        validate(flow, peak)
        # nothing should happen
        self.uo.discard_inlet_until_min_c = np.array([0])
        self.uo.discard_outlet_until_min_c = np.array([0])
        validate(flow, peak)

        # reset
        self.reset_cut_start_data()
        peak = self._peak_cut_source_c.copy()
        flow = self._peak_cut_source_flow.copy()
        # concentration
        c_lim = peak.max(1)[:, np.newaxis] * 0.7
        self.uo.discard_inlet_until_min_c = np.array([c_lim])
        b_lim = np.all(peak >= c_lim, 0) * (flow > 0)

        empty = not np.any(b_lim)
        if empty:
            flow[:] = 0
            peak[:] = 0
            # test
            with self.assertWarns(Warning):
                self.uo._cut_start_of_c_and_f(outlet=False)
                np.testing.assert_array_equal(self.uo._f, flow)
                np.testing.assert_array_equal(self.uo._c, peak)
        else:
            i_lim = utils.vectors.true_start(np.all(peak >= c_lim, 0) * (flow > 0))
            i_cycle_lim = get_cycle_start(i_lim)
            flow[:i_cycle_lim] = 0
            peak[:, :i_cycle_lim] = 0
            # test
            self.uo._cut_start_of_c_and_f(outlet=False)
            np.testing.assert_array_equal(self.uo._f, flow)
            np.testing.assert_array_equal(self.uo._c, peak)
        # reset
        self.reset_cut_start_data()
        # relative concentration
        c_rel_lim = 0.7 * np.ones([self.uo._n_species, 1])
        self.uo.discard_inlet_until_min_c_rel = c_rel_lim
        # test (should be the same as above)
        validate(flow, peak)

        # reset
        self.reset_cut_start_data()
        peak = self._peak_cut_source_c.copy()
        flow = self._peak_cut_source_flow.copy()
        # time
        t_lim = self.t[-1] / 10
        i_lim = int(t_lim / self.dt)
        i_cycle_lim = get_cycle_start(i_lim)
        flow[:i_cycle_lim] = 0
        peak[:, :i_cycle_lim] = 0
        self.uo.discard_inlet_until_t = t_lim
        # test
        validate(flow, peak)

        # reset
        self.reset_cut_start_data()
        peak = self._peak_cut_source_c.copy()
        flow = self._peak_cut_source_flow.copy()
        # cycle
        self.uo.discard_inlet_n_cycles = 2
        i_cycle_lim = i_start_list[2]
        flow[:i_cycle_lim] = 0
        peak[:, :i_cycle_lim] = 0
        # assert (in box shaped no cycles)
        validate(flow, peak)

        # reset
        self.reset_cut_start_data()
        peak = self._peak_cut_source_c.copy()
        flow = self._peak_cut_source_flow.copy()
        # cycle
        self.uo.discard_inlet_n_cycles = 20
        flow[:] = 0
        peak[:] = 0
        # assert (in box shaped no cycles)
        with self.assertWarns(Warning):
            self.uo._cut_start_of_c_and_f(outlet=False)

    def test_cut_start_periodic(self):
        # -------- periodic -----------
        for delay_start in [True, False]:
            for clip_last_at_end_of_t in [True, False]:
                for rel_on_time in [0.2, 0.8, 0.5]:
                    # flow
                    i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = self.set_periodic_flow(
                        rel_on_time=rel_on_time,
                        delay_start=delay_start,
                        clip_last_at_end_of_t=clip_last_at_end_of_t
                    )
                    self._peak_cut_source_flow = self.uo._f.copy()

                    # -------- single component -----------
                    # concentration
                    self.uo._n_species = 1
                    c, max_c_mean = self.generate_periodic_peaks(
                        i_flow_on_start_list=i_flow_on_start_list,
                        i_flow_on_duration=i_flow_on_duration
                    )
                    self._peak_cut_source_c = c[np.newaxis, :].copy()
                    self._peak_cut_source_i_flow_on_duration = i_flow_on_duration
                    self._peak_cut_source_i_flow_on_start_list = i_flow_on_start_list

                    # run tests
                    self.run_cut_start_periodic()

                    # -------- multi component -----------
                    # concentration
                    self.uo._n_species = 3
                    c1, max_c_mean_1 = self.generate_periodic_peaks(
                        i_flow_on_start_list=i_flow_on_start_list,
                        i_flow_on_duration=i_flow_on_duration,
                        add_gradient_slope=0.2,
                    )
                    c2, max_c_mean_2 = self.generate_periodic_peaks(
                        i_flow_on_start_list=i_flow_on_start_list,
                        i_flow_on_duration=i_flow_on_duration,
                        add_gradient_slope=-0.002,
                    )
                    self._peak_cut_source_c = np.stack((c1, np.zeros_like(self.t), c2), axis=0)
                    self._peak_cut_source_flow = self.uo._f.copy()

                    # run tests
                    self.run_cut_start_periodic()

    def test_evaluate_supporting_logic(self):
        f_in = np.ones_like(self.t)
        c_in = np.ones([1, self.t.size])

        # check log reset
        tree = self.log.get_data_tree(self.uo.uo_id)
        tree['test'] = 'sss'
        self.uo.evaluate(f_in, c_in)
        self.assertTrue(len(tree.keys()) == 0)

        # assertions
        with self.assertRaises(AssertionError):  # negative flow
            self.uo.evaluate(f_in - 2, c_in)
        with self.assertRaises(AssertionError):  # negative concentration
            self.uo.evaluate(f_in, c_in - 2)

        # return
        np.testing.assert_array_equal(self.uo._f, self.uo.get_result()[0])
        np.testing.assert_array_equal(self.uo._c, self.uo.get_result()[1])

        # warning
        with self.assertWarns(Warning):  # zero flow
            f_out, c_out = self.uo.evaluate(f_in * 0, c_in)
            np.testing.assert_array_equal(f_out, f_in * 0)
            np.testing.assert_array_equal(c_out, c_in * 0)

        # n_species
        f_in = np.ones_like(self.t)
        c_in = np.ones([1, self.t.size])
        self.uo.evaluate(f_in, c_in)
        self.assertEqual(self.uo._n_species, c_in.shape[0])
        c_in = np.ones([3, self.t.size])
        self.uo.evaluate(f_in, c_in)
        self.assertEqual(self.uo._n_species, c_in.shape[0])

        # cut
        self.uo.set_to_constant_inlet = False
        self.uo.discard_inlet_until_t = self.t[10]
        f_target = f_in.copy()
        f_target[:10] = 0
        c_target = c_in.copy()
        c_target[:, :10] = 0
        f_out, c_out = self.uo.evaluate(f_in, c_in)
        np.testing.assert_array_equal(f_out, f_target)
        np.testing.assert_array_equal(c_out, c_target)
        # mock-up undo cutting
        self.uo.set_to_constant_inlet = True
        f_out, c_out = self.uo.evaluate(f_in, c_in)
        np.testing.assert_array_equal(f_out, f_in)
        np.testing.assert_array_equal(c_out, c_in)
        self.uo.discard_outlet_until_t = self.t[5]
        f_target = f_in.copy()
        f_target[:5] = 0
        c_target = c_in.copy()
        c_target[:, :5] = 0
        f_out, c_out = self.uo.evaluate(f_in, c_in)
        np.testing.assert_array_equal(f_out, f_target)
        np.testing.assert_array_equal(c_out, c_target)


class MockUpParameterSetList(core.ParameterSetList):
    POSSIBLE_KEY_GROUPS = [["key1"], ["key2a", "key2b"]]
    OPTIONAL_KEYS = ["optional_key_i", "optional_key_ii", "optional_key_iii"]


class TestParameterSetList(unittest.TestCase):

    def test_parameter_set_list(self):
        psl = MockUpParameterSetList()

        # test logic
        self.assertEqual(
            {"key1": 1.2},
            psl.assert_and_get_provided_kv_pairs(key1=1.2)
        )
        self.assertEqual(
            {"key2a": 1.2, "key2b": 1},
            psl.assert_and_get_provided_kv_pairs(key2a=1.2, key2b=1)
        )
        # key group is missing
        with self.assertRaises(KeyError):
            psl.assert_and_get_provided_kv_pairs(key2a=1.2, optional_key_ii=1)
        # pick first group as default
        self.assertEqual(
            {"key1": 1.2},
            psl.assert_and_get_provided_kv_pairs(key1=1.2, key2a=1.2, key2b=1)
        )
        self.assertEqual(
            {"key1": 1.2},
            psl.assert_and_get_provided_kv_pairs(key2a=1.2, key2b=1, key1=1.2)
        )
        # optional parameters
        self.assertEqual(
            {"key1": 1.2, "optional_key_i": 1, "optional_key_iii": 1},
            psl.assert_and_get_provided_kv_pairs(key1=1.2, optional_key_i=1, optional_key_iii=1)
        )
        self.assertEqual(
            {"key1": 1.2, "optional_key_i": 1, "optional_key_iii": 1},
            psl.assert_and_get_provided_kv_pairs(key1=1.2, optional_key_iii=1, optional_key_i=1)
        )
        self.assertEqual(
            {"key1": 1.2, "optional_key_ii": 1},
            psl.assert_and_get_provided_kv_pairs(key1=1.2, optional_key_ii=1, optional_key_iv=1)
        )


class MockUpPDF(core.PDF):
    POSSIBLE_KEY_GROUPS = [["key1"], ["key2a", "key2b"]]
    OPTIONAL_KEYS = ["optional_key_i", "optional_key_ii", "optional_key_iii"]

    def _calc_pdf(self, kw_pars: dict) -> np.ndarray:

        if "key1" in kw_pars.keys():
            p = np.ones(10) * kw_pars["key1"]
        else:
            p = np.ones(self._t_steps_max) * kw_pars["key2a"] * kw_pars["key2b"]

        if "optional_key_i" in kw_pars.keys():
            p += kw_pars["optional_key_i"]
        if "optional_key_ii" in kw_pars.keys():
            p += kw_pars["optional_key_ii"]
        if "optional_key_iii" in kw_pars.keys():
            p += kw_pars["optional_key_iii"]

        return p


class TestPDF(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 10, 100)
        self.dt = self.t[1]
        self.t_steps = self.t.size
        self.pdf_id = "mockUp_pdf"
        self.pdf = MockUpPDF(self.t, self.pdf_id)

    def test_init(self):
        # proper t
        with self.assertRaises(AssertionError):
            MockUpPDF(self.t[3:], self.pdf_id)
        with self.assertRaises(AssertionError):
            MockUpPDF(-self.t, self.pdf_id)

        # logger
        self.assertEqual(self.pdf._instance_id, self.pdf_id)
        # dt
        self.assertEqual(self.pdf._dt, self.dt)
        # t steps
        self.assertEqual(self.pdf._t_steps_max, self.t_steps)

        # trim default settings
        self.assertTrue(self.pdf.trim_and_normalize)
        self.assertEqual(self.pdf.cutoff_relative_to_max, 0.0001)

        # placeholder for the result of the pdf calculation
        self.assertTrue(self.pdf._pdf.size == 0)

    def mock_trim_and_normalize(self, p: np.ndarray) -> np.ndarray:
        p_norm = p * (p >= self.pdf.cutoff_relative_to_max * p.max())
        p_norm = p_norm / p_norm.sum() / self.dt
        return p_norm

    def test_apply_cutoff_and_normalize(self):
        p = np.linspace(0, 10, 100)
        self.pdf._pdf = p.copy()
        self.pdf.cutoff_relative_to_max = 0.2
        p_norm = self.mock_trim_and_normalize(p)

        self.pdf._apply_cutoff_and_normalize()
        np.testing.assert_array_almost_equal(self.pdf._pdf, p_norm)
        self.assertTrue(p_norm.sum() * self.dt == 1)

    def test_update_pdf(self):
        def run_test(**kwargs):
            self.pdf.update_pdf(**kwargs)
            d = self.pdf.assert_and_get_provided_kv_pairs(**kwargs)
            ref_pdf = self.pdf._calc_pdf(d)
            if self.pdf.trim_and_normalize:
                ref_pdf = self.mock_trim_and_normalize(ref_pdf)

            np.testing.assert_array_almost_equal(self.pdf.get_p(), ref_pdf)

        run_test(key1=1, key2a=2, key2b=3,
                 optional_key_i=1.1, optional_key_ii=1.2, optional_key_iii=1.3)
        run_test(key2a=2, key2b=3,
                 optional_key_i=1.1, optional_key_iii=1.3)
        run_test(key2a=2, key2b=3,
                 optional_key_i=1.1, optional_key_iv=1.3)
        run_test(key2a=2, key2b=3)
        with self.assertRaises(KeyError):
            run_test(key2a=2)
        with self.assertRaises(KeyError):
            run_test(optional_key_i=1.1, optional_key_ii=1.2, optional_key_iii=1.3)

    def test_get_p(self):
        self.pdf._pdf = np.array([])
        with self.assertRaises(AssertionError, msg="PDF is empty. Make sure `update_pdf` was called before `get_pdf`"):
            self.pdf.get_p()
        self.pdf._pdf = np.ones(5)
        with self.assertRaises(RuntimeError, msg="PDF should have a length of at least 5 time steps"):
            self.pdf.get_p()
        self.pdf._pdf = np.random.rand(13)
        np.testing.assert_array_equal(self.pdf._pdf, self.pdf.get_p())


class MockUpChromatographyLoadBreakthrough(core.ChromatographyLoadBreakthrough):
    POSSIBLE_KEY_GROUPS = [["bound_proportion"]]
    OPTIONAL_KEYS = ["add_to_bound_proportion"]

    bound_proportion = 0.3

    def _update_btc_parameters(self, kw_pars: dict):
        self.bound_proportion = kw_pars.get("bound_proportion")

        if "add_to_bound_proportion" in kw_pars.keys():
            self.bound_proportion += kw_pars["add_to_bound_proportion"]

    def _calc_unbound_to_load_ratio(self, loaded_material: np.ndarray) -> np.ndarray:
        return np.ones_like(loaded_material) * self.bound_proportion

    def get_total_bc(self) -> float:  # pragma: no cover
        pass


class TestChromatographyLoadBreakthrough(unittest.TestCase):

    def setUp(self) -> None:
        self.dt = 0.025
        self.bt_id = "mockUp_bt_profile"
        self.bt = MockUpChromatographyLoadBreakthrough(self.dt, self.bt_id)

    def test_init(self):
        # assert proper dt
        with self.assertRaises(AssertionError):
            MockUpChromatographyLoadBreakthrough(0, self.bt_id)
        with self.assertRaises(AssertionError):
            MockUpChromatographyLoadBreakthrough(-self.dt, self.bt_id)

        # logger
        self.assertEqual(self.bt._instance_id, self.bt_id)

        # dt
        self.assertEqual(self.bt._dt, self.dt)

    def test_update_btc_parameters(self):
        # normal
        self.bt.update_btc_parameters(bound_proportion=0.2)
        self.assertEqual(self.bt.bound_proportion, 0.2)

        # do not apply extra only
        with self.assertRaises(KeyError):
            self.bt.update_btc_parameters(add_to_bound_proportion=0.15)
        self.assertEqual(self.bt.bound_proportion, 0.2)

        # test with extra
        self.bt.update_btc_parameters(bound_proportion=0.2,
                                      add_to_bound_proportion=0.15)
        self.assertEqual(self.bt.bound_proportion, 0.35)

    def test_calc_c_bound(self):
        def run_test(_f, _c):
            # target
            m_cum_sum = np.cumsum((_c * _f[np.newaxis, :]).sum(0)) * self.dt
            unbound_to_load_ratio = self.bt._calc_unbound_to_load_ratio(m_cum_sum)
            c_target = _c * (1 - unbound_to_load_ratio[np.newaxis, :])
            # calc
            c_out = self.bt.calc_c_bound(_f, _c)
            # compare
            np.testing.assert_array_almost_equal(c_out, c_target)

        # normal test
        f = np.ones(100)
        c = np.ones([2, f.size])
        c[1, :] = 0.2
        run_test(f, c)
        # zeros
        c[:] = 0
        self.assertTrue(self.bt.calc_c_bound(f, c).sum() == 0)
        c[:, :20] = 1
        f[:20] = 0
        self.assertTrue(self.bt.calc_c_bound(f, c).sum() == 0)
        c[1, :] = np.linspace(2.5, 3.5, c.shape[1])
        run_test(f, c)


class MockUpUserInterface(core.UserInterface):
    ui_constructed = False
    updated_ui_list = []

    def build_ui(self):
        self.ui_constructed = True

    def _update_ui_for_uo(self, uo_i, f, c):
        self.updated_ui_list.append([uo_i, f[0], c[0][0]])


# RtdModel and UserInterface
class TestRtdModelAndUserInterface(unittest.TestCase):

    def setUp(self) -> None:
        self.logger = DefaultLogger()
        self.logger.log_data = True
        self.logger.log_level = self.logger.DEBUG
        self.title = "model_1"
        self.desc = "Test RTD Model"
        t = np.linspace(0, 100, 1000)
        species = ["c1", "c2"]
        c_init = np.ones([len(species), t.size])
        f_init = np.ones_like(t)
        self.inlet = MockUpInlet(t, f_init, c_init, species, "inlet", "Inlet")
        self.dsp = [MockUpUnitOperation(t, "uo_1", "Unit Operation 1"),
                    MockUpUnitOperation(t, "uo_2", "Unit Operation 2"),
                    MockUpUnitOperation(t, "uo_3", "Unit Operation 3")]
        self.dsp[0].add_to_c = 0.1
        self.dsp[1].add_to_c = 0.01
        self.dsp[2].add_to_c = 0.001

        self.rtd_model = core.RtdModel(self.inlet, self.dsp, self.logger, self.title, self.desc)

        # print to internal output rather than standard input to test print statement
        self.old_stdout = sys.stdout
        sys.stdout = self.print_output = io.StringIO()

    def tearDown(self):
        # restore print
        sys.stdout = self.old_stdout

    def assert_print(self, last_msg_list: Sequence[str]):
        self.assertEqual(self.print_output.getvalue().split('\n')[-len(last_msg_list) - 1:-1], last_msg_list)

    def test_init(self):
        # assert unique ids
        self.dsp[1].uo_id = "uo_1"
        with self.assertRaises(AssertionError):
            core.RtdModel(self.inlet, self.dsp, self.logger)
        self.dsp[1].uo_id = "uo_2"

        # asset title and description
        self.assertEqual(self.title, self.rtd_model.title)
        self.assertEqual(self.desc, self.rtd_model.desc)

        # assert logger
        self.assertEqual(self.logger, self.rtd_model.log)
        self.assertEqual(self.logger, self.inlet.log)
        for uo in self.dsp:
            self.assertEqual(self.logger, uo.log)

        # logger RtdModel tree
        # noinspection PyTypeChecker
        self.assertEqual(
            self.rtd_model.log.get_data_tree(self.title),
            OrderedDict({uo.uo_id: dict() for uo in [self.inlet] + self.dsp})
        )

    def test_get_dsp_uo(self):
        self.assertEqual(self.dsp[1],
                         self.rtd_model.get_dsp_uo(self.dsp[1].uo_id))
        self.assertEqual(self.dsp[0],
                         self.rtd_model.get_dsp_uo(self.dsp[0].uo_id))
        with self.assertRaises(KeyError):
            self.rtd_model.get_dsp_uo("non_existing_uo_id")

    def test_recalculate_callback(self):
        recalculated = []

        def callback_fnc(uo_i: int):
            recalculated.append(uo_i)

        # check callback
        self.rtd_model.recalculate(-1, callback_fnc)
        self.assertEqual(recalculated, [-1, 0, 1, 2])
        self.assert_print(["model_1: Inlet profile updated",
                           "model_1: Unit operation `uo_1` updated",
                           "model_1: Unit operation `uo_2` updated",
                           "model_1: Unit operation `uo_3` updated"])
        recalculated.clear()
        self.rtd_model.recalculate(0, callback_fnc)
        self.assertEqual(recalculated, [0, 1, 2])
        self.assert_print(["model_1: Unit operation `uo_1` updated",
                           "model_1: Unit operation `uo_2` updated",
                           "model_1: Unit operation `uo_3` updated"])
        recalculated.clear()
        self.rtd_model.recalculate(1, callback_fnc)
        self.assertEqual(recalculated, [1, 2])
        self.assert_print(["model_1: Unit operation `uo_2` updated",
                           "model_1: Unit operation `uo_3` updated"])

        # no callback
        recalculated.clear()
        self.rtd_model.recalculate(-1)
        self.assertEqual(recalculated, [])

    def test_recalculate_f_c(self):
        c0 = self.inlet.get_result()[1][0][0]

        self.rtd_model.recalculate(-1)
        self.assertAlmostEqual(c0 + 1.5, self.inlet.get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 1.6, self.dsp[0].get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 1.61, self.dsp[1].get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 1.611, self.dsp[2].get_result()[1][0][0])

        self.dsp[0].add_to_c = 1000
        self.dsp[1].add_to_c = 10
        self.rtd_model.recalculate(1)
        self.assertAlmostEqual(c0 + 11.6, self.dsp[1].get_result()[1][0][0])
        self.assertAlmostEqual(c0 + 11.601, self.dsp[2].get_result()[1][0][0])

    def test_notify_updated(self):
        self.rtd_model.recalculate(-1)
        self.assert_print(["model_1: Inlet profile updated",
                           "model_1: Unit operation `uo_1` updated",
                           "model_1: Unit operation `uo_2` updated",
                           "model_1: Unit operation `uo_3` updated"])

        data_tree = self.logger.get_data_tree(self.title)

        np.testing.assert_array_equal(data_tree[self.inlet.uo_id]['f'], self.inlet.get_result()[0])
        np.testing.assert_array_equal(data_tree[self.inlet.uo_id]['c'], self.inlet.get_result()[1])
        np.testing.assert_array_equal(data_tree[self.dsp[2].uo_id]['f'], self.dsp[2].get_result()[0])
        np.testing.assert_array_equal(data_tree[self.dsp[2].uo_id]['c'], self.dsp[2].get_result()[1])

    def test_user_interface(self):
        ui = MockUpUserInterface(self.rtd_model)

        # default values
        self.assertEqual(ui.x_label, 't')
        self.assertEqual(ui.y_label_c, 'c')
        self.assertEqual(ui.y_label_f, 'f')
        self.assertEqual(len(ui.species_label), self.inlet.get_n_species())
        self.assertEqual(ui.start_at, -1)

        self.assertFalse(ui.ui_constructed)
        ui.build_ui()
        self.assertTrue(ui.ui_constructed)

        # recalculate
        ui.recalculate()
        np.testing.assert_array_almost_equal(
            np.array(ui.updated_ui_list),
            np.array([[-1, 2, 2.5], [0, 2, 2.6], [1, 2, 2.61], [2, 2, 2.611]])
        )

        # recalculate subset
        ui.start_at = 1
        ui.updated_ui_list.clear()
        ui.recalculate()
        np.testing.assert_array_almost_equal(
            np.array(ui.updated_ui_list),
            np.array([[1, 2, 2.61], [2, 2, 2.611]])
        )

        # recalculate subset
        ui.start_at = 1
        ui.updated_ui_list.clear()
        ui.recalculate(True)
        np.testing.assert_array_almost_equal(
            np.array(ui.updated_ui_list),
            np.array([[-1, 3, 4], [0, 3, 4.1], [1, 3, 4.11], [2, 3, 4.111]])
        )
