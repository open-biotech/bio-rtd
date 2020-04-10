import numbers
import unittest

import numpy as np

from bio_rtd import peak_shapes, utils
from bio_rtd.uo import surge_tank
from bio_rtd.utils import vectors
from bio_rtd_test.aux_bio_rtd_test import TestLogger


class MockUpNoSimCstr(surge_tank.CSTR):
    sim_conv = False
    sim_num = False

    def _sim_convolution(self):
        assert not self.sim_conv
        self.sim_conv = True

    def _sim_numerical(self):
        assert not self.sim_num
        self.sim_num = True


def f_constant(self, f_const=1.0):
    return np.ones_like(self.t) * f_const


def f_box_shaped(self, f_on=1.0):
    _f = np.ones_like(self.t) * f_on
    t_on = 20
    t_off = self.t[-1] * 19 / 20
    _f[int(round(t_on / self.dt)):int(round(t_off / self.dt))] = f_on
    return _f


# noinspection DuplicatedCode
def f_periodic(self, f_on=1.0):
    _f = np.zeros_like(self.t)
    t_period = 20.23
    t_on = 5.34
    i_on = int(round(t_on / self.dt))
    t_on = i_on * self.dt
    t_delay = 20
    t_shorter_end = 40
    t_period_start = np.arange(t_delay, self.t[-1] - t_shorter_end, t_period)
    dt = int(round(t_period_start[0])) - t_period_start[0]
    t_period_start += dt
    i_f_start = [t_p / self.dt for t_p in t_period_start]
    df_init = round(i_f_start[0]) - i_f_start[0]
    i_f_start = [i + df_init for i in i_f_start]
    for i in i_f_start:
        i_r = int(round(i))
        _f[i_r:i_r + i_on] = f_on
    _f[self.t.size - int(round(t_shorter_end / self.dt)):] = 0
    self.f_period_average = f_on * i_on * self.dt / t_period
    self.t_period = t_period
    self.t_on = t_on
    self.i_f_start = i_f_start
    return _f


# noinspection DuplicatedCode
def f_periodic_2(self, f_on=1.0):
    # one full period and one clipped
    _f = np.zeros_like(self.t)
    t_period = 120
    t_on = 40
    i_on = int(round(t_on / self.dt))
    t_on = i_on * self.dt
    # t_delay = 30
    t_shorter_end = 30
    i_f_start = [t_p / self.dt for t_p in [30, 150]]
    df_init = round(i_f_start[0]) - i_f_start[0]
    i_f_start = [i + df_init for i in i_f_start]
    for i in i_f_start:
        i_r = int(round(i))
        _f[i_r:i_r + i_on] = f_on
    _f[self.t.size - int(round(t_shorter_end / self.dt)):] = 0
    self.f_period_average = f_on * i_on * self.dt / t_period
    self.t_period = t_period
    self.t_on = t_on
    self.i_f_start = i_f_start
    return _f


def c_profile_1_specie(self):
    c = np.ones([1, self.t.size]) * 5.2
    c[0, 40:110] = 0
    return c


def c_profile_2_species(self):
    c = np.ones([2, self.t.size])
    c[0, :20] = 0
    c[1, :] = 2
    c[1, 30:] = 0
    return c


class CstrTest(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 200, 1200)
        self.dt = self.t[1]
        self.uo_id = "cstr"
        self.gui_title = "Ideal CSTR"
        self.cstr = surge_tank.CSTR(self.t, self.uo_id, self.gui_title)
        self.cstr.log = TestLogger()
        self.f_period_average = 0
        self.t_period = 0
        self.i_f_start = 0
        self.t_on = 0

    def assert_positive_value(self, par_name, func):
        v = getattr(self.cstr, par_name)
        if isinstance(v, numbers.Number):
            setattr(self.cstr, par_name, -1)
        if isinstance(v, np.ndarray):
            setattr(self.cstr, par_name, np.array([]))
        if isinstance(v, bool):
            setattr(self.cstr, par_name, not v)
        with self.assertRaises(AssertionError):
            func()
        setattr(self.cstr, par_name, v)

    def test_init(self):
        # test passed parameters
        np.testing.assert_array_equal(self.cstr._t, self.t)
        self.assertEqual(self.cstr.uo_id, self.uo_id)
        self.assertEqual(self.cstr.gui_title, self.gui_title)

        # test default parameters
        # volume
        self.assertEqual(-1, self.cstr.rt_target)
        self.assertEqual(-1, self.cstr.v_void)
        self.assertEqual(-1, self.cstr.v_min)
        self.assertEqual(-1, self.cstr.v_min_ratio)
        # init volume
        self.assertEqual(-1, self.cstr.v_init)
        self.assertEqual(-1, self.cstr.v_init_ratio)
        # init conc
        self.assertTrue(self.cstr.c_init.size == 0)
        # empty start
        self.assertFalse(self.cstr.starts_empty)

    def test_calc_f_out_target_and_t_cycle(self):
        # constant
        self.cstr._f = f_constant(self, 5)
        self.cstr._calc_f_out_target_and_t_cycle()
        self.assertTrue(self.cstr._is_f_in_box_shaped)
        self.assertEqual(5, self.cstr._f_out_target)
        self.assertEqual(0, self.cstr._t_cycle)
        # box shaped
        self.cstr._f = f_box_shaped(self, 15)
        self.cstr._calc_f_out_target_and_t_cycle()
        self.assertTrue(self.cstr._is_f_in_box_shaped)
        self.assertEqual(15, self.cstr._f_out_target)
        self.assertEqual(0, self.cstr._t_cycle)

        def check_periodic():
            self.cstr._calc_f_out_target_and_t_cycle()
            self.assertFalse(self.cstr._is_f_in_box_shaped)
            self.assertAlmostEqual(self.f_period_average,
                                   self.cstr._f_out_target,
                                   0)
            self.assertAlmostEqual(self.t_period, self.cstr._t_cycle, 0)

        # periodic 1
        self.cstr._f = f_periodic(self, 15)
        check_periodic()
        # periodic 2
        self.cstr._f = f_periodic_2(self, 25)
        check_periodic()

    def test_calc_v_void(self):
        # prepare
        self.cstr._f = f_periodic_2(self, 1.43)
        self.cstr._calc_f_out_target_and_t_cycle()

        # assert
        with self.assertRaises(RuntimeError):
            self.cstr._calc_v_void()

        def calc_delta_v():
            f_in = self.cstr._f.max()
            return self.cstr._f_out_target \
                * (1 - self.cstr._f_out_target / f_in) \
                * self.cstr._t_cycle

        def use_rt_target():
            self.cstr.rt_target = 10.2
            self.cstr._calc_v_void()
            self.assertEqual(self.cstr.rt_target * self.cstr._f_out_target,
                             self.cstr._v_void)

        def use_v_min_ratio():
            self.cstr.v_min_ratio = 0.2
            self.cstr._t_cycle = -1
            with self.assertRaises(AssertionError):
                self.cstr._calc_v_void()
            self.cstr._t_cycle = 15.2
            self.cstr._calc_v_void()
            self.assertEqual(
                calc_delta_v() / (1 - self.cstr.v_min_ratio),
                self.cstr._v_void
            )

        def use_v_min():
            self.cstr.v_min = 14.3
            self.cstr._t_cycle = -1
            with self.assertRaises(AssertionError):
                self.cstr._calc_v_void()
            self.cstr._t_cycle = 11.2
            self.cstr._calc_v_void()
            self.assertEqual(
                calc_delta_v() + self.cstr.v_min,
                self.cstr._v_void
            )

        def use_v_void():
            self.cstr.v_void = 22.2
            self.cstr._calc_v_void()
            self.assertEqual(
                self.cstr.v_void,
                self.cstr._v_void
            )

        # calc
        # rt_target
        use_rt_target()

        # v_min_ratio
        with self.assertWarns(Warning):
            # test priority over rt_target
            use_v_min_ratio()
        self.cstr.rt_target = -1
        use_v_min_ratio()

        # v_min
        with self.assertWarns(Warning):  # test parameter priority
            # test priority over v_min_ratio
            use_v_min()
        self.cstr.v_min_ratio = -1
        use_v_min()

        # v_void
        with self.assertWarns(Warning):  # test parameter priority
            # test priority over v_min
            use_v_void()
        self.cstr.v_min = -1
        use_v_void()

    # noinspection DuplicatedCode
    def test_calc_v_init(self):
        # default: `_v_init = _v_void` & warning
        with self.assertRaises(AssertionError):
            self.cstr._calc_v_init()
        self.cstr._v_void = 1.2
        with self.assertWarns(Warning):
            self.cstr._calc_v_init()
            self.assertEqual(self.cstr._v_void, self.cstr._v_init)

        # v_init_ratio
        self.cstr.v_init_ratio = 0.2
        self.cstr._v_void = -1
        with self.assertRaises(AssertionError):
            self.cstr._calc_v_init()
        self.cstr._v_void = 1.3
        self.cstr._calc_v_init()
        self.assertEqual(self.cstr._v_void * self.cstr.v_init_ratio,
                         self.cstr._v_init)
        self.cstr._v_void = -1

        # v_init
        self.cstr.v_init = 35.2
        with self.assertWarns(Warning):
            # priority over v_init_ratio
            self.cstr._calc_v_init()
            self.assertEqual(self.cstr.v_init, self.cstr._v_init)
        self.cstr.v_init_ratio = -1
        self.cstr._calc_v_init()
        self.assertEqual(self.cstr.v_init, self.cstr._v_init)
        self.cstr.v_init = 0
        self.cstr._calc_v_init()
        self.assertEqual(self.cstr.v_init, self.cstr._v_init)

        # starts empty
        # to ignore nby the method
        self.cstr.v_init = 335.2
        self.cstr.v_init_ratio = 24.2
        # set starts_empty
        self.cstr.starts_empty = True
        # test results
        self.cstr._calc_v_init()
        self.assertEqual(0, self.cstr._v_init)

    # noinspection DuplicatedCode
    def test_calc_c_init(self):
        # prepare
        self.cstr._n_species = 2

        # default
        self.cstr._calc_c_init()
        np.testing.assert_array_equal(np.array([[0], [0]]),
                                      self.cstr._c_init)

        # defined
        self.cstr.c_init = np.array([[2.2], [0.3]])
        self.cstr._calc_c_init()
        np.testing.assert_array_equal(np.array([[2.2], [0.3]]),
                                      self.cstr._c_init)

        # defined 2
        self.cstr.c_init = np.array([3.2, 0.2])
        self.cstr._calc_c_init()
        np.testing.assert_array_equal(np.array([[3.2], [0.2]]),
                                      self.cstr._c_init)

        # defined wrong
        self.cstr.c_init = np.array([3.2])
        with self.assertRaises(ValueError):
            self.cstr._calc_c_init()

    def sim_convolution(self):
        rt = self.cstr._v_void / self.cstr._f_out_target
        t = np.arange(0, min(rt * 10, self.t[-1]), self.dt)
        p = peak_shapes.tanks_in_series(t, rt, 1, self.cstr.log)
        c = utils.convolution.time_conv(self.dt,
                                        self.cstr._c, p,
                                        self.cstr._c_init)
        return c, p, rt

    def test_sim_convolution(self):
        # prepare
        self.cstr._c = c_profile_2_species(self)
        self.cstr._n_species = 2
        data_log = self.cstr.log.get_data_tree(self.cstr.uo_id)
        v_void = 14.5
        f_out_target = 2.3
        c_init = np.array([[2.2], [3.1]])

        # assign
        self.cstr._v_void = v_void
        self.cstr._f_out_target = f_out_target
        self.cstr._is_f_in_box_shaped = True
        self.cstr._c_init = c_init

        # assert parameters
        self.assert_positive_value("_f_out_target", self.cstr._sim_convolution)
        self.assert_positive_value("_v_void", self.cstr._sim_convolution)
        self.assert_positive_value("_c_init", self.cstr._sim_convolution)
        self.assert_positive_value("_f_out_target", self.cstr._sim_convolution)
        self.assert_positive_value("_is_f_in_box_shaped",
                                   self.cstr._sim_convolution)

        def eval_sim_conv() -> (np.ndarray, np.ndarray, float):
            # targets
            c, p, rt = self.sim_convolution()
            # calc
            self.cstr._sim_convolution()
            # compare
            self.assertEqual(rt, data_log["rt_mean"])
            np.testing.assert_array_almost_equal(p, data_log["p_rtd"])
            np.testing.assert_array_almost_equal(c, self.cstr._c)

        # sim 2 species
        # warning due to low temporal resolution
        with self.assertWarns(Warning, msg=f"Warning: Peak shape: integral:"
                                           f" 1.0132418166911727"):
            eval_sim_conv()

        # sim 1 specie
        self.cstr._c = c_profile_1_specie(self)
        self.cstr._n_species = 1
        self.cstr._c_init = np.array([[2.1]])
        # warning due to low temporal resolution
        with self.assertWarns(Warning, msg=f"Warning: Peak shape: integral:"
                                           f" 1.0132418166911727"):
            eval_sim_conv()

    def sim_numerical(self):
        # status in cstr
        v = self.cstr._v_init
        m = self.cstr._c_init.flatten() * v

        # init result vectors
        _c = np.zeros_like(self.cstr._c)
        _f = np.zeros_like(self.cstr._f)

        # do not turn off outlet once started
        keep_outlet_on = False

        for i in range(utils.vectors.true_start(self.cstr._f > 0),
                       self.cstr._t.size):
            # fill in
            dv = self.cstr._f[i] * self.dt
            v += dv
            m += self.cstr._c[:, i] * dv

            if v < self.cstr._f_out_target * self.cstr._dt:  # CSTR is empty
                if not keep_outlet_on:  # it was not yet turned on
                    _c[:, i] = 0
                    _f[i] = 0
                else:  # CSTR dry during operation -> shout it down
                    self.cstr.log.i(f"CSTR ran dry during operation"
                                    f" -> shutting down")
                    _c[:, i:] = 0
                    _f[i:] = 0
                    return _f, _c 
            elif v < self.cstr._v_void and not keep_outlet_on:
                # Wait until filled.
                _c[:, i] = 0
                _f[i] = 0
            else:  # outlet on
                keep_outlet_on = True
                # calc current concentration
                c = m / v
                # get outlet
                _c[:, i] = c
                _f[i] = self.cstr._f_out_target
                # subtract outlet from cstr
                v -= _f[i] * self.dt
                m -= _f[i] * self.dt * c

        return _f, _c

    def test_sim_numerical(self):
        # prepare
        self.cstr._c = c_profile_2_species(self)
        self.cstr._n_species = 2
        v_void = 12.5
        v_init = 6.1
        f_out_target = 2.4
        c_init = np.array([[3.2], [5.1]])

        # bind
        self.cstr._f_out_target = f_out_target
        self.cstr._v_void = v_void
        self.cstr._v_init = v_init
        self.cstr._c_init = c_init

        # assert parameters
        self.assert_positive_value("_f_out_target", self.cstr._sim_convolution)
        self.assert_positive_value("_v_void", self.cstr._sim_convolution)
        self.assert_positive_value("_v_init", self.cstr._sim_convolution)
        self.assert_positive_value("_c_init", self.cstr._sim_convolution)

        def eval_sim(include_convolution=False, include_init_small_bump=False):
            # prepare
            self.cstr._calc_f_out_target_and_t_cycle()
            if include_init_small_bump:
                self.cstr._f[0] = 0.002
            if include_convolution:
                self.cstr._v_init = self.cstr._v_void
            else:
                self.cstr._v_init = v_init

            # targets
            if include_convolution:
                f_conv = self.cstr._f.copy()
                c_conv, _, _ = self.sim_convolution()
            f_num, c_num = self.sim_numerical()

            # calc
            self.cstr._sim_numerical()

            # compare
            np.testing.assert_array_almost_equal(f_num, self.cstr._f)
            np.testing.assert_array_almost_equal(c_num, self.cstr._c)
            if include_convolution:
                # noinspection PyUnboundLocalVariable
                i_f_end = utils.vectors.true_end(f_conv > 0)
                np.testing.assert_array_almost_equal(f_conv[:i_f_end],
                                                     self.cstr._f[:i_f_end])
                np.testing.assert_array_almost_equal(c_num[:, :i_f_end],
                                                     self.cstr._c[:, :i_f_end])

        # sim 2 species
        self.cstr._f = f_constant(self, )
        eval_sim(True)
        self.cstr._f = f_periodic(self)
        eval_sim()
        self.cstr._f = f_periodic_2(self)
        eval_sim()

        # sim 1 specie
        self.cstr._c = c_profile_1_specie(self)
        self.cstr._n_species = 1
        self.cstr._c_init = np.array([[2.1]])
        # sim
        self.cstr._f = f_constant(self)
        eval_sim(True)
        self.cstr._f = f_periodic(self)
        eval_sim()
        self.cstr._f = f_periodic_2(self)
        eval_sim()
        self.cstr._f = f_periodic(self)
        self.cstr._f[int(round(self.t.size / 2)):] = 0
        with self.assertWarns(Warning):
            eval_sim()
        # special case with small f at start
        self.cstr._f = f_periodic(self)
        v_init = 0
        eval_sim(False, True)

    def test_calculate(self):
        self.m_cstr = MockUpNoSimCstr(self.t, self.uo_id, self.gui_title)
        self.m_cstr.log = TestLogger()
        self.m_cstr.rt_target = 12.2
        self.m_cstr.v_init = 12.2
        data_log = self.m_cstr.log.get_data_tree(self.uo_id)

        def run_test(assert_conv=True):
            data_log.clear()
            self.m_cstr._calculate()
            self.assertTrue(self.m_cstr.sim_conv is assert_conv)
            self.assertTrue(self.m_cstr.sim_num is not assert_conv)
            self.m_cstr.sim_conv = False
            self.m_cstr.sim_num = False
            # make sure parameters are in log
            self.assertTrue("f_out_target" in data_log.keys())
            self.assertTrue("t_cycle" in data_log.keys())
            self.assertTrue("v_void" in data_log.keys())
            self.assertTrue("v_init" in data_log.keys())
            self.assertTrue("c_init" in data_log.keys())

        self.m_cstr._f = f_constant(self, )
        self.m_cstr._c = c_profile_2_species(self)
        self.m_cstr._n_species = 2
        run_test(True)
        self.m_cstr._f = f_periodic_2(self)
        self.m_cstr._c = c_profile_2_species(self)
        self.m_cstr._n_species = 2
        run_test(False)
        self.m_cstr._f = f_periodic_2(self)
        self.m_cstr._c = c_profile_1_specie(self)
        self.m_cstr._n_species = 1
        self.m_cstr.v_init = 2.2
        run_test(False)


class MockUpNoSimTwoAlternatingCSTRs(surge_tank.TwoAlternatingCSTRs):

    def __init__(self, t: np.ndarray):
        super().__init__(t, "mock_up_a2cstr")
        self.f_calls = []

    def _calc_f_out_target_and_t_cycle(self):
        self.f_calls.append(1)

    def _calc_t_leftover(self):
        self.f_calls.append(2)

    def _calc_i_switch_list(self):
        self.f_calls.append(3)

    def _simulate_cycle_by_cycle(self):
        self.f_calls.append(4)

    def _ensure_box_shape(self):
        self.f_calls.append(5)


class TestTwoAlternatingCSTRs(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 200, 1200)
        self.dt = self.t[1]
        self.uo_id = "twin_cstr_system"
        self.gui_title = "2 Alternating CSTRs"
        self.a2cstr = surge_tank.TwoAlternatingCSTRs(self.t,
                                                     self.uo_id,
                                                     self.gui_title)
        self.a2cstr.log = TestLogger()
        self.f_period_average = 0
        self.t_period = 0
        self.t_on = 0
        self.i_f_start = []

    def assert_defined_value(self, par_name, func):
        v = getattr(self.a2cstr, par_name)
        delattr(self.a2cstr, par_name)
        with self.assertRaises(AssertionError):
            func()
        setattr(self.a2cstr, par_name, v)

    def assert_positive_value(self, par_name, func):
        v = getattr(self.a2cstr, par_name)
        setattr(self.a2cstr, par_name, -1)
        with self.assertRaises(AssertionError):
            func()
        setattr(self.a2cstr, par_name, v)

    def test_init(self):
        # Test passed parameters.
        np.testing.assert_array_equal(self.a2cstr._t, self.t)
        self.assertEqual(self.a2cstr.uo_id, self.uo_id)
        self.assertEqual(self.a2cstr.gui_title, self.gui_title)

        # Test default parameters.
        self.assertEqual(1, self.a2cstr.collect_n_periods)
        self.assertEqual(0.9, self.a2cstr.relative_role_switch_time)
        self.assertEqual(-1, self.a2cstr.t_cycle)
        self.assertEqual(-1, self.a2cstr.v_cycle)
        self.assertEqual(-1, self.a2cstr.v_leftover)
        self.assertEqual(-1, self.a2cstr.v_leftover_rel)

    def test_calc_f_out_target_and_t_cycle_constant(self):

        def test_non_periodic_flow(f_out):
            self.assertTrue(self.a2cstr._is_flow_box_shaped())
            self.a2cstr.t_cycle = -1
            self.a2cstr.v_cycle = -1
            with self.assertRaises(AssertionError):
                self.a2cstr._calc_f_out_target_and_t_cycle()
            self.a2cstr.t_cycle = 15
            self.a2cstr.v_cycle = 15 * f_out
            with self.assertRaises(AssertionError):
                self.a2cstr._calc_f_out_target_and_t_cycle()
            self.a2cstr.v_cycle = -1
            self.a2cstr._calc_f_out_target_and_t_cycle()
            self.assertEqual(f_out, self.a2cstr._f_out_target)
            self.assertEqual(15, self.a2cstr._t_cycle)
            self.a2cstr.t_cycle = -1
            self.a2cstr.v_cycle = 15 * f_out
            self.a2cstr._calc_f_out_target_and_t_cycle()
            self.assertEqual(f_out, self.a2cstr._f_out_target)
            self.assertEqual(15, self.a2cstr._t_cycle)

        # Constant inlet flow rate.
        self.a2cstr._f = f_constant(self, 5)
        test_non_periodic_flow(5)
        # Box shaped inlet profile.
        self.a2cstr._f = f_box_shaped(self, 15)
        test_non_periodic_flow(15)

    def test_calc_f_out_target_and_t_cycle_periodic(self):

        def test_periodic_flow():
            self.a2cstr._calc_f_out_target_and_t_cycle()
            self.assertFalse(self.a2cstr._is_flow_box_shaped())
            self.assertAlmostEqual(
                self.f_period_average,
                self.a2cstr._f_out_target, 0)
            self.assertAlmostEqual(
                self.t_period * self.a2cstr.collect_n_periods,
                self.a2cstr._t_cycle, 0)

        # Periodic inlet flow rate 1.
        self.a2cstr._f = f_periodic(self, 15)
        test_periodic_flow()
        # Periodic inlet flow rate 2.
        self.a2cstr._f = f_periodic_2(self, 25)
        test_periodic_flow()

    def test_calc_t_leftover(self):

        def run_test(v_leftover, v_leftover_rel,
                     f_out_target, t_cycle):
            self.a2cstr.v_leftover = v_leftover
            self.a2cstr.v_leftover_rel = v_leftover_rel
            self.a2cstr._f_out_target = f_out_target
            self.a2cstr._t_cycle = t_cycle
            self.assert_defined_value('_f_out_target',
                                      self.a2cstr._calc_t_leftover)
            self.assert_defined_value('_t_cycle',
                                      self.a2cstr._calc_t_leftover)
            if v_leftover > 0 and v_leftover_rel > 0:
                with self.assertRaises(AssertionError):
                    self.a2cstr._calc_t_leftover()
                return
            if v_leftover > 0:
                t_leftover = v_leftover / f_out_target
            elif v_leftover_rel > 0:
                t_leftover = v_leftover_rel * t_cycle
            else:
                t_leftover = 0
            # Call method.
            self.a2cstr._calc_t_leftover()
            # Compare.
            self.assertAlmostEqual(t_leftover, self.a2cstr._t_leftover)

        run_test(3.4, 0, 3.5, 14.5)
        run_test(-1, 0.5, 3.5, 14.5)
        run_test(-1, 0.5, 3.5, 0)
        run_test(0, 1.5, 3.5, 1)
        run_test(1.1, -1, 3.5, 1)
        run_test(1.1, 0, 3.5, 1)
        run_test(3.4, 0.2, 3.5, 14.5)
        run_test(-1, -1, 3.5, 14.5)
        run_test(-1, 0, 3.5, 14.5)
        run_test(0, -1, 3.5, 14.5)

    def test_calc_i_switch_list_constant(self):

        def test_non_periodic_flow(f_out: float,
                                   t_cycle: float,
                                   t_leftovers: float):
            self.assertTrue(self.a2cstr._is_flow_box_shaped())
            self.a2cstr._f_out_target = f_out
            self.a2cstr._t_cycle = t_cycle
            self.a2cstr._t_leftover = t_leftovers
            self.assert_defined_value('_f_out_target',
                                      self.a2cstr._calc_i_switch_list)
            self.assert_defined_value('_t_cycle',
                                      self.a2cstr._calc_i_switch_list)
            self.assert_defined_value('_t_leftover',
                                      self.a2cstr._calc_i_switch_list)
            self.a2cstr._calc_i_switch_list()
            i_start, i_end = vectors.true_start_and_end(self.a2cstr._f > 0)
            i_cycle = t_cycle / self.dt
            i_leftovers = t_leftovers / self.dt
            i_switch_list = []
            i_switch = i_start + i_cycle + 2 * i_leftovers
            i_switch_list.append(i_switch)
            i_switch += i_cycle + i_leftovers
            i_switch_list.append(i_switch)
            i_switch += i_cycle
            while i_switch < self.t.size:
                i_switch_list.append(i_switch)
                i_switch += i_cycle
            np.testing.assert_array_almost_equal(
                i_switch_list,
                self.a2cstr._i_switch_list
            )

        # Constant inlet flow rate.
        self.a2cstr._f = f_constant(self, 5)
        test_non_periodic_flow(5, 15.4, 0)
        test_non_periodic_flow(5, 15.4, 2)
        # Box shaped inlet profile.
        self.a2cstr._f = f_box_shaped(self, 15)
        test_non_periodic_flow(15, 10.2, 0)
        test_non_periodic_flow(15, 15.4, 0.8)
        test_non_periodic_flow(15, 15.4, 3)

    def test_calc_i_switch_list_periodic(self):

        def test_periodic_flow(t_leftovers: float,
                               relative_role_switch_time: float):
            self.assertFalse(self.a2cstr._is_flow_box_shaped())
            self.a2cstr._f_out_target = self.f_period_average
            self.a2cstr._t_cycle = self.t_period
            self.a2cstr._t_leftover = t_leftovers
            self.a2cstr.relative_role_switch_time = relative_role_switch_time
            self.assert_defined_value('_f_out_target',
                                      self.a2cstr._calc_i_switch_list)
            self.assert_defined_value('_t_cycle',
                                      self.a2cstr._calc_i_switch_list)
            self.assert_defined_value('_t_leftover',
                                      self.a2cstr._calc_i_switch_list)
            self.assert_positive_value('relative_role_switch_time',
                                       self.a2cstr._calc_i_switch_list)
            # Should be <= 1.
            self.a2cstr.relative_role_switch_time = 1.1
            with self.assertRaises(AssertionError):
                self.a2cstr._calc_i_switch_list()
            self.a2cstr.relative_role_switch_time = relative_role_switch_time
            # Calc reference.
            i_f_on = self.t_on / self.dt
            i_cycle = self.t_period / self.dt
            i_leftovers = t_leftovers / self.dt
            i_delay = relative_role_switch_time * (i_cycle - i_f_on)
            if i_delay < 2 * i_leftovers:
                # Running function should raise assertion error.
                with self.assertRaises(AssertionError):
                    self.a2cstr._calc_i_switch_list()
                return
            i_switch_list = []
            for i, i_sw in enumerate(self.i_f_start.copy()):
                i_sw += i_delay + i_f_on
                if i == 1:
                    i_sw -= i_leftovers
                elif i > 1:
                    i_sw -= 2 * i_leftovers
                if i_sw >= self.t.size:
                    break
                i_switch_list.append(i_sw)
            # Add extra entries after flow rate ends.
            i_switch = i_switch_list[-1] + i_cycle
            while i_switch < self.t.size:
                i_switch_list.append(i_switch)
                i_switch += i_cycle

            # Run function in the model.
            self.a2cstr._calc_i_switch_list()
            # Compare.
            np.testing.assert_array_almost_equal(
                i_switch_list,
                self.a2cstr._i_switch_list
            )

        # Periodic inlet flow rate 1.
        self.a2cstr._f = f_periodic(self, 15)
        self.a2cstr._calc_f_out_target_and_t_cycle()
        test_periodic_flow(0, 0)
        test_periodic_flow(2, 0.1)
        test_periodic_flow(2., 0.9)
        test_periodic_flow(3., 0)
        # Periodic inlet flow rate 2.
        self.a2cstr._f = f_periodic_2(self, 25)
        self.a2cstr._calc_f_out_target_and_t_cycle()
        test_periodic_flow(0, 0)
        test_periodic_flow(2, 0.1)
        test_periodic_flow(2., 0.9)
        test_periodic_flow(12., 0.9)
        test_periodic_flow(3., 0)

    def run_simulation(self):
        f_in = self.a2cstr._f.copy()
        c_in = self.a2cstr._c.copy()
        self.a2cstr._calc_f_out_target_and_t_cycle()
        self.a2cstr._calc_t_leftover()
        self.a2cstr._calc_i_switch_list()
        self.a2cstr._simulate_cycle_by_cycle()
        f_out, c_out = self.a2cstr._f, self.a2cstr._c

        cycles_data = self.a2cstr._log_tree["cycles"]
        v1_leftover = cycles_data[-1]["v_after_discharge"]
        v2_leftover = cycles_data[-2]["v_after_discharge"]
        m1_leftover = cycles_data[-1]["m_after_discharge"]
        m2_leftover = cycles_data[-2]["m_after_discharge"]

        # from bokeh.plotting import figure, show
        # f = figure()
        # f.line(self.t, f_in)
        # f.line(self.t, f_out)
        # f.circle(cycles_data[-1]["i_start_discharge"] * self.dt, 5)
        # f.circle(cycles_data[-2]["i_start_discharge"] * self.dt, 5)
        # show(f, browser='firefox')

        # print(f_in.sum() * self.dt)
        # print(sum([v['v_after_collection'] for v in cycles_data]))
        # print(f_out.sum() * self.dt + v1_leftover + v2_leftover)
        # print(f_out.sum() * self.dt)
        # print(v1_leftover + v2_leftover)
        # print((f_in.sum() - f_out.sum()) / self.a2cstr._f_out_target)
        # print((f_in.sum() - f_out.sum()
        #        - (v1_leftover + v2_leftover) / self.dt)
        #       / self.a2cstr._f_out_target)
        # print(f_out.mean())
        # print(f_out.max())

        # Mass balance test.
        self.assertAlmostEqual(
            f_in.sum() * self.dt,
            f_out.sum() * self.dt + v1_leftover + v2_leftover
        )
        self.assertAlmostEqual(
            (c_in * f_in[np.newaxis, :]).sum() * self.dt,
            (c_out * f_out[np.newaxis, :]).sum() * self.dt
            + m1_leftover.sum() + m2_leftover.sum()
        )

    def test_simulate_cycle_by_cycle_constant(self):
        self.a2cstr.t_cycle = 14.5
        # # Constant inlet flow rate.
        self.a2cstr._f = f_constant(self, 5)
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()
        self.a2cstr._f = f_constant(self, 5)
        self.a2cstr._c = c_profile_2_species(self)
        self.run_simulation()
        # Box shaped inlet profile.
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.run_simulation()
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr.v_leftover = 1.3
        self.run_simulation()
        # Border case 1 (at end of t).
        self.a2cstr._f = np.ones_like(self.t) * 14
        self.a2cstr._f[-250:] = 0
        self.a2cstr.t_cycle = 50.01
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()
        # Border case 2 (< 1 time step discharge before t end).
        self.a2cstr._f = np.ones_like(self.t) * 14
        self.a2cstr._f[-650:] = 0
        self.a2cstr.t_cycle = 91.71
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()

    def test_simulate_cycle_by_cycle_periodic(self):
        # Periodic inlet flow rate 1.
        self.a2cstr._f = f_periodic(self, 15)
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()
        self.a2cstr._f = f_periodic(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.run_simulation()
        # Periodic inlet flow rate 2.
        self.a2cstr._f = f_periodic_2(self, 25)
        self.a2cstr._c = c_profile_1_specie(self)
        self.run_simulation()
        self.a2cstr._f = f_periodic_2(self, 25)
        self.a2cstr._c = c_profile_2_species(self)
        self.run_simulation()

    def test_calculate_dry_run(self):
        self.a2cstr.t_cycle = 14.5
        # # Constant inlet flow rate.
        self.a2cstr._f = f_constant(self, 5)
        self.a2cstr._c = c_profile_1_specie(self)
        self.a2cstr._calculate()
        self.a2cstr._f = f_constant(self, 5)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr._calculate()
        # Box shaped inlet profile.
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_1_specie(self)
        self.a2cstr._calculate()
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr._calculate()
        self.a2cstr._f = f_box_shaped(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr.v_leftover = 1.3
        self.a2cstr._calculate()
        # Periodic inlet flow rate 1.
        self.a2cstr._f = f_periodic(self, 15)
        self.a2cstr._c = c_profile_1_specie(self)
        self.a2cstr._calculate()
        self.a2cstr._f = f_periodic(self, 15)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr._calculate()
        # Periodic inlet flow rate 2.
        self.a2cstr._f = f_periodic_2(self, 25)
        self.a2cstr._c = c_profile_1_specie(self)
        self.a2cstr._calculate()
        self.a2cstr._f = f_periodic_2(self, 25)
        self.a2cstr._c = c_profile_2_species(self)
        self.a2cstr._calculate()

    def test_ensure_box_shape(self):

        def run_test(_f_in, _f_out):
            self.a2cstr._f = _f_in.copy()
            self.a2cstr._ensure_box_shape()
            np.testing.assert_array_almost_equal(
                _f_out,
                self.a2cstr._f
            )

        # Keep.
        f_in = np.ones_like(self.t) * 3.5
        run_test(f_in, f_in)
        # Clip start.
        f_in[0] *= 0.98
        f_out = f_in.copy()
        f_out[0] = 0
        run_test(f_in, f_out)
        # Clip end.
        f_in[-1] *= 0.98
        f_out[-1] = 0
        run_test(f_in, f_out)
        # Clip start + delay start.
        f_in[:10] = 0
        f_in[10] *= 0.98
        f_out[:11] = 0
        run_test(f_in, f_out)
        # Clip end + short end.
        f_in[-10:] = 0
        f_in[-11] *= 0.98
        f_out[-11:] = 0
        run_test(f_in, f_out)
        # Fix minimal bump.
        f_in[20] *= 0.99999
        run_test(f_in, f_out)
        # Fix minimal bump.
        f_in[22] *= 1.00001
        run_test(f_in, f_out)
        # Do not allow bigger bump.
        f_in[25] *= 0.9999
        with self.assertRaises(AssertionError):
            run_test(f_in, f_out)
        # Do not allow bigger bump.
        f_in[25] = f_in[24] * 1.0001
        with self.assertRaises(AssertionError):
            run_test(f_in, f_out)

    def test_calculate(self):
        mu_uo = MockUpNoSimTwoAlternatingCSTRs(self.t)
        mu_uo._calculate()
        self.assertTrue([1, 2, 3, 4, 5], mu_uo.f_calls)
