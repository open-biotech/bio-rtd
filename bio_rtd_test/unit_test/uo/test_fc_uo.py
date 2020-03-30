import unittest

import numpy as np

from bio_rtd.uo import fc_uo
from bio_rtd import pdf
from bio_rtd.utils import convolution
from bio_rtd.utils import vectors

from bio_rtd_test.aux_bio_rtd_test import TestLogger


class TestDilution(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 230)
        self.dilution_ratio = 1.2
        self.uo_id = "dilution"
        self.gui_title = "Dilution Step"
        self.uo = fc_uo.Dilution(
            self.t, self.dilution_ratio, self.uo_id, self.gui_title
        )
        self.uo.log = TestLogger()

    def test_init(self):
        # assert proper dilution ratio ( > 1)
        with self.assertRaises(AssertionError):
            fc_uo.Dilution(self.t, 0.99, self.uo_id)
        with self.assertRaises(AssertionError):
            fc_uo.Dilution(self.t, 1, self.uo_id)
        fc_uo.Dilution(self.t, 1.01, self.uo_id)

        # assert bindings
        np.testing.assert_array_equal(self.t, self.uo._t)
        self.assertEqual(self.dilution_ratio, self.uo.dilution_ratio)
        self.assertEqual(self.uo_id, self.uo.uo_id)
        self.assertEqual(self.gui_title, self.uo.gui_title)

        # assert default values
        self.assertTrue(self.uo.c_add_buffer.size == 0)

    def test_calc_c_add(self):
        # prepare
        self.uo._n_species = 3

        # assert proper c_add size
        with self.assertRaises(AssertionError):
            self.uo.c_add_buffer = np.array([2])
            self.uo._calc_c_add_buffer()

        # calc
        # empty -> zero
        self.uo.c_add_buffer = np.array([])
        self.uo._calc_c_add_buffer()
        np.testing.assert_array_almost_equal(np.zeros([3, 1]), self.uo._c_add_buffer)
        # filled -> reshape
        self.uo.c_add_buffer = np.array([1, 2, 3])
        self.uo._calc_c_add_buffer()
        np.testing.assert_array_almost_equal(self.uo.c_add_buffer.reshape(3, 1), self.uo._c_add_buffer)
        # filled -> reshape
        self.uo.c_add_buffer = np.array([[1], [2], [3]])
        self.uo._calc_c_add_buffer()
        np.testing.assert_array_almost_equal(self.uo.c_add_buffer, self.uo._c_add_buffer)

    # noinspection DuplicatedCode
    def test_calculate(self):
        # prepare
        self.uo._n_species = 3
        self.uo._f = np.ones_like(self.t)
        self.uo._c = np.ones([self.uo._n_species, self.t.size]) * np.array([[1], [2], [3]])
        self.uo.dilution_ratio = 1.2

        # assert _c_add
        with self.assertRaises(AssertionError):
            self.uo.c_add_buffer = np.array([2])
            self.uo._calc_c_add_buffer()
        self.uo._c_add_buffer = np.array([])

        # assert dilution_ratio >= 1
        with self.assertRaises(AssertionError):
            self.uo.dilution_ratio = 0.99
            self.uo._calculate()
        self.uo.dilution_ratio = 1.2

        # calc
        def run_test(dilution_ratio, c_add_buffer):
            self.uo.c_add_buffer = c_add_buffer
            self.uo.dilution_ratio = dilution_ratio
            f_target = self.uo._f.copy() * dilution_ratio
            c_target = self.uo._c.copy() / dilution_ratio
            if c_add_buffer.size > 0:
                c_target += c_add_buffer * (dilution_ratio - 1) / dilution_ratio
            self.uo._calc_c_add_buffer()
            self.uo._calculate()
            np.testing.assert_array_almost_equal(f_target, self.uo._f)
            np.testing.assert_array_almost_equal(c_target, self.uo._c)

        with self.assertWarns(Warning, msg="Dilution ratio is set to 1"):
            run_test(1, np.array([]))
        run_test(1.2, np.array([[0.5], [0.6], [0.7]]))
        run_test(1.2, np.array([]))
        with self.assertWarns(Warning, msg="Dilution ratio is set to 1"):
            run_test(1, np.array([[0.52], [0.63], [0.72]]))


class TestConcentration(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 230)
        self.flow_reduction = 1.2
        self.uo_id = "concentration"
        self.gui_title = "Concentration Step"
        self.uo = fc_uo.Concentration(
            self.t, self.flow_reduction, self.uo_id, self.gui_title
        )
        self.uo.log = TestLogger()

    # noinspection DuplicatedCode
    def test_init(self):
        # assert proper flow reduction ( > 1)
        with self.assertRaises(AssertionError):
            fc_uo.Concentration(self.t, 0.99, self.uo_id)
        with self.assertRaises(AssertionError):
            fc_uo.Concentration(self.t, 1, self.uo_id)
        fc_uo.Concentration(self.t, 1.01, self.uo_id)

        # assert bindings
        np.testing.assert_array_equal(self.t, self.uo._t)
        self.assertEqual(self.flow_reduction, self.uo.flow_reduction)
        self.assertEqual(self.uo_id, self.uo.uo_id)
        self.assertEqual(self.gui_title, self.uo.gui_title)

        # assert default values
        self.assertEqual(0, self.uo.relative_losses)
        self.assertEqual([], self.uo.non_retained_species)

    # noinspection DuplicatedCode
    def test_calculate(self):
        # prepare
        self.uo._n_species = 3
        self.uo._f = np.ones_like(self.t)
        self.uo._c = np.ones([self.uo._n_species, self.t.size]) \
                     * np.array([[1], [2], [3]])
        self.uo.relative_losses = 0.2

        # assert valid non_retained_species
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 3]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 0]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [1, 0]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [-1]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 1, 2]
            self.uo._calculate()
        self.uo.non_retained_species = [0]

        # assert valid relative_losses
        with self.assertRaises(AssertionError):
            self.uo.relative_losses = -0.01
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.relative_losses = 1.01
            self.uo._calculate()
        self.uo.relative_losses = 0.

        # calc
        def run_test(flow_reduction, non_retained_species, relative_losses):
            self.uo.flow_reduction = flow_reduction
            self.uo.non_retained_species = non_retained_species
            self.uo.relative_losses = relative_losses
            f_target = self.uo._f.copy() / flow_reduction
            c_target = self.uo._c.copy()
            retentate_list = list(set(range(self.uo._n_species))
                                  - set(non_retained_species))
            c_target[retentate_list] *= flow_reduction * (1 - relative_losses)
            self.uo._calculate()
            np.testing.assert_array_almost_equal(f_target, self.uo._f)
            np.testing.assert_array_almost_equal(c_target, self.uo._c)

        run_test(1, [0], 1)
        run_test(2.3, [0, 1], 1)
        run_test(1, [2], 0)
        run_test(1.6, [0, 2], 0)
        run_test(1, [1], 0.2)
        run_test(1.5, [1], 0.2)


class TestBufferExchange(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 230)
        self.exchange_ratio = 0.95
        self.uo_id = "buffer_exchange"
        self.gui_title = "Buffer Exchange Step"
        self.uo = fc_uo.BufferExchange(
            self.t, self.exchange_ratio, self.uo_id, self.gui_title
        )
        self.uo.log = TestLogger()

    # noinspection DuplicatedCode
    def test_init(self):
        # assert proper exchange_ratio ( 1 >= exchange_ratio > 0)
        with self.assertRaises(AssertionError):
            fc_uo.BufferExchange(self.t, 0, self.uo_id)
        with self.assertRaises(AssertionError):
            fc_uo.BufferExchange(self.t, 1.01, self.uo_id)
        fc_uo.BufferExchange(self.t, 1.0, self.uo_id)

        # assert bindings
        np.testing.assert_array_equal(self.t, self.uo._t)
        self.assertEqual(self.exchange_ratio, self.uo.exchange_ratio)
        self.assertEqual(self.uo_id, self.uo.uo_id)
        self.assertEqual(self.gui_title, self.uo.gui_title)

        # assert default values
        self.assertTrue(self.uo.c_exchange_buffer.size == 0)
        self.assertEqual([], self.uo.non_retained_species)
        self.assertEqual(0, self.uo.relative_losses)

    def test_calc_c_exchange_buffer(self):
        # prepare
        self.uo._n_species = 4

        # assert proper c_add size
        with self.assertRaises(AssertionError):
            self.uo.c_exchange_buffer = np.array([2])
            self.uo._calc_c_exchange_buffer()

        # calc
        # empty -> zero
        self.uo.c_exchange_buffer = np.array([])
        self.uo._calc_c_exchange_buffer()
        np.testing.assert_array_almost_equal(np.zeros([4, 1]), self.uo._c_exchange_buffer)
        # filled -> reshape
        self.uo.c_exchange_buffer = np.array([1, 2, 3, 2])
        self.uo._calc_c_exchange_buffer()
        np.testing.assert_array_almost_equal(self.uo.c_exchange_buffer.reshape(4, 1), self.uo._c_exchange_buffer)
        # filled -> reshape
        self.uo.c_exchange_buffer = np.array([[1], [2], [3], [4]])
        self.uo._calc_c_exchange_buffer()
        np.testing.assert_array_almost_equal(self.uo.c_exchange_buffer, self.uo._c_exchange_buffer)

    # noinspection DuplicatedCode
    def test_calculate(self):
        # prepare
        self.uo._n_species = 4
        self.uo._f = np.ones_like(self.t)
        self.uo._c = np.ones([self.uo._n_species, self.t.size]) * np.array([[1], [2], [3], [4]])
        self.uo.exchange_ratio = 0.95
        self.uo.relative_losses = 0.2
        self.uo._c_exchange_buffer = np.array([[0], [0], [0], [0]])

        # assert valid non_retained_species
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 4]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 0]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [1, 0]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [-1]
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.non_retained_species = [0, 1, 2, 3]
            self.uo._calculate()
        self.uo.non_retained_species = [0]

        # assert valid relative_losses
        with self.assertRaises(AssertionError):
            self.uo.relative_losses = -0.01
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.relative_losses = 1.01
            self.uo._calculate()
        self.uo.relative_losses = 0.

        # assert valid exchange_ratio
        with self.assertRaises(AssertionError):
            self.uo.exchange_ratio = -0.01
            self.uo._calculate()
        with self.assertRaises(AssertionError):
            self.uo.exchange_ratio = 1.01
            self.uo._calculate()
        with self.assertWarns(Warning):
            self.uo.exchange_ratio = 0
            self.uo._calculate()
        self.uo.exchange_ratio = 0.95

        # calc
        def run_test(exchange_ratio, non_retained_species,
                     relative_losses, c_exchange_buffer):
            # prepare
            self.uo.exchange_ratio = exchange_ratio
            self.uo.non_retained_species = non_retained_species
            self.uo.relative_losses = relative_losses
            self.uo.c_exchange_buffer = c_exchange_buffer
            self.uo._calc_c_exchange_buffer()
            # calc reference
            retentate_mask = np.ones(self.uo._n_species, dtype=bool)
            retentate_mask[non_retained_species] = False
            f_target = self.uo._f.copy()
            c_target = self.uo._c * (1 - exchange_ratio)
            c_target[retentate_mask] = \
                self.uo._c[retentate_mask] * (1 - relative_losses)
            c_target += self.uo._c_exchange_buffer * exchange_ratio
            # calc
            self.uo._calculate()
            # asset
            np.testing.assert_array_almost_equal(f_target, self.uo._f)
            np.testing.assert_array_almost_equal(c_target, self.uo._c)

        run_test(1, [0], 1, np.array([]))
        run_test(0.2, [0, 1], 1, np.array([0, 0.2, 0, 0.2]))
        with self.assertWarns(Warning):
            run_test(0, [2], 0, np.array([1, 0.3, 0.4, 1]))
        with self.assertWarns(Warning):
            run_test(0, [0, 2], 1, np.array([]))
        with self.assertWarns(Warning):
            run_test(0, [2], 0.5, np.array([1, 0.5, 0.4, 1]))
        run_test(0.3, [0, 2], 0, np.array([2, 2, 2, 2]))
        run_test(0.6, [2, 3], 0.2, np.array([]))
        run_test(0.5, [], 0.2, np.array([0.2, 0.1, 0.2, 0.3]))


class TestFlowThrough(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 230)
        self.pdf = pdf.GaussianFixedDispersion(self.t, 0.2)
        self.uo_id = "ft_uo"
        self.gui_title = "Flow-Through Unit Operation"
        self.uo = fc_uo.FlowThrough(self.t, self.pdf, self.uo_id, self.gui_title)
        self.uo.log = TestLogger()
        self.uo._f = np.ones_like(self.t) * 0.53
        self.uo._n_species = 2
        self.uo._c = np.ones([self.uo._n_species, self.t.size]) * np.array([[0.2], [0.7]])

    # noinspection DuplicatedCode
    def test_init(self):
        # assert bindings
        np.testing.assert_array_equal(self.t, self.uo._t)
        self.assertEqual(self.pdf, self.uo.pdf)
        self.assertEqual(self.uo_id, self.uo.uo_id)
        self.assertEqual(self.gui_title, self.uo.gui_title)

        # assert default values
        self.assertEqual(-1, self.uo.v_void)
        self.assertEqual(-1, self.uo.rt_target)
        self.assertEqual(-1, self.uo.v_init)
        self.assertEqual(-1, self.uo.v_init_ratio)
        self.assertTrue(self.uo.c_init.size == 0)
        self.assertTrue(len(self.uo.losses_species_list) == 0)
        self.assertEqual(0, self.uo.losses_share)

    # noinspection DuplicatedCode
    def test_calc_v_void(self):
        # no default
        with self.assertRaises(AssertionError):
            self.uo._calc_v_void()

        # rt_target
        self.uo.rt_target = 1.2
        self.uo._calc_v_void()
        self.assertEqual(self.uo.rt_target * self.uo._f.max(), self.uo._v_void)
        self.uo.rt_target = -1

        # v_void
        self.uo.v_void = 1.2
        self.uo._calc_v_void()
        self.assertEqual(self.uo.v_void, self.uo._v_void)

        # both
        self.uo.rt_target = 1.2
        self.uo.v_void = 3.2
        with self.assertWarns(Warning):
            self.uo._calc_v_void()
            # use v_void
            self.assertEqual(self.uo.v_void, self.uo._v_void)

    # noinspection DuplicatedCode
    def test_calc_v_init(self):
        # default: `_v_init = _v_void`
        with self.assertRaises(AssertionError):
            self.uo._calc_v_init()
        self.uo._v_void = 1.2
        self.uo._calc_v_init()
        self.assertEqual(self.uo._v_void, self.uo._v_init)

        # v_init_ratio
        self.uo.v_init_ratio = 0.2
        self.uo._v_void = -1
        with self.assertRaises(AssertionError):
            self.uo._calc_v_init()
        self.uo._v_void = 1.3
        self.uo._calc_v_init()
        self.assertEqual(self.uo._v_void * self.uo.v_init_ratio, self.uo._v_init)
        self.uo._v_void = -1

        # v_init
        self.uo.v_init = 35.2
        with self.assertWarns(Warning):
            # priority over v_init_ratio
            self.uo._calc_v_init()
            self.assertEqual(self.uo.v_init, self.uo._v_init)
        self.uo.v_init_ratio = -1
        self.uo._calc_v_init()
        self.assertEqual(self.uo.v_init, self.uo._v_init)
        self.uo.v_init = 0
        self.uo._calc_v_init()
        self.assertEqual(self.uo.v_init, self.uo._v_init)

    # noinspection DuplicatedCode
    def test_calc_c_init(self):
        # prepare
        self.uo._n_species = 2

        # default
        self.uo._calc_c_init()
        np.testing.assert_array_equal(np.array([[0], [0]]), self.uo._c_init)

        # defined
        self.uo.c_init = np.array([[2.2], [0.3]])
        self.uo._calc_c_init()
        np.testing.assert_array_equal(np.array([[2.2], [0.3]]), self.uo._c_init)

        # defined 2
        self.uo.c_init = np.array([3.2, 0.2])
        self.uo._calc_c_init()
        np.testing.assert_array_equal(np.array([[3.2], [0.2]]), self.uo._c_init)

        # defined wrong
        self.uo.c_init = np.array([3.2])
        with self.assertRaises(AssertionError):
            self.uo._calc_c_init()

    def test_calc_p(self):
        self.uo._v_void = 11.2
        # assert _v_void
        self.assert_defined_values("_v_void", self.uo._sim_init_fill_up)

        # run
        self.assertFalse(hasattr(self.uo, '_p'))  # _p undefined
        self.assertTrue(self.pdf._pdf.size == 0)  # pdf not updated
        self.uo._calc_p()
        self.assertTrue(hasattr(self.uo, '_p'))  # _p defined
        self.assertTrue(self.pdf._pdf.size > 0)  # pdf updated
        np.testing.assert_array_equal(self.pdf.get_p(), self.uo._p)

    def test_pre_calc(self):
        # prepare
        self.uo.v_void = 11.2
        # assert undefined
        self.assertFalse(hasattr(self.uo, "_v_void"))
        self.assertFalse(hasattr(self.uo, "_v_init"))
        self.assertFalse(hasattr(self.uo, "_c_init"))
        self.assertFalse(hasattr(self.uo, "_p"))
        # run
        self.uo._pre_calc()
        # assert defined
        self.assertTrue(hasattr(self.uo, "_v_void"))
        self.assertTrue(hasattr(self.uo, "_v_init"))
        self.assertTrue(hasattr(self.uo, "_c_init"))
        self.assertTrue(hasattr(self.uo, "_p"))
        # numeric values
        self.assertEqual(11.2, self.uo._v_void)
        self.assertEqual(11.2, self.uo._v_init)
        self.assertTrue(self.uo._c_init.sum() == 0)
        np.testing.assert_array_equal(self.pdf._pdf, self.uo._p)

    def assert_defined_values(self, par_name, func):
        # assert values
        v = getattr(self.uo, par_name)
        delattr(self.uo, par_name)
        with self.assertRaises(AssertionError):
            func()
        setattr(self.uo, par_name, v)

    def test_sim_init_fill_up(self):
        # prepare
        self.uo._v_void = 15.5
        self.uo._v_init = 5.5
        self.uo._c_init = np.array([[0], [0.2]])

        self.assert_defined_values("_v_void", self.uo._sim_init_fill_up)
        self.assert_defined_values("_v_init", self.uo._sim_init_fill_up)
        self.assert_defined_values("_c_init", self.uo._sim_init_fill_up)

        # skip if `_v_init == _v_void`
        self.uo._v_init = self.uo._v_void
        c = self.uo._c.copy()
        f = self.uo._f.copy()
        c_init = self.uo._c_init
        self.uo._sim_init_fill_up()
        np.testing.assert_array_equal(c_init, self.uo._c_init)
        np.testing.assert_array_equal(f, self.uo._f)
        np.testing.assert_array_equal(c, self.uo._c)

        # set v_init to < v_void
        self.uo._v_init = 5.5
        # target
        vi = self.uo._v_init
        vv = self.uo._v_void
        c_init = (c_init * vi + c[:, 0:1] * (vv - vi)) / vv
        # noinspection PyTypeChecker
        i_v = vectors.true_start(np.cumsum(self.uo._f) * self.uo._dt >= vv - vi)
        f[:i_v] = 0
        c[:, :i_v] = 0
        # run sim
        self.uo._sim_init_fill_up()
        # compare results
        np.testing.assert_array_almost_equal(c_init, self.uo._c_init, 2)
        np.testing.assert_array_equal(f, self.uo._f)
        np.testing.assert_array_equal(c, self.uo._c)

    def test_sim_convolution(self):
        # prepare
        self.uo._c_init = np.array([[0.4], [0.7]])
        self.uo._v_void = 11.4
        self.uo._calc_p()
        self.uo._f[:30] = 0

        # assert non-missing parameters
        self.assert_defined_values("_c_init", self.uo._sim_convolution)
        self.assert_defined_values("_p", self.uo._sim_convolution)
        # assert box shaped flow
        self.uo._f[:30] = 0.98
        with self.assertRaises(AssertionError):
            self.uo._sim_convolution()
        self.uo._f[:30] = 0

        # target
        f_target = self.uo._f.copy()
        c_target = self.uo._c.copy()
        c_target[:, :30] = 0
        c_target[:, self.uo._f > 0] = convolution.time_conv(
            self.t[1], self.uo._c[:, self.uo._f > 0], self.uo._p, self.uo._c_init, logger=self.uo.log
        )
        self.uo._sim_convolution()
        # check results
        np.testing.assert_array_almost_equal(f_target, self.uo._f)
        np.testing.assert_array_almost_equal(c_target, self.uo._c)

        # make sure logger is passed to convolution function
        self.uo._p = np.array([])
        with self.assertWarns(Warning):
            self.uo._sim_convolution()

    def test_sim_losses(self):
        c_in = self.uo._c.copy()

        # default -> no changes
        self.uo._sim_losses()
        np.testing.assert_array_equal(c_in, self.uo._c)

        # define parameters for loss
        self.uo.losses_share = 0.1
        self.uo.losses_species_list = [1]

        # assert proper losses_species_list
        with self.assertRaises(AssertionError):
            self.uo.losses_species_list = [1, 0]
            self.uo._sim_losses()
        with self.assertRaises(AssertionError):
            self.uo.losses_species_list = [0, 0]
            self.uo._sim_losses()
        with self.assertRaises(AssertionError):
            self.uo.losses_species_list = [2]
            self.uo._sim_losses()
        with self.assertRaises(AssertionError):
            self.uo.losses_species_list = [-1]
            self.uo._sim_losses()
        self.uo.losses_species_list = [1]
        # assert proper losses_share
        with self.assertRaises(AssertionError):
            self.uo.losses_share = 1.01
            self.uo._sim_losses()
        with self.assertRaises(AssertionError):
            self.uo.losses_share = -0.01
            self.uo._sim_losses()
        self.uo.losses_share = 0.1

        def run_sim_losses(losses_share, losses_species_list):
            self.uo.losses_share = losses_share
            self.uo.losses_species_list = losses_species_list
            c_ref = self.uo._c.copy()
            c_ref[losses_species_list, :] *= 1 - self.uo.losses_share
            self.uo._sim_losses()
            np.testing.assert_array_almost_equal(c_ref, self.uo._c)

        # test 1
        run_sim_losses(0.2, [1])
        run_sim_losses(0, [1])
        run_sim_losses(1, [1])
        run_sim_losses(0.7, [0, 1])
        run_sim_losses(0.4, [0])

    def test_calculate(self):
        # prepare
        self.uo.v_void = 15.2
        self.uo.v_init = 3.2
        self.uo.losses_share = 0.2
        self.uo.losses_species_list = [1]

        self.uo._calculate()

        self.assertTrue(hasattr(self.uo, '_v_void'))
        self.assertTrue(hasattr(self.uo, '_v_init'))
        self.assertTrue(hasattr(self.uo, '_c_init'))
        self.assertTrue(hasattr(self.uo, '_p'))

        self.assertAlmostEqual(0.56, self.uo._c.max())
        self.assertAlmostEqual(0.2726, self.uo._c.mean(), 5)
        self.assertEqual(51, vectors.true_start(self.uo._f > 0))


class TestFlowThroughWithSwitching(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 230)
        self.pdf = pdf.GaussianFixedDispersion(self.t, 0.2)
        self.uo_id = "ft_ws"
        self.gui_title = "Flow-Through with switching"
        self.uo = fc_uo.FlowThroughWithSwitching(self.t, self.pdf, self.uo_id, self.gui_title)
        self.uo.log = TestLogger()
        self.uo._f = np.ones_like(self.t) * 0.53
        self.uo._n_species = 2
        self.uo._c = np.ones([self.uo._n_species, self.t.size]) * np.array([[0.2], [0.7]])

    def assert_defined_values(self, par_name, func):
        # assert values
        v = getattr(self.uo, par_name)
        delattr(self.uo, par_name)
        with self.assertRaises(AssertionError):
            func()
        setattr(self.uo, par_name, v)

    # noinspection DuplicatedCode
    def test_init(self):
        # assert super class
        self.assertTrue(isinstance(self.uo, fc_uo.FlowThrough))

        # assert bindings
        np.testing.assert_array_equal(self.t, self.uo._t)
        self.assertEqual(self.pdf, self.uo.pdf)
        self.assertEqual(self.uo_id, self.uo.uo_id)
        self.assertEqual(self.gui_title, self.uo.gui_title)

        # assert default values
        self.assertEqual(-1, self.uo.t_cycle)
        self.assertEqual(-1, self.uo.v_cycle)
        self.assertEqual(-1, self.uo.v_cycle_relative)

    # noinspection DuplicatedCode
    def test_calc_t_cycle(self):
        # prepare
        self.uo._v_void = 11.2

        # no default
        with self.assertRaises(AssertionError):
            self.uo._calc_t_cycle()
        # set some value
        self.uo.t_cycle = 1.2

        # _v_void needed
        self.assert_defined_values('_v_void', self.uo._calc_t_cycle)

        # t_cycle
        self.uo.t_cycle = 1.2
        self.uo._calc_t_cycle()
        self.assertEqual(self.uo.t_cycle, self.uo._t_cycle)
        self.uo.t_cycle = -1

        # v_cycle
        self.uo.v_cycle = 1.2
        self.uo._calc_t_cycle()
        self.assertEqual(self.uo.v_cycle / self.uo._f.max(), self.uo._t_cycle)
        self.uo.v_cycle = -1

        # v_cycle_relative
        self.uo.v_cycle_relative = 0.2
        self.uo._calc_t_cycle()
        self.assertEqual(self.uo.v_cycle_relative * self.uo._v_void / self.uo._f.max(), self.uo._t_cycle)
        self.uo.v_cycle_relative = -1

        # t_cycle and v_cycle (t_cycle has priority)
        self.uo.t_cycle = 1.2
        self.uo.v_cycle = 3.2
        with self.assertWarns(Warning):
            self.uo._calc_t_cycle()
            self.assertEqual(self.uo.t_cycle, self.uo._t_cycle)

        # t_cycle and v_cycle and v_cycle_relative (t_cycle has priority)
        self.uo.v_cycle_relative = 0.2
        with self.assertWarns(Warning):
            self.uo._calc_t_cycle()
            self.assertEqual(self.uo.t_cycle, self.uo._t_cycle)

        # v_cycle and v_cycle_relative (v_cycle has priority)
        self.uo.t_cycle = -1
        with self.assertWarns(Warning):
            self.uo._calc_t_cycle()
            self.assertEqual(self.uo.v_cycle / self.uo._f.max(), self.uo._t_cycle)

    def test_sim_piece_wise_convolution(self):
        # prepare
        self.uo._c_init = np.array([[0.4], [0.7]])
        self.uo._v_void = 11.4
        self.uo._calc_p()
        self.uo._f[:30] = 0
        self.uo._t_cycle = 35.5

        # assert non-missing parameters
        self.assert_defined_values("_v_void", self.uo._sim_piece_wise_convolution)
        self.assert_defined_values("_c_init", self.uo._sim_piece_wise_convolution)
        self.assert_defined_values("_p", self.uo._sim_piece_wise_convolution)
        self.assert_defined_values("_t_cycle", self.uo._sim_piece_wise_convolution)
        # assert box shaped flow
        self.uo._f[:30] = 0.98
        with self.assertRaises(AssertionError):
            self.uo._sim_piece_wise_convolution()
        self.uo._f[:30] = 0

        # target
        f_target = self.uo._f.copy()
        c_target = convolution.piece_wise_conv_with_init_state(
            self.t[1], f_in=self.uo._f, c_in=self.uo._c,
            t_cycle=self.uo._t_cycle, rt_mean=self.uo._v_void / self.uo._f.max(),
            rtd=self.uo._p, c_wash=self.uo._c_init, logger=self.uo.log
        )
        self.uo._sim_piece_wise_convolution()
        # check results
        np.testing.assert_array_almost_equal(f_target, self.uo._f)
        np.testing.assert_array_almost_equal(c_target, self.uo._c)

        # make sure logger is passed to convolution function
        self.uo._p = np.array([])
        with self.assertWarns(Warning):
            self.uo._sim_piece_wise_convolution()

    def test_calculate(self):
        # prepare
        self.uo.v_void = 15.2
        self.uo.v_init = 3.2
        self.uo.losses_share = 0.2
        self.uo.losses_species_list = [1]
        self.uo.t_cycle = 34.3

        self.uo._calculate()

        self.assertTrue(hasattr(self.uo, '_v_void'))
        self.assertTrue(hasattr(self.uo, '_v_init'))
        self.assertTrue(hasattr(self.uo, '_c_init'))
        self.assertTrue(hasattr(self.uo, '_p'))
        self.assertTrue(hasattr(self.uo, '_t_cycle'))

        self.assertAlmostEqual(0.56, self.uo._c.max())
        self.assertAlmostEqual(0.17918, self.uo._c.mean(), 5)
        self.assertEqual(51, vectors.true_start(self.uo._f > 0))
