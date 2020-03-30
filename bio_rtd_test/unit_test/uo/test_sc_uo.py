import numbers
import unittest

import numpy as np
import scipy.interpolate as sc_interp
from bio_rtd import pdf, utils, chromatography
from bio_rtd.uo import sc_uo
from bio_rtd_test.aux_bio_rtd_test import TestLogger


def set_ac_parameters(ac: sc_uo.AlternatingChromatography):
    """ Set CV """
    ac.cv = 14.5

    """ Define one """
    ac.load_cv = 30  # preferred
    # ac.load_c_end_ss = -1
    # ac.load_c_end_relative_ss = -1
    # ac.load_c_end_estimate_iterative_solver = -1  # takes longer to calc load_cv but is more precise

    """ Optional """
    ac.load_extend_first_cycle = False
    ac.load_extend_first_cycle_cv = -1

    """ Values are added if both defined """
    ac.wash_cv = 3
    # ac.wash_t = -1

    """ Define one """
    # ac.wash_f = -1
    ac.wash_f_rel = 1  # relative to the load flow rate

    """ Values are added if both defined """
    ac.elution_cv = 12
    # ac.elution_t = -1

    """ Define one """
    # ac.elution_f = -1
    ac.elution_f_rel = 1  # relative to the load flow rate

    """ If empty -> values are 0"""
    ac.elution_buffer_c = np.array([0])

    """ Define one; The value corresponds to 1st moment (and not necessarily peak max)"""
    ac.elution_peak_position_cv = 1.2
    # ac.elution_peak_position_t = -1

    """ Define one"""
    # ac.elution_peak_cut_start_t = -1
    # ac.elution_peak_cut_start_cv = -1
    # ac.elution_peak_cut_start_c_rel_to_peak_max = -1
    ac.elution_peak_cut_start_peak_area_share = 0.05

    """ Define one"""
    # ac.elution_peak_cut_end_t = -1
    # ac.elution_peak_cut_end_cv = -1
    # ac.elution_peak_cut_end_c_rel_to_peak_max = -1
    ac.elution_peak_cut_end_peak_area_share = 0.10

    """ Values are added if both defined """
    ac.regeneration_cv = -1
    ac.regeneration_t = -1

    """ Define one"""
    ac.regeneration_f = -1
    ac.regeneration_f_rel = 1  # relative to the load flow rate


def get_bt_constant_pattern_solution(dt, dbc_100=240, k=0.2):
    load_bt = chromatography.bt_load.ConstantPatternSolution(
        dt, dbc_100=dbc_100, k=k, bt_profile_id="Constant Pattern Solution"
    )
    return load_bt


def get_pdf_gaussian_fixed_dispersion(t, dispersion_index):
    gfd = pdf.GaussianFixedDispersion(t, dispersion_index=dispersion_index)
    return gfd


def get_pdf_gaussian_fixed_relative_width(t, relative_sigma=0.2):
    return pdf.GaussianFixedRelativeWidth(t, relative_sigma=relative_sigma)


def prep_for_calc_cycle_t(t, ac: sc_uo.AlternatingChromatography):
    f = np.zeros_like(t)
    i_f_start = int(round(t[-1] / 10))
    i_f_end = int(round(4 * t[-1] / 5))
    f[i_f_start:i_f_end] = 3.5
    load_c = 2.5
    c = load_c * np.ones([3, t.size])
    i_c_trace_start = int(round(3 * t[-1] / 10))
    i_c_trace_end = int(round(4 * t[-1] / 10))
    c[1, :] = 0
    c[0, i_c_trace_start:i_c_trace_end] = 0
    c[1, i_c_trace_start:i_c_trace_end] = load_c
    ac._f = f.copy()
    ac._c = c.copy()
    ac._n_species = 3
    # define non-binding species
    ac.non_binding_species = [2]
    # define wash
    ac.wash_cv = 15
    # define leftover propagation dynamics
    ac.column_porosity_retentate = 0.64
    ac.load_recycle_pdf = get_pdf_gaussian_fixed_dispersion(t, 10)
    # define flow switch criteria
    ac.load_c_end_relative_ss = 0.7
    # turn off high sensitive solver by default
    ac.load_c_end_estimate_with_iterative_solver = False


class TestAlternatingChromatography(unittest.TestCase):

    def assert_defined_value(self, par_name, func, *args):
        v = getattr(self.uo, par_name)
        setattr(self.uo, par_name, -1 if isinstance(v, numbers.Number) else None)
        with self.assertRaises(AssertionError):
            func(*args)
        setattr(self.uo, par_name, v)

    def setUp(self) -> None:
        self.t = np.linspace(0, 2000, 20000)
        self.dt = self.t[1] - self.t[0]
        self.uo = sc_uo.AlternatingChromatography(
            t=self.t,
            uo_id="AlternatingChromatography_Test",
            load_bt=get_bt_constant_pattern_solution(self.dt, dbc_100=240, k=0.2),
            peak_shape_pdf=get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=100)
        )

        """ Set logger (optional) """
        self.log = TestLogger()
        self.uo.log = self.log

        """ Set other parameters """
        set_ac_parameters(self.uo)

    def test_get_flow_value(self):
        # set load flow rate
        load_f = 9.81
        self.uo._load_f = load_f
        # make sure it sends warnings and returns load_f if f < 0 and f_rel < 0
        with self.assertWarns(Warning):
            self.assertEqual(self.uo._get_flow_value("", "", -1, -1), load_f)
        with self.assertWarns(Warning):
            self.assertEqual(self.uo._get_flow_value("", "", 0, 0), load_f)
        # make sure it returns f
        self.assertEqual(self.uo._get_flow_value("", "", 0.15, -1), 0.15)
        self.assertEqual(self.uo._get_flow_value("", "", 12.15, 1), 12.15)
        # make sure it returns rf * load_f
        self.assertEqual(self.uo._get_flow_value("", "", 0, 0.02), 0.02 * load_f)
        self.assertEqual(self.uo._get_flow_value("", "", -1, 2.02), 2.02 * load_f)

    def test_get_time_value(self):
        # make sure it sends warnings and returns load_f if f < 0 and f_rel < 0
        with self.assertRaises(AssertionError):
            self.uo._get_time_value("", "", -1, 1, 0)
        with self.assertRaises(AssertionError):
            self.uo._get_time_value("", "", 5, 1, 0)
        with self.assertWarns(Warning):
            self.assertEqual(self.uo._get_time_value("", "", 0, 0, 0), 0)
        with self.assertWarns(Warning):
            self.assertEqual(self.uo._get_time_value("", "", -1, -1, -1), 0)
        # make sure it returns t
        self.assertEqual(self.uo._get_time_value("", "", 0.15, -1, 0), 0.15)
        self.assertEqual(self.uo._get_time_value("", "", 12.15, 0, 15), 12.15)
        # make sure it returns CV
        cv = 10
        self.uo._cv = cv
        self.assertEqual(self.uo._get_time_value("", "", 0, 0.02, 4), 0.02 * cv / 4)
        self.assertEqual(self.uo._get_time_value("", "", -1, 2.02, 3), 2.02 * cv / 3)

    def test_assert_non_binding_species(self):
        self.uo._n_species = 1
        self.uo.non_binding_species = [0]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo._n_species = 3
        self.uo.non_binding_species = [3]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [0, 0]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo.non_binding_species = []
        self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [1]
        self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [0, 2]
        self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [0, 1, 2]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [1, 3]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [2, 1]
        with self.assertRaises(AssertionError):
            self.uo._assert_non_binding_species()
        self.uo.non_binding_species = [0, 2]
        self.uo._assert_non_binding_species()

    def test_calc_load_f(self):
        # zeros
        f = np.zeros_like(self.t)
        self.uo._f = f
        with self.assertRaises(AssertionError):
            with self.assertWarns(Warning):
                self.uo._calc_load_f()
        # ones
        f = np.ones_like(self.t)
        self.uo._f = f
        self.uo._calc_load_f()
        self.assertEqual(f[0], self.uo._load_f)
        # negative
        f[-1] = -1
        self.uo._f = f
        with self.assertRaises(AssertionError):
            self.uo._calc_load_f()
        # box shape
        f[-10:] = 0
        f[:10] = 0
        self.uo._f = f
        self.uo._calc_load_f()
        self.assertEqual(f.max(), self.uo._load_f)

    def test_calc_cv(self):
        # reset state
        def reset_state():
            self.uo.cv = -1
            self.uo.column_porosity_retentate = -1
            self.uo.ft_mean_retentate = -1
            self.uo._load_f = -1

        # cv
        reset_state()
        cv_ref = 342.34  # target
        self.uo.cv = cv_ref
        self.uo._calc_cv()
        self.assertEqual(cv_ref, self.uo._cv)
        # flow-through
        reset_state()
        ft_time = 33.3
        load_flow = 2.2
        protein_porosity = 0.85
        cv_ref = ft_time * load_flow / protein_porosity  # target
        self.uo.ft_mean_retentate = ft_time
        self.uo.column_porosity_retentate = protein_porosity
        self.uo._load_f = load_flow
        self.uo._calc_cv()
        self.assertEqual(cv_ref, self.uo._cv)

    def test_report_column_dimensions(self):
        cv = 2.5
        f = 2.7
        v_lin = 12.1
        self.uo._cv = cv
        self.uo.load_target_lin_velocity = v_lin
        self.uo._load_f = f
        self.uo._report_column_dimensions()
        self.assertTrue(self.uo._col_h, cv / f * v_lin)

    # noinspection DuplicatedCode
    def test_calc_equilibration_t(self):
        def reset():
            self.uo.equilibration_cv = -1
            self.uo.equilibration_f = -1
            self.uo.equilibration_f_rel = -1
            self.uo.equilibration_t = -1
            self.uo.equilibration_cv = -1
            self.uo._cv = -1
            self.uo._load_f = -1

        reset()
        t = 12
        self.uo.equilibration_t = t
        self.uo._calc_equilibration_t()
        self.assertEqual(t, self.uo._equilibration_t)

        reset()
        t = 12
        l_cv = 2

        self.uo.equilibration_t = t
        self.uo.equilibration_cv = l_cv
        with self.assertWarns(Warning):  # due to missing equilibration_f
            with self.assertRaises(AssertionError):
                self.uo._calc_equilibration_t()

        f_load = 2.2
        self.uo._load_f = f_load
        with self.assertWarns(Warning):  # due to missing equilibration_f
            with self.assertRaises(AssertionError):
                self.uo._calc_equilibration_t()

        cv = 2.7
        self.uo._cv = cv
        with self.assertWarns(Warning):  # due to missing equilibration_f
            self.uo._calc_equilibration_t()
            self.assertEqual(t + l_cv * cv / f_load, self.uo._equilibration_t)

        f = 3.4
        self.uo.equilibration_f = f
        self.uo._calc_equilibration_t()
        self.assertEqual(t + l_cv * cv / f, self.uo._equilibration_t)

    # noinspection DuplicatedCode
    def test_calc_wash_t_and_f(self):

        # just makes sure it calls `_get_flow_value` which is already tested
        self.uo.wash_f = 1.2
        self.uo.wash_t = 21.1
        self.uo.wash_cv = -1
        self.uo._wash_f = -1
        self.uo._cv = -1
        self.uo._load_f = -1
        self.uo.wash_f_rel = -1
        self.uo._calc_wash_t_and_f()
        self.assertEqual(self.uo.wash_f, self.uo._wash_f)
        self.assertEqual(self.uo.wash_t, self.uo._wash_t)

        self.uo.wash_f = -1
        self.uo.wash_t = 21.1
        self.uo.wash_cv = 0.9
        self.uo._cv = 1.3
        self.uo.wash_f_rel = 0.5
        self.uo._load_f = 1.7
        self.uo._calc_wash_t_and_f()
        self.assertEqual(self.uo.wash_f_rel * self.uo._load_f, self.uo._wash_f)
        self.assertEqual(
            self.uo.wash_t + self.uo.wash_cv * self.uo._cv / self.uo._wash_f,
            self.uo._wash_t
        )

    # noinspection DuplicatedCode
    def test_calc_elution_t_and_f(self):

        # just makes sure it calls `_get_flow_value` which is already tested
        self.uo.elution_f = 1.2
        self.uo.elution_t = 21.1
        self.uo.elution_cv = -1
        self.uo._elution_f = -1
        self.uo._cv = -1
        self.uo._load_f = -1
        self.uo.elution_f_rel = -1
        self.uo._calc_elution_t_and_f()
        self.assertEqual(self.uo.elution_f, self.uo._elution_f)
        self.assertEqual(self.uo.elution_t, self.uo._elution_t)

        self.uo.elution_f = -1
        self.uo.elution_t = 21.1
        self.uo.elution_cv = 0.9
        self.uo._cv = 1.3
        self.uo.elution_f_rel = 0.5
        self.uo._load_f = 1.7
        self.uo._calc_elution_t_and_f()
        self.assertEqual(self.uo.elution_f_rel * self.uo._load_f, self.uo._elution_f)
        self.assertEqual(
            self.uo.elution_t + self.uo.elution_cv * self.uo._cv / self.uo._elution_f,
            self.uo._elution_t
        )

    def test_calc_elution_peak_t(self):

        # just makes sure it calls `_get_time_value` which is already tested
        self.uo.elution_peak_position_t = 1.2
        self.uo.elution_peak_position_cv = -1
        self.uo._elution_f = -1
        self.uo._calc_elution_peak_t()
        self.assertEqual(self.uo.elution_peak_position_t, self.uo._elution_peak_t)

        self.uo.elution_peak_position_t = 1.2
        self.uo.elution_peak_position_cv = 0.6
        self.uo._elution_f = 1.43
        self.uo._cv = 0.15
        self.uo._calc_elution_peak_t()
        self.assertEqual(
            self.uo.elution_peak_position_t +
            self.uo.elution_peak_position_cv * self.uo._cv / self.uo._elution_f,
            self.uo._elution_peak_t
        )

    def test_update_load_btc(self):
        # make sure it calls `elution_peak_shape.update_pdf`

        dbc_100 = 35
        k = 0.0022

        btc = get_bt_constant_pattern_solution(self.dt, dbc_100=dbc_100, k=k)
        self.uo.load_bt = get_bt_constant_pattern_solution(self.dt, dbc_100=dbc_100, k=k)

        self.uo._cv = 27
        btc.update_btc_parameters(cv=self.uo._cv)
        self.assertNotEqual(btc.cv, self.uo.load_bt.cv)

        self.uo._update_load_btc()
        self.assertEqual(btc.cv, self.uo.load_bt.cv)

    def test_update_elution_peak_pdf(self):
        # prepare
        self.uo._elution_peak_t = 26.7
        self.uo._elution_f = 3.5

        self.assertFalse(hasattr(self.uo, '_p_elution_peak'))

        self.assert_defined_value('_elution_peak_t',
                                  self.uo._update_elution_peak_pdf)
        self.assert_defined_value('_elution_f',
                                  self.uo._update_elution_peak_pdf)

        self.uo._update_elution_peak_pdf()

        np.testing.assert_array_equal(
            self.uo.elution_peak_shape.get_p(),
            self.uo._p_elution_peak
        )

        # add losses
        self.uo.unaccounted_losses_rel = 0.15
        self.uo._update_elution_peak_pdf()
        np.testing.assert_array_equal(
            self.uo.elution_peak_shape.get_p()
            * (1 - self.uo.unaccounted_losses_rel),
            self.uo._p_elution_peak
        )

    def test_calc_elution_peak_cut_i_start_and_i_end(self):

        def reset_peak_cut_start():
            self.uo.elution_peak_cut_start_t = -1
            self.uo.elution_peak_cut_start_cv = -1
            self.uo.elution_peak_cut_start_peak_area_share = -1
            self.uo.elution_peak_cut_start_c_rel_to_peak_max = -1

        def reset_peak_cut_end():
            self.uo.elution_peak_cut_end_t = -1
            self.uo.elution_peak_cut_end_cv = -1
            self.uo.elution_peak_cut_end_peak_area_share = -1
            self.uo.elution_peak_cut_end_c_rel_to_peak_max = -1

        # prepare
        self.uo._elution_peak_t = 5
        self.uo.elution_peak_shape.update_pdf(rt_mean=self.uo._elution_peak_t)
        self.uo._p_elution_peak = self.uo.elution_peak_shape.get_p()
        peak_pdf = self.uo._p_elution_peak.copy()
        # set inlet concentration and flow
        self.uo._c = np.ones([3, self.t.size])
        self.uo._c[1] = 0
        self.uo._c[2] = 2.3
        self._f = np.ones_like(self.t) * 5
        i_delay = int(round(15 / self.dt))
        self.uo._c[:, :i_delay] = 0
        self._f[:i_delay] = 0
        # set load flow rate
        self.uo._load_f = self._f.max()
        # set elution flow to be * 3 the loading flow
        f_elution = 3 * self.uo._load_f
        self.uo._elution_f = f_elution
        self.uo._elution_t = 400

        # ====== PRE-CHECK =====
        # error if defined in more than one way
        reset_peak_cut_start()
        reset_peak_cut_end()
        self.uo.elution_peak_cut_start_t = 1
        self.uo.elution_peak_cut_start_cv = 1
        with self.assertRaises(RuntimeError):
            self.uo._calc_elution_peak_cut_i_start_and_i_end()

        def run_and_assert(i_st_target, i_e_target):
            self.uo._calc_elution_peak_cut_i_start_and_i_end()
            self.assertEqual(self.uo._elution_peak_cut_start_i, i_st_target)
            self.assertEqual(self.uo._elution_peak_cut_end_i, i_e_target)

        # ====== TEST =====

        # reset
        reset_peak_cut_start()
        reset_peak_cut_end()
        # set fixed end
        i_end_target = int(round(self.uo._elution_t - 10 / self.dt))
        self.uo.elution_peak_cut_end_t = self.t[i_end_target]

        # ## test: peak cut start ##
        # start (blank) -> 0 and warning
        with self.assertWarns(Warning):
            run_and_assert(0, i_end_target)
        # start (time)
        reset_peak_cut_start()
        i_start_target = int(round(15 / self.dt))
        self.uo.elution_peak_cut_start_t = self.t[i_start_target]
        run_and_assert(i_start_target, i_end_target)
        # start (CV)
        self.uo._cv = 26
        reset_peak_cut_start()
        self.uo.elution_peak_cut_start_cv = 2
        i_start_target = int(self.uo._cv * self.uo.elution_peak_cut_start_cv / f_elution / self.dt)
        run_and_assert(i_start_target, i_end_target)
        # start (peak area share)
        reset_peak_cut_start()
        self.uo.elution_peak_cut_start_peak_area_share = 0.05
        i_start_target = utils.vectors.true_start(
            np.cumsum(peak_pdf * self.dt) >= self.uo.elution_peak_cut_start_peak_area_share
        )
        run_and_assert(i_start_target, i_end_target)
        # start (c relative to peak max)
        reset_peak_cut_start()
        self.uo.elution_peak_cut_start_c_rel_to_peak_max = 0.15
        i_start_target = utils.vectors.true_start(
            peak_pdf >= peak_pdf.max() * self.uo.elution_peak_cut_start_c_rel_to_peak_max
        )
        run_and_assert(i_start_target, i_end_target)

        # reset
        reset_peak_cut_start()
        reset_peak_cut_end()
        # set fixed start
        i_start_target = int(round(10 / self.dt))
        self.uo.elution_peak_cut_start_t = self.t[i_start_target]

        # ## test: peak cut end ##
        # end (blank) -> 0 and warning
        with self.assertWarns(Warning):
            run_and_assert(i_start_target, peak_pdf.size)
        # end (time)
        reset_peak_cut_end()
        i_end_target = int(round(self.uo._elution_t - 15 / self.dt))
        self.uo.elution_peak_cut_end_t = self.t[i_end_target]
        run_and_assert(i_start_target, i_end_target)
        # end (CV) -> warning on to short elution phase
        self.uo._cv = 24
        reset_peak_cut_end()
        self.uo.elution_peak_cut_end_cv = 2.3
        i_end_target = int(self.uo._cv * self.uo.elution_peak_cut_end_cv / f_elution / self.dt)
        with self.assertWarns(Warning):
            run_and_assert(i_start_target, i_end_target)
        # end (CV)
        self.uo.elution_peak_cut_end_cv = 23
        i_end_target = int(self.uo._cv * self.uo.elution_peak_cut_end_cv / f_elution / self.dt)
        run_and_assert(i_start_target, i_end_target)
        # end (peak area share)
        reset_peak_cut_end()
        self.uo.elution_peak_cut_end_peak_area_share = 0.07
        i_end_target = utils.vectors.true_start(
            np.cumsum(peak_pdf * self.dt) >= (1 - self.uo.elution_peak_cut_end_peak_area_share)
        )
        run_and_assert(i_start_target, i_end_target)
        # end (c relative to peak max)
        reset_peak_cut_end()
        self.uo.elution_peak_cut_end_c_rel_to_peak_max = 0.12
        i_end_target = utils.vectors.true_end(
            peak_pdf >= peak_pdf.max() * self.uo.elution_peak_cut_end_c_rel_to_peak_max
        )
        run_and_assert(i_start_target, i_end_target)
        self.uo._elution_t = i_end_target * self.dt - self.dt
        with self.assertWarns(Warning):
            run_and_assert(i_start_target, i_end_target)

    def test_calc_elution_peak_mask(self):
        self.uo._elution_peak_t = 15.2
        self.uo._elution_f = 1.5
        self.uo._elution_t = 45.34
        peak_mask = np.ones(int(round(self.uo._elution_t / self.dt)), dtype=bool)
        self.uo._elution_peak_cut_start_i = 10
        self.uo._elution_peak_cut_end_i = peak_mask.size - 100
        peak_mask[:self.uo._elution_peak_cut_start_i] = False
        peak_mask[self.uo._elution_peak_cut_end_i:] = False
        self.uo._calc_elution_peak_mask()
        np.testing.assert_array_equal(peak_mask, self.uo._elution_peak_mask)

    def test_update_elution_peak_pdf_2(self):
        dispersion_index = 30
        ep_shape = get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=dispersion_index)
        # make sure it calls `elution_peak_shape.update_pdf`
        self.uo._elution_peak_t = 15.2
        self.uo._elution_f = 1.5
        self.uo.elution_peak_shape = get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=dispersion_index)
        ep_shape.update_pdf(
            v_void=self.uo._elution_peak_t * self.uo._elution_f,
            f=self.uo._elution_f,
            rt_mean=self.uo._elution_peak_t
        )
        self.assertFalse(hasattr(self.uo, '_p_elution_peak'))
        with self.assertRaises(AssertionError):
            self.uo.elution_peak_shape.get_p()
        self.uo._update_elution_peak_pdf()
        np.testing.assert_array_equal(ep_shape.get_p(), self.uo.elution_peak_shape.get_p())
        np.testing.assert_array_equal(ep_shape.get_p(), self.uo._p_elution_peak)

    # noinspection DuplicatedCode
    def test_calc_regeneration_t(self):
        def reset():
            self.uo.regeneration_cv = -1
            self.uo.regeneration_f = -1
            self.uo.regeneration_f_rel = -1
            self.uo.regeneration_t = -1
            self.uo.regeneration_cv = -1
            self.uo._cv = -1
            self.uo._load_f = -1

        reset()
        t = 12
        self.uo.regeneration_t = t
        self.uo._calc_regeneration_t()
        self.assertEqual(t, self.uo._regeneration_t)

        reset()
        t = 12
        l_cv = 2

        self.uo.regeneration_t = t
        self.uo.regeneration_cv = l_cv
        with self.assertWarns(Warning):  # due to missing regeneration_f
            with self.assertRaises(AssertionError):
                self.uo._calc_regeneration_t()

        f_load = 2.2
        self.uo._load_f = f_load
        with self.assertWarns(Warning):  # due to missing regeneration_f
            with self.assertRaises(AssertionError):
                self.uo._calc_regeneration_t()

        cv = 2.7
        self.uo._cv = cv
        with self.assertWarns(Warning):  # due to missing regeneration_f
            self.uo._calc_regeneration_t()
            self.assertEqual(t + l_cv * cv / f_load, self.uo._regeneration_t)

        f = 3.4
        self.uo.regeneration_f = f
        self.uo._calc_regeneration_t()
        self.assertEqual(t + l_cv * cv / f, self.uo._regeneration_t)

    def test_update_load_recycle_pdf(self):
        cv = 14.5
        r_pdf = get_pdf_gaussian_fixed_relative_width(self.t, )
        porosity_retentate = 0.64
        self.uo._cv = cv
        self.uo.load_recycle_pdf = r_pdf
        self.uo.column_porosity_retentate = 0
        with self.assertRaises(AssertionError):
            self.uo._update_load_recycle_pdf(1.5)
        self.uo._cv = cv
        self.uo.load_recycle_pdf = None
        self.uo.column_porosity_retentate = porosity_retentate
        with self.assertRaises(AssertionError):
            self.uo._update_load_recycle_pdf(1.5)
        self.uo._cv = -1
        self.uo.load_recycle_pdf = r_pdf
        self.uo.column_porosity_retentate = porosity_retentate
        with self.assertRaises(AssertionError):
            self.uo._update_load_recycle_pdf(1.5)
        self.uo._cv = cv
        self.uo.load_recycle_pdf = r_pdf
        self.uo.column_porosity_retentate = porosity_retentate
        self.uo._update_load_recycle_pdf(1.5)
        self.assertTrue(self.uo.load_recycle_pdf.get_p().size > 0)

    def test_calc_load_recycle_wash_i(self):
        # if undefined, defaults to entire wash step
        self.uo.wash_recycle_duration_t = -1
        self.uo.wash_recycle_duration_cv = -1
        self.uo._wash_t = 146.3
        self.uo._calc_load_recycle_wash_i()
        i_target = int(round(self.uo._wash_t / self.dt))
        self.assertEqual(i_target, self.uo._wash_recycle_i_duration)

        # just makes sure it calls `_get_time_value` which is already tested
        self.uo.wash_recycle_duration_t = 1.2
        self.uo.wash_recycle_duration_cv = -1
        self.uo._wash_f = 1.43
        self.uo._calc_load_recycle_wash_i()
        i_target = int(self.uo.wash_recycle_duration_t / self.dt)
        self.assertEqual(i_target, self.uo._wash_recycle_i_duration)

        self.uo.wash_recycle_duration_t = 1.2
        self.uo.wash_recycle_duration_cv = 0.6
        self.uo._wash_f = 1.43
        self.uo._cv = 0.15
        self.uo._calc_load_recycle_wash_i()
        i_target = int((self.uo.wash_recycle_duration_t +
                        self.uo.wash_recycle_duration_cv * self.uo._cv / self.uo._wash_f
                        ) / self.dt)
        self.assertEqual(i_target, self.uo._wash_recycle_i_duration)

    def test_get_load_bt_cycle_switch_limit(self):
        self.uo.load_c_end_ss = None
        self.uo.load_c_end_relative_ss = -1
        load_c_ss = np.array([[1.3], [2.2]])

        # assertion
        with self.assertRaises(AssertionError):
            self.uo._get_load_bt_cycle_switch_limit(load_c_ss)

        # warning (specify both)
        self.uo.load_c_end_ss = np.array([[3.9], [9.02]])
        self.uo.load_c_end_relative_ss = 0.6
        with self.assertWarns(Warning):
            np.testing.assert_array_equal(
                self.uo._get_load_bt_cycle_switch_limit(load_c_ss),
                self.uo.load_c_end_ss
            )

        # specify conc
        self.uo.load_c_end_relative_ss = -1
        load_c_end_ss = np.array([[3.7], [8.7]])
        self.uo.load_c_end_ss = load_c_end_ss.copy()
        np.testing.assert_array_equal(self.uo._get_load_bt_cycle_switch_limit(load_c_ss), load_c_end_ss)

        # specify conc ratio
        self.uo.load_c_end_ss = None
        load_c_end_relative_ss = 0.3
        self.uo.load_c_end_relative_ss = load_c_end_relative_ss
        np.testing.assert_array_almost_equal(
            self.uo._get_load_bt_cycle_switch_limit(load_c_ss),
            load_c_end_relative_ss * load_c_ss
        )

    # noinspection DuplicatedCode
    def test_calc_cycle_t(self):
        """
        This test depends on `_sim_c_wash_desorption` and  `_sim_c_recycle_propagation`.

        This is unlike other tests that depend only on functions tested above the test.
        """

        self.uo._cv = 14.3
        self.uo._load_f = 3.5
        self.uo._n_species = 3

        self.uo.load_cv = 2.3
        self.uo.load_c_end_ss = None
        self.uo.load_c_end_relative_ss = -1

        self.assert_defined_value("_cv", self.uo._calc_cycle_t)
        self.assert_defined_value("_load_f", self.uo._calc_cycle_t)
        self.assert_defined_value("load_cv", self.uo._calc_cycle_t)

        # ## test definition by `load_cv` ##
        self.uo.load_cv = 2.3
        self.assertFalse(hasattr(self.uo, "_cycle_t"))
        self.uo._calc_cycle_t()
        self.assertEqual(self.uo._cycle_t, self.uo.load_cv * self.uo._cv / self.uo._load_f)
        # warn if defined in multiple ways
        self.uo.load_c_end_ss = np.array([[1], [0.5]])
        with self.assertWarns(Warning):
            self.uo._calc_cycle_t()
        self.uo.load_c_end_ss = None
        self.uo.load_c_end_relative_ss = 0.7
        with self.assertWarns(Warning):
            self.uo._calc_cycle_t()
        self.uo.load_cv = -1

        # ## test definition by `load_c_end_relative_ss` ##
        # prepare flow and conc profile
        prep_for_calc_cycle_t(self.t, self.uo)
        self.uo._update_load_btc()
        self.uo._calc_wash_t_and_f()

        def run_test(cycle_t):

            self.uo._calc_cycle_t()

            # steady state conc estimation
            binding_species = [i for i in range(self.uo._n_species)
                               if i not in self.uo.non_binding_species]
            load_c_ss = self.uo._estimate_steady_state_mean_c(binding_species)
            np.testing.assert_array_almost_equal(
                self.log.get_data_tree(self.uo.uo_id)['load_c_ss'], load_c_ss
            )

            # cycle switch limit concentration
            if self.uo.load_c_end_ss is not None:
                load_c_end_ss = self.uo.load_c_end_ss
            else:
                load_c_end_ss = self.uo.load_c_end_relative_ss * load_c_ss
            np.testing.assert_array_almost_equal(
                self.log.get_data_tree(self.uo.uo_id)['load_c_end_ss'], load_c_end_ss
            )

            # cycle duration
            self.assertAlmostEqual(self.uo._cycle_t, cycle_t, 2)

        with self.assertWarns(Warning):

            self.uo.load_recycle = False
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(404.52)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(404.52)

            self.uo.load_recycle = True
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(394.335)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(394.32)

            self.uo.load_recycle = False
            self.uo.wash_recycle = True
            self.uo.wash_recycle_duration_cv = 12
            self.uo._calc_load_recycle_wash_i()

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(400.73)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(400.72)

            self.uo.load_recycle = True
            self.uo.wash_recycle = True

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(390.548)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(390.5195)

        self.uo.load_recycle_pdf = get_pdf_gaussian_fixed_dispersion(self.t, 1)

        with self.assertWarns(Warning):

            self.uo.load_recycle = False
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(401.92)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(401.92)

            self.uo.load_recycle = True
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(392.03)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(392.02)

            self.uo.load_recycle = False
            self.uo.wash_recycle = True
            self.uo.wash_recycle_duration_cv = 3
            self.uo._calc_load_recycle_wash_i()

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(399.90)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(399.92)

            self.uo.load_recycle = True
            self.uo.wash_recycle = True

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(390.012)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(390.0195)

            self.uo.load_c_end_estimate_with_iterative_solver_max_iter = 1
            with self.assertWarns(Warning):
                self.uo._calc_cycle_t()

        # test absolute definition load_c_end_ss
        self.uo.load_c_end_estimate_with_iterative_solver = False
        with self.assertWarns(Warning):
            binding_species = [i for i in range(self.uo._n_species)
                               if i not in self.uo.non_binding_species]
            _load_c_ss = self.uo._estimate_steady_state_mean_c(binding_species)
            self.uo.load_c_end_ss = _load_c_ss.copy() * self.uo.load_c_end_relative_ss
            self.uo.load_c_end_relative_ss = -1
            run_test(390.012)
            # test scrambled values
            self.uo.load_c_end_ss[0] = self.uo.load_c_end_ss.sum()
            self.uo.load_c_end_ss[1:] = 0
            run_test(390.012)

    def test_calc_first_cycle_extension_t(self):

        def warn_and_set_zero():
            with self.assertWarns(Warning):
                self.uo._calc_first_cycle_extension_t()
                self.assertEqual(self.uo._first_cycle_extension_t, 0)

        # ## prepare ##
        self.uo.load_extend_first_cycle = True
        self.uo.load_recycle = True
        self.uo.wash_recycle = True
        self.uo.load_extend_first_cycle_t = -1
        self.uo.load_extend_first_cycle_cv = -1

        # ## `zero cases` and assertions ##
        self.uo.load_recycle = False
        self.uo.wash_recycle = False
        warn_and_set_zero()
        self.uo.load_recycle = True
        self.uo.wash_recycle = True

        self.uo.load_extend_first_cycle = False
        warn_and_set_zero()
        self.uo.load_extend_first_cycle = True

        self.uo.load_cv = 1
        with self.assertRaises(NotImplementedError):
            self.uo._calc_first_cycle_extension_t()
        self.uo.load_cv = -1

        # ## functionality test ##
        # defined as t
        t_target = 14.5
        self.uo.load_extend_first_cycle_t = t_target
        self.uo._calc_first_cycle_extension_t()
        self.assertEqual(self.uo._first_cycle_extension_t, t_target)
        self.uo.load_extend_first_cycle_t = -1

        # defined as cv
        self.uo.load_extend_first_cycle_cv = 2.3
        # assertions (cv and _load_f need to be defined)
        self.uo._cv = -1
        self.uo._load_f = 3
        with self.assertRaises(AssertionError):
            self.uo._calc_first_cycle_extension_t()
        self.uo._cv = 2.7
        self.uo._load_f = -1
        with self.assertRaises(AssertionError):
            self.uo._calc_first_cycle_extension_t()
        self.uo._cv = 2.7
        self.uo._load_f = 4.3
        # results
        t_target = self.uo.load_extend_first_cycle_cv * self.uo._cv / self.uo._load_f
        self.uo._calc_first_cycle_extension_t()
        self.assertEqual(self.uo._first_cycle_extension_t, t_target)
        self.uo.load_extend_first_cycle_cv = -1

        # automatic
        # assert call to `_calc_cycle_t()`
        self.uo._n_species = 3
        self.uo.load_recycle = True
        self.uo.wash_recycle = False
        with self.assertRaises(AssertionError):
            self.uo._calc_first_cycle_extension_t()
        self.uo.load_recycle = True
        self.uo.wash_recycle = True
        with self.assertRaises(AssertionError):
            self.uo._calc_first_cycle_extension_t()
        self.uo.load_recycle = False
        self.uo.wash_recycle = True
        with self.assertRaises(AssertionError):
            self.uo._calc_first_cycle_extension_t()
        # make sure calling `_calc_cycle_t()` produces dependencies
        self.uo.load_recycle = True
        self.uo.wash_recycle = True
        self.uo.wash_desorption = False
        prep_for_calc_cycle_t(self.t, self.uo)
        self.uo._update_load_btc()
        self.uo._calc_wash_t_and_f()
        self.uo.wash_recycle_duration_cv = 12
        self.uo._calc_load_recycle_wash_i()
        with self.assertWarns(Warning):
            self.uo._calc_cycle_t()
        self.assertTrue(hasattr(self.uo, "_load_recycle_m_ss"))
        self.assertTrue(hasattr(self.uo, "_wash_recycle_m_ss"))
        # prep for `_calc_cycle_t()`
        prep_for_calc_cycle_t(self.t, self.uo)
        self.uo._update_load_btc()
        self.uo._calc_wash_t_and_f()

        # define checker

        def test_delay(t_delay):
            with self.assertWarns(Warning):
                self.uo._calc_first_cycle_extension_t()
            self.assertAlmostEqual(self.uo._first_cycle_extension_t, t_delay, 2)

        # test delays
        self.uo.load_recycle = True
        self.uo._load_recycle_m_ss = 250
        self.uo.wash_recycle = True
        self.uo._wash_recycle_m_ss = 500
        test_delay(70.2)
        self.uo.load_recycle = False
        self.uo.wash_recycle = True
        test_delay(47)
        self.uo.load_recycle = True
        self.uo.wash_recycle = False
        test_delay(23.7)

    def test_calc_cycle_start_i_list(self):
        self.uo._cycle_t = 14.7
        f = np.ones_like(self.t) * 2.2
        f_start_i = 10
        f_end_i = f.size - 100
        f[:f_start_i] = 0
        f[f_end_i + 1:] = 0
        cycle_start_t_list = np.arange(self.t[f_start_i], self.t[f_end_i - 1], self.uo._cycle_t)
        cycle_start_i_list_target = np.rint(cycle_start_t_list / self.dt).astype(np.int32)
        self.uo._f = f
        self.uo._calc_cycle_start_i_list()
        np.testing.assert_array_equal(cycle_start_i_list_target, self.uo._cycle_start_i_list)

        # with prolong of first cycle
        self.uo.load_extend_first_cycle = True
        self.uo.load_recycle = True
        with self.assertRaises(AttributeError):
            self.uo._calc_cycle_start_i_list()

        def test_with_delay():
            t_list = np.arange(self.t[f_start_i] + dt, self.t[f_end_i - 1], self.uo._cycle_t)
            t_list[0] = self.t[f_start_i]
            i_list = np.rint(t_list / self.dt).astype(np.int32)
            self.uo._calc_first_cycle_extension_t()
            self.uo._calc_cycle_start_i_list()
            np.testing.assert_array_equal(i_list, self.uo._cycle_start_i_list)

        self.uo.load_extend_first_cycle_cv = 1
        self.uo._cv = 1.5
        self.uo._load_f = 3.5
        dt = self.uo.load_extend_first_cycle_cv * self.uo._cv / self.uo._load_f
        test_with_delay()

        self.uo.load_extend_first_cycle_cv = -1
        self.uo.load_extend_first_cycle_t = 12.3
        self.uo._cv = 1.5
        self.uo._load_f = 3.5
        dt = self.uo.load_extend_first_cycle_t
        test_with_delay()

        # ignores delay and sends warning
        self.uo.load_recycle = False
        dt = 0
        with self.assertWarns(Warning):
            test_with_delay()

    def test_prepare_simulation(self):
        # individual parts are already tested, so here we want to ensure that they are called and in right order
        def ensure_parameters():
            self.assertTrue(self.uo._n_species > 0)
            self.assertTrue(
                len(self.uo.non_binding_species) < self.uo._n_species)
            self.assertTrue(self.uo._load_f > 0)
            self.assertTrue(self.uo._cv > 0)
            if self.uo.load_target_lin_velocity > 0:
                self.assertTrue(self.uo._col_h > 0)
            self.assertTrue(self.uo._equilibration_t >= 0)
            self.assertTrue(self.uo._wash_f > 0)
            self.assertTrue(self.uo._wash_t >= 0)
            self.assertTrue(self.uo._elution_f > 0)
            self.assertTrue(self.uo._elution_t > 0)
            self.assertTrue(self.uo.elution_peak_shape.get_p().size > 0)
            self.assertTrue(self.uo._elution_peak_cut_start_i >= 0)
            self.assertTrue(self.uo._elution_peak_cut_end_i > 0)
            self.assertTrue(self.uo._elution_peak_mask.size > 0)
            self.assertTrue(self.uo.load_bt.cv > 0)
            self.assertTrue(self.uo._cycle_t > 0)
            self.assertTrue(self.uo._cycle_start_i_list.size > 0)
            if self.uo.load_recycle and self.uo.wash_recycle:
                self.assertTrue(self.uo._wash_recycle_i_duration > 0)
                self.assertTrue(self.uo.load_recycle_pdf.get_p().size > 0)
            self.assertTrue(self.uo._cycle_t > 0)
            self.assertTrue(self.uo._regeneration_t >= 0)

        # needed parameters
        self.uo._f = np.ones_like(self.t)
        for n_species in (1, 5):
            # single specie, add wash recycle
            self.uo.load_extend_first_cycle = False
            self.uo._n_species = n_species
            self.uo.load_recycle = True
            self.uo.load_recycle_pdf = get_pdf_gaussian_fixed_relative_width(self.t)
            self.uo.column_porosity_retentate = 0.84
            self.uo.wash_recycle = True
            self.uo.wash_recycle_duration_t = 1.2
            self.uo.load_target_lin_velocity = -1
            self.uo._prepare_simulation()
            ensure_parameters()

            self.uo.load_extend_first_cycle = True
            self.uo.load_extend_first_cycle_cv = 2
            self.uo._prepare_simulation()
            ensure_parameters()
            self.uo.load_extend_first_cycle = False

            self.uo.wash_recycle = False
            self.uo.wash_recycle_duration_t = -1
            self.uo._prepare_simulation()
            ensure_parameters()
            self.uo.load_recycle = False
            self.uo.load_recycle_pdf = None
            self.uo.column_porosity_retentate = -1
            self.uo.load_target_lin_velocity = 2.4
            self.uo._prepare_simulation()
            ensure_parameters()

            self.uo.load_extend_first_cycle = True
            with self.assertWarns(Warning):
                self.uo._prepare_simulation()
            self.uo.load_extend_first_cycle = False

            # cycle length wrong (load to short compare to the rest of the process)
            self.uo.regeneration_cv = 100
            with self.assertRaises(RuntimeError):
                self.uo._prepare_simulation()
            self.uo.regeneration_cv = 2

    def test_sim_c_load_binding(self):
        # pre-requirements
        self.uo.load_bt = get_bt_constant_pattern_solution(self.dt)
        self.uo._cv = 13.4
        self.uo._update_load_btc()
        self.uo._n_species = 5
        self.uo.non_binding_species = [1, 3]
        # definitions
        c_load = np.array([[1], [5], [3]]) * np.ones([3, 150])
        f_load = 1.6 * np.ones(c_load.shape[1])
        # test assertions
        with self.assertRaises(AssertionError):
            self.uo._sim_c_load_binding(f_load, c_load[:, :140])
        with self.assertRaises(AssertionError):
            self.uo._sim_c_load_binding(f_load, c_load[:2, :])
        with self.assertRaises(AssertionError):
            self.uo._sim_c_load_binding(f_load[:140], c_load)
        # test functionality
        c_bound_target = self.uo.load_bt.calc_c_bound(f_load, c_load)
        c_bound, c_unbound = self.uo._sim_c_load_binding(f_load, c_load)
        np.testing.assert_array_almost_equal(c_bound_target, c_bound)
        np.testing.assert_array_almost_equal(c_load - c_bound_target, c_unbound)

    def test_sim_c_wash_desorption(self):
        with self.assertRaises(NotImplementedError):
            self.uo._sim_c_wash_desorption(np.array([]), np.array([]))

    def test_sim_c_recycle_propagation(self):
        # pre-requirements
        load_recycle_pdf = get_pdf_gaussian_fixed_relative_width(self.t, )
        self.uo.load_recycle_pdf = get_pdf_gaussian_fixed_relative_width(self.t, )
        self.uo._cv = 13.44
        self.uo.column_porosity_retentate = 0.64
        v_void = self.uo._cv * self.uo.column_porosity_retentate
        load_f = 1.4
        self.uo._update_load_recycle_pdf(load_f)
        self.uo._n_species = 5
        self.uo.non_binding_species = [1, 3]

        # definitions
        wash_f = 3.1
        wash_t = 12.23
        c_load_unbound = np.array([[1], [5], [3]]) * np.ones([3, 150])
        f_load = load_f * np.ones(c_load_unbound.shape[1])
        f_load[:50] = wash_f
        c_wash_desorbed = \
            np.array([[1], [5], [3]]) * \
            np.linspace(3, 0, int(round(wash_t / self.dt)))[np.newaxis, :]
        self.uo._wash_f = wash_f
        self.uo._wash_t = wash_t

        # test assertions
        self.assert_defined_value("_wash_f", self.uo._sim_c_recycle_propagation, f_load, c_load_unbound, None)
        self.assert_defined_value("_wash_t", self.uo._sim_c_recycle_propagation, f_load, c_load_unbound, None)
        self.assert_defined_value("load_recycle_pdf", self.uo._sim_c_recycle_propagation, f_load, c_load_unbound, None)
        with self.assertRaises(AssertionError):
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound[:, :140], c_wash_desorbed)
        with self.assertRaises(AssertionError):
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound[:2, :], c_wash_desorbed)
        with self.assertRaises(AssertionError):
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, c_wash_desorbed[:, :40])
        with self.assertRaises(AssertionError):
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, c_wash_desorbed[:2, :])
        with self.assertRaises(AssertionError):
            self.uo._sim_c_recycle_propagation(f_load[:140], c_load_unbound, c_wash_desorbed)

        # test functionality (base)
        v_load = self.dt * f_load.cumsum()
        v_wash = v_load[-1] + self.dt * np.arange(1, 1 + c_wash_desorbed.shape[1]) * wash_f
        dv = min(f_load.min(), wash_f) * self.dt
        v = np.arange(dv, v_wash[-1] + dv, dv)
        c_v_combined = sc_interp.interp1d(
            np.concatenate((v_load, v_wash), axis=0),
            np.concatenate((c_load_unbound, c_wash_desorbed), axis=1),
            fill_value="extrapolate"
        )(v)
        c_v_combined[c_v_combined < 0] = 0
        # simulate traveling of leftover material through the column
        load_recycle_pdf.update_pdf(v_void=v_void, f=load_f, rt_mean=v_void / wash_f)
        c_v_combined_propagated = utils.convolution.time_conv(self.dt, c_v_combined, load_recycle_pdf.get_p())
        # split back on time scale
        c_combined_propagated = sc_interp.interp1d(
            v,
            c_v_combined_propagated,
            fill_value="extrapolate"
        )(np.concatenate((v_load, v_wash), axis=0))
        c_combined_propagated[c_combined_propagated < 0] = 0
        c_unbound_propagated, c_wash_desorbed_propagated = \
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, c_wash_desorbed)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, :v_load.size], c_unbound_propagated)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, v_load.size:], c_wash_desorbed_propagated)

        # test functionality (constant flow)
        f_load[:] = wash_f
        # simulate traveling of leftover material through the column
        load_recycle_pdf.update_pdf(v_void=v_void, f=wash_f, rt_mean=v_void / wash_f)
        c_combined_propagated = utils.convolution.time_conv(self.dt,
                                                            np.concatenate((c_load_unbound, c_wash_desorbed), axis=1),
                                                            load_recycle_pdf.get_p())
        c_unbound_propagated, c_wash_desorbed_propagated = \
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, c_wash_desorbed)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, :v_load.size], c_unbound_propagated)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, v_load.size:], c_wash_desorbed_propagated)

        # test functionality (f_load = 2 * wash_f)
        f_load[:] = 2 * wash_f
        # simulate traveling of leftover material through the column
        load_recycle_pdf.update_pdf(v_void=v_void, f=wash_f, rt_mean=v_void / wash_f)
        c_combined_propagated = utils.convolution.time_conv(self.dt, np.concatenate(
            (c_load_unbound.repeat(2, axis=1), c_wash_desorbed), axis=1), load_recycle_pdf.get_p())
        c_unbound_propagated, c_wash_desorbed_propagated = \
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, c_wash_desorbed)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, 1:v_load.size * 2:2], c_unbound_propagated)
        np.testing.assert_array_almost_equal(c_combined_propagated[:, v_load.size * 2:], c_wash_desorbed_propagated)

        # test functionality (no wash)
        f_load[:50] = wash_f
        v_load = self.dt * f_load.cumsum()
        dv = f_load.min() * self.dt
        v = np.arange(dv, v_load[-1] + dv, dv)
        c_v_combined = sc_interp.interp1d(v_load, c_load_unbound, fill_value="extrapolate")(v)
        # simulate traveling of leftover material through the column
        load_recycle_pdf.update_pdf(v_void=v_void, f=wash_f, rt_mean=v_void / wash_f)
        c_v_combined_propagated = utils.convolution.time_conv(self.dt, c_v_combined, load_recycle_pdf.get_p())
        # split back on time scale
        c_load_propagated = sc_interp.interp1d(v, c_v_combined_propagated, fill_value="extrapolate")(v_load)
        c_load_propagated[c_load_propagated < 0] = 0
        c_unbound_propagated, c_wash_desorbed_propagated = \
            self.uo._sim_c_recycle_propagation(f_load, c_load_unbound, None)
        np.testing.assert_array_almost_equal(c_load_propagated, c_unbound_propagated)
        self.assertAlmostEqual(
            c_wash_desorbed_propagated.sum() * wash_f,
            ((c_load_unbound - c_load_propagated) * f_load).sum(),
            3
        )

    def test_sim_c_elution_buffer(self):
        def calc_target_outlet(c, n):
            return c.reshape(c.size, 1) * np.ones([1, n])

        n_steps = 102
        self.uo._n_species = 1
        self.uo.elution_buffer_c = np.array([])
        np.testing.assert_array_equal(calc_target_outlet(np.array([0]), n_steps),
                                      self.uo._sim_c_elution_buffer(n_steps))

        self.uo.elution_buffer_c = np.array([3])
        np.testing.assert_array_equal(calc_target_outlet(np.array([3]), n_steps),
                                      self.uo._sim_c_elution_buffer(n_steps))

        self.uo._n_species = 2
        self.uo.elution_buffer_c = np.array([3])
        with self.assertRaises(AssertionError):
            self.uo._sim_c_elution_buffer(n_steps)

        self.uo.elution_buffer_c = np.array([1, 3])
        np.testing.assert_array_equal(calc_target_outlet(self.uo.elution_buffer_c, n_steps),
                                      self.uo._sim_c_elution_buffer(n_steps))

        self.uo.elution_buffer_c = np.array([[1], [3]])
        np.testing.assert_array_equal(calc_target_outlet(self.uo.elution_buffer_c, n_steps),
                                      self.uo._sim_c_elution_buffer(n_steps))

        self.uo.elution_buffer_c = np.array([])
        np.testing.assert_array_equal(calc_target_outlet(np.array([0, 0]), n_steps),
                                      self.uo._sim_c_elution_buffer(n_steps))

        n_steps = 123
        # wrong number of species
        self.uo.elution_buffer_c = np.array([1])
        self.uo._n_species = 3
        with self.assertRaises(AssertionError):
            self.uo._sim_c_elution_buffer(n_steps)
        # wrong number of species pt. 2
        self.uo.elution_buffer_c = np.array([1, 4, 5, -1])
        self.uo._n_species = 3
        with self.assertRaises(AssertionError):
            self.uo._sim_c_elution_buffer(n_steps)
        # negative conc value
        self.uo.elution_buffer_c = np.array([1, 4, -1])
        self.uo._n_species = 3
        with self.assertRaises(AssertionError):
            self.uo._sim_c_elution_buffer(n_steps)

        # zero array
        self.uo.elution_buffer_c = np.array([])
        self.uo._n_species = 3
        c_out = np.zeros([self.uo._n_species, n_steps])
        np.testing.assert_array_equal(c_out, self.uo._sim_c_elution_buffer(n_steps))
        # array
        self.uo.elution_buffer_c = np.array([1, 3, 4])
        c_out = self.uo.elution_buffer_c[:, np.newaxis] * np.ones([1, n_steps])
        np.testing.assert_array_equal(c_out, self.uo._sim_c_elution_buffer(n_steps))
        # array
        self.uo._n_species = 1
        self.uo.elution_buffer_c = np.array(3)
        c_out = np.ones([1, n_steps]) * 3
        np.testing.assert_array_equal(c_out, self.uo._sim_c_elution_buffer(n_steps))

    def test_sim_c_elution_desorption(self):
        # prepare
        m_bound = np.array([2300, 800])
        self.uo._elution_f = 3.5
        self.uo._cv = 12.3
        self.uo._elution_t = 35
        self.uo._elution_peak_t = 15
        self.uo._elution_peak_cut_start_i = int(round(5 / self.dt))
        self.uo._elution_peak_cut_end_i = int(round(25 / self.dt))
        self.uo._update_elution_peak_pdf()
        self.uo._calc_elution_peak_mask()
        i_el_d = int(round(self.uo._elution_t / self.dt))

        # calc
        c_elution_target = \
            self.uo.elution_peak_shape.get_p()[np.newaxis, :i_el_d] * m_bound[:, np.newaxis] / self.uo._elution_f
        c_elution, mask_elution_peak = self.uo._sim_c_elution_desorption(m_bound)
        np.testing.assert_array_equal(mask_elution_peak, self.uo._elution_peak_mask)
        np.testing.assert_array_almost_equal(c_elution, c_elution_target)

        # pad too short pdf
        self.uo.elution_peak_shape = get_pdf_gaussian_fixed_relative_width(self.t, 0.002)
        self.uo._update_elution_peak_pdf()
        self.assertTrue(self.uo.elution_peak_shape.get_p().size < i_el_d)  # pdf is shorter than elution duration
        c_elution_target = np.pad(
            self.uo.elution_peak_shape.get_p()[np.newaxis, :i_el_d] * m_bound[:, np.newaxis] / self.uo._elution_f,
            ((0, 0), (0, i_el_d - self.uo.elution_peak_shape.get_p().size)),
            mode="constant"
        )
        c_elution, mask_elution_peak = self.uo._sim_c_elution_desorption(m_bound)
        np.testing.assert_array_equal(mask_elution_peak, self.uo._elution_peak_mask)
        np.testing.assert_array_almost_equal(c_elution, c_elution_target)

    def test_sim_c_regeneration(self):
        self.assertTrue(self.uo._sim_c_regeneration(np.array([[1]])) is None)

    # noinspection DuplicatedCode
    def test_sim_c_out_cycle(self):
        # prepare
        self.uo._load_f = 1.5
        self.uo._wash_f = 1.4
        self.uo._wash_t = 12.32
        self.uo._elution_f = 3.2
        self.uo._elution_t = 4.2
        self.uo._cv = 15.4
        # prepare pt 2
        self.wash_desorption = False
        self.uo._n_species = 5
        self.uo.non_binding_species = [1, 3]
        self.uo.column_porosity_retentate = 0.64
        self.uo.load_recycle_pdf = get_pdf_gaussian_fixed_relative_width(self.t, )
        self.uo._update_load_recycle_pdf(self.uo._load_f)
        self.uo._elution_peak_t = 3.64
        self.uo._update_elution_peak_pdf()
        self.uo._update_load_btc()
        self.uo._elution_peak_mask = np.zeros(int(round(self.uo._elution_t / self.dt)), dtype=bool)
        self.uo._elution_peak_mask[50:150] = True

        # definitions
        c_load = np.array([[1], [5], [3]]) * np.ones([3, 150])
        f_load = 1.6 * np.ones(c_load.shape[1])
        log_tree = dict()
        self.uo._cycle_tree = log_tree
        self.log.set_data_tree("cycle_tree", self.uo._cycle_tree)

        # test assertions
        self.assert_defined_value("_load_f", self.uo._sim_c_out_cycle, f_load, c_load)
        self.assert_defined_value("_wash_f", self.uo._sim_c_out_cycle, f_load, c_load)
        self.assert_defined_value("_wash_t", self.uo._sim_c_out_cycle, f_load, c_load)
        self.assert_defined_value("_elution_f", self.uo._sim_c_out_cycle, f_load, c_load)
        self.assert_defined_value("_elution_t", self.uo._sim_c_out_cycle, f_load, c_load)
        self.assert_defined_value("_cv", self.uo._sim_c_out_cycle, f_load, c_load)

        # test functionality
        self.uo.load_recycle = True
        c_bound, c_unbound = self.uo._sim_c_load_binding(f_load, c_load)
        c_out_load_target, c_out_wash_target = self.uo._sim_c_recycle_propagation(f_load, c_unbound, None)
        m_bound = (c_bound * f_load[np.newaxis, :]).sum(1) * self.dt
        c_out_elution_target, elution_peak_mask_target = self.uo._sim_c_elution_desorption(m_bound)
        # sim
        c_out_load, c_out_wash, c_out_elution, elution_peak_mask, c_out_regeneration = \
            self.uo._sim_c_out_cycle(f_load, c_load)
        # assert
        np.testing.assert_array_almost_equal(c_out_load_target, c_out_load)
        np.testing.assert_array_almost_equal(c_out_wash_target, c_out_wash)
        np.testing.assert_array_almost_equal(c_out_elution_target, c_out_elution)
        np.testing.assert_array_equal(elution_peak_mask_target, elution_peak_mask)
        self.assertIsNone(c_out_regeneration)

        # test functionality (no recycle)
        self.uo.load_recycle = False
        # sim
        c_out_load, c_out_wash, c_out_elution, elution_peak_mask, c_out_regeneration = \
            self.uo._sim_c_out_cycle(f_load, c_load)
        # assert
        np.testing.assert_array_almost_equal(c_unbound, c_out_load)
        self.assertIsNone(c_out_wash)
        np.testing.assert_array_almost_equal(c_out_elution_target, c_out_elution)
        np.testing.assert_array_equal(elution_peak_mask_target, elution_peak_mask)
        self.assertIsNone(c_out_regeneration)

        # test functionality (recycle both)
        self.uo.wash_recycle = True
        self.uo.load_recycle = True
        c_out_load_target, c_out_wash_target = self.uo._sim_c_recycle_propagation(f_load, c_unbound, None)
        m_bound = (c_bound * f_load[np.newaxis, :]).sum(1) * self.dt
        c_out_elution_target, elution_peak_mask_target = self.uo._sim_c_elution_desorption(m_bound)
        # sim
        c_out_load, c_out_wash, c_out_elution, elution_peak_mask, c_out_regeneration = \
            self.uo._sim_c_out_cycle(f_load, c_load)
        # assert
        np.testing.assert_array_almost_equal(c_out_load_target, c_out_load)
        np.testing.assert_array_almost_equal(c_out_wash_target, c_out_wash)
        np.testing.assert_array_almost_equal(c_out_elution_target, c_out_elution)
        np.testing.assert_array_equal(elution_peak_mask_target, elution_peak_mask)
        self.assertIsNone(c_out_regeneration)

    def test_calculate(self):
        self.uo.non_binding_species = [2]
        self.uo._n_species = 3
        c_load = np.array([[1], [5], [3]]) * np.ones([self.uo._n_species, self.t.size])
        f_load = 1.6 * np.ones(c_load.shape[1])
        f_load[int(round(13 / self.dt)):] = 0
        self.uo._c = c_load
        self.uo._f = f_load
        self.uo.load_recycle = False
        self.uo.wash_recycle = False
        self.uo._calculate()

        self.uo.load_recycle = True
        self.uo.column_porosity_retentate = 0.64
        self.uo.load_recycle_pdf = get_pdf_gaussian_fixed_relative_width(self.t, relative_sigma=0.2)
        self.uo._calculate()

        # TODO: Validate logged data and end profiles of `_calculate()` method.


class TestACC(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 2000, 20000)
        self.dt = self.t[1] - self.t[0]
        self.uo = sc_uo.ACC(
            t=self.t,
            uo_id="ACC_Test",
            load_bt=get_bt_constant_pattern_solution(self.dt, dbc_100=240, k=0.2),
            peak_shape_pdf=get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=100)
        )

        """ Set logger (optional) """
        self.log = TestLogger()
        self.uo.log = self.log

        """ Set other parameters """
        set_ac_parameters(self.uo)

    def test_sim_c_wash_desorption(self):
        with self.assertRaises(NotImplementedError):
            self.uo._sim_c_wash_desorption(np.array([]), np.array([]))

    def test_calculate(self):
        self.uo.non_binding_species = [2]
        self.uo._n_species = 4
        c_load = np.array([[1], [5], [3], [3]]) * np.ones([self.uo._n_species, self.t.size])
        f_load = 1.8 * np.ones(c_load.shape[1])
        f_load[int(round(15 / self.dt)):] = 0
        self.uo._c = c_load
        self.uo._f = f_load
        self.uo.load_recycle = False
        self.uo.wash_recycle = False
        self.uo._calculate()


class TestPCC(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 2000, 20000)
        self.dt = self.t[1] - self.t[0]
        self.uo = sc_uo.PCC(
            t=self.t,
            uo_id="PCC_Test",
            load_bt=get_bt_constant_pattern_solution(self.dt, dbc_100=240, k=0.2),
            peak_shape_pdf=get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=100),
            load_recycle_pdf=get_pdf_gaussian_fixed_relative_width(self.t, relative_sigma=0.2),
            column_porosity_retentate=0.64
        )

        self.assertTrue(self.uo.column_porosity_retentate == 0.64)
        self.assertTrue(self.uo.load_recycle_pdf is not None)
        self.assertTrue(self.uo.load_recycle)
        self.assertFalse(self.uo.wash_recycle)

        """ Set logger (optional) """
        self.log = TestLogger()
        self.uo.log = self.log

        """ Set other parameters """
        set_ac_parameters(self.uo)

    def test_sim_c_wash_desorption(self):
        with self.assertRaises(NotImplementedError):
            self.uo._sim_c_wash_desorption(np.array([]), np.array([]))

    def test_calculate(self):
        self.uo.non_binding_species = [2]
        self.uo._n_species = 3
        c_load = np.array([[1], [0.2], [3]]) * np.ones([self.uo._n_species, self.t.size])
        f_load = 1.6 * np.ones(c_load.shape[1])
        f_load[int(round(13 / self.dt)):] = 0
        self.uo._c = c_load
        self.uo._f = f_load
        self.uo._calculate()


class TestPCCWithWashDesorption(unittest.TestCase):

    def assert_defined_value(self, par_name, func, *args):
        v = getattr(self.uo, par_name)
        setattr(self.uo, par_name, -1 if isinstance(v, numbers.Number) else None)
        with self.assertRaises(AssertionError):
            func(*args)
        setattr(self.uo, par_name, v)

    def setUp(self) -> None:
        self.t = np.linspace(0, 2000, 20000)
        self.dt = self.t[1] - self.t[0]
        self.uo = sc_uo.PCCWithWashDesorption(
            t=self.t,
            uo_id="PCCWithWashDesorption_Test",
            load_bt=get_bt_constant_pattern_solution(self.dt, dbc_100=240, k=0.2),
            peak_shape_pdf=get_pdf_gaussian_fixed_dispersion(self.t, dispersion_index=100),
            load_recycle_pdf=get_pdf_gaussian_fixed_relative_width(self.t, relative_sigma=0.2),
            column_porosity_retentate=0.64
        )

        self.assertTrue(self.uo.load_recycle)
        self.assertTrue(self.uo.wash_recycle)
        self.assertTrue(self.uo.wash_desorption)

        """ Set logger (optional) """
        self.log = TestLogger()
        self.uo.log = self.log

        """ Set other parameters """
        set_ac_parameters(self.uo)

    def test_sim_c_wash_desorption(self):
        # prepare
        self.uo._cv = 12.3
        self.uo._load_f = 1.43
        self.uo._wash_f = 3.1
        self.uo._wash_t = 5 * self.uo._cv / self.uo._wash_f
        self.uo.wash_desorption_tail_half_time_cv = 1
        c_bound = np.array([[1], [0.2], [3], [3]]) * np.ones([4, int(self.t.size / 12)])
        f_load = self.uo._load_f * np.ones(c_bound.shape[1])
        wash_recycle_duration_t = 3 * self.uo._cv / self.uo._wash_f
        wash_recycle_duration_cv = 3
        wash_desorption_desorbable_material_share = 0.1
        wash_desorption_desorbable_above_dbc = 80
        self.uo._n_species = 4
        self.uo.non_binding_species = []
        self.uo.wash_recycle_duration_t = wash_recycle_duration_t
        self.uo.wash_recycle_duration_cv = wash_recycle_duration_cv
        self.uo.wash_desorption_desorbable_material_share = wash_desorption_desorbable_material_share
        self.uo.wash_desorption_desorbable_above_dbc = wash_desorption_desorbable_above_dbc

        # test assertions
        self.assert_defined_value("wash_desorption_tail_half_time_cv",
                                  self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.assert_defined_value("_load_f", self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.assert_defined_value("_wash_f", self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.assert_defined_value("_wash_t", self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.assert_defined_value("_cv", self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.uo.wash_desorption_desorbable_material_share = -1
        self.assert_defined_value("wash_desorption_desorbable_above_dbc",
                                  self.uo._sim_c_wash_desorption, f_load, c_bound)
        self.uo.wash_desorption_desorbable_material_share = wash_desorption_desorbable_material_share
        self.uo.wash_desorption_desorbable_above_dbc = -1
        self.assert_defined_value("wash_desorption_desorbable_material_share",
                                  self.uo._sim_c_wash_desorption, f_load, c_bound)

        # test functionality
        # defined by desorbable material share
        self.uo.wash_desorption_desorbable_above_dbc = -1
        self.uo.wash_desorption_desorbable_material_share = wash_desorption_desorbable_material_share
        c_desorbed = self.uo._sim_c_wash_desorption(f_load, c_bound)
        m_load = (f_load[np.newaxis, :] * c_bound).sum() * self.dt
        self.assertAlmostEqual(
            c_desorbed.sum() * self.uo._wash_f * self.dt,
            wash_desorption_desorbable_material_share * m_load
        )
        # with waring (if defined in 2 ways)
        self.uo.wash_desorption_desorbable_above_dbc = wash_desorption_desorbable_above_dbc
        self.uo.wash_desorption_desorbable_material_share = wash_desorption_desorbable_material_share
        with self.assertWarns(Warning):
            c_desorbed = self.uo._sim_c_wash_desorption(f_load, c_bound)
            m_load = (f_load[np.newaxis, :] * c_bound).sum() * self.dt
            self.assertAlmostEqual(
                c_desorbed.sum() * self.uo._wash_f * self.dt,
                wash_desorption_desorbable_material_share * m_load
            )
        # defined by dbc limit
        self.uo.wash_desorption_desorbable_above_dbc = wash_desorption_desorbable_above_dbc
        self.uo.wash_desorption_desorbable_material_share = -1
        c_desorbed = self.uo._sim_c_wash_desorption(f_load, c_bound)
        m_load = (f_load[np.newaxis, :] * c_bound).sum() * self.dt
        self.assertAlmostEqual(
            c_desorbed.sum() * self.uo._wash_f * self.dt,
            m_load - wash_desorption_desorbable_above_dbc * self.uo._cv
        )

    # noinspection DuplicatedCode
    def test_calc_cycle_t(self):
        self.uo._cv = 14.3
        self.uo._load_f = 3.5
        self.uo._n_species = 3

        self.uo.load_cv = 2.3
        self.uo.load_c_end_ss = None
        self.uo.load_c_end_relative_ss = -1

        self.assert_defined_value("_cv", self.uo._calc_cycle_t)
        self.assert_defined_value("_load_f", self.uo._calc_cycle_t)
        self.assert_defined_value("load_cv", self.uo._calc_cycle_t)

        # ## test definition by `load_cv` ##
        self.uo.load_cv = 2.3
        self.assertFalse(hasattr(self.uo, "_cycle_t"))
        self.uo._calc_cycle_t()
        self.assertEqual(self.uo._cycle_t, self.uo.load_cv * self.uo._cv / self.uo._load_f)
        # warn if defined in multiple ways
        self.uo.load_c_end_ss = np.array([[1], [0.5]])
        with self.assertWarns(Warning):
            self.uo._calc_cycle_t()
        self.uo.load_c_end_ss = None
        self.uo.load_c_end_relative_ss = 0.7
        with self.assertWarns(Warning):
            self.uo._calc_cycle_t()
        self.uo.load_cv = -1

        # ## test definition by `load_c_end_relative_ss` ##
        # prepare flow and conc profile
        prep_for_calc_cycle_t(self.t, self.uo)
        self.uo._update_load_btc()
        self.uo._calc_wash_t_and_f()

        def run_test(cycle_t):
            self.uo._calc_cycle_t()

            # steady state conc estimation
            binding_species = [i for i in range(self.uo._n_species)
                               if i not in self.uo.non_binding_species]
            load_c_ss = self.uo._estimate_steady_state_mean_c(binding_species)
            np.testing.assert_array_almost_equal(
                self.log.get_data_tree(self.uo.uo_id)['load_c_ss'], load_c_ss
            )

            # cycle switch limit concentration
            load_c_end_ss = self.uo.load_c_end_relative_ss * load_c_ss
            np.testing.assert_array_almost_equal(
                self.log.get_data_tree(self.uo.uo_id)['load_c_end_ss'], load_c_end_ss
            )

            # cycle duration
            self.assertAlmostEqual(self.uo._cycle_t, cycle_t, 2)

        self.uo.wash_desorption = True
        self.uo.wash_desorption_tail_half_time_cv = 3
        self.uo.wash_desorption_desorbable_material_share = 0.1

        with self.assertWarns(Warning):
            self.uo.load_recycle = False
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(404.52)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(404.52)

            self.uo.load_recycle = True
            self.uo.wash_recycle = False

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(394.335)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(394.32)

            self.uo.load_recycle = False
            self.uo.wash_recycle = True
            self.uo.wash_recycle_duration_cv = 12
            self.uo._calc_load_recycle_wash_i()

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(363.873)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(363.918)

            self.uo.load_recycle = True
            self.uo.wash_recycle = True

            self.uo.load_c_end_estimate_with_iterative_solver = False
            run_test(353.69)
            self.uo.load_c_end_estimate_with_iterative_solver = True
            run_test(353.72)

    def test_calculate(self):
        self.uo.wash_desorption = True
        self.uo.wash_desorption_tail_half_time_cv = 3
        self.uo.wash_desorption_desorbable_material_share = 0.1
        self.uo.wash_recycle_duration_cv = 3

        self.uo.wash_recycle = True

        self.uo.non_binding_species = [2, 3]
        self.uo._n_species = 4
        c_load = np.array([[1], [0.2], [3], [3]]) * np.ones([4, self.t.size])
        f_load = 1.8 * np.ones(c_load.shape[1])
        self.uo._c = c_load
        self.uo._f = f_load
        self.uo._prepare_simulation()
        self.uo._calculate()

        # TODO: Validate logged data and end profiles of `_calculate()` method.
