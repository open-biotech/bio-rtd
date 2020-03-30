import unittest

from bio_rtd import adj_par


class TestAdjustableParameter(unittest.TestCase):

    def test_adjustable_parameter(self):
        var_list = 'par1'
        v_range = (0, 10)
        v_init = 2
        scale_factor = 1.2
        par_name = 'vip_par'
        gui_type = 'checkbox'

        ap = adj_par.AdjustableParameter(
            [var_list], v_range, v_init, scale_factor, par_name, gui_type
        )

        self.assertEqual([var_list], ap.var_list)
        self.assertEqual(v_range, ap.v_range)
        self.assertEqual(v_init, ap.v_init)
        self.assertEqual(scale_factor, ap.scale_factor)
        self.assertEqual(par_name, ap.par_name)
        self.assertEqual(gui_type, ap.gui_type)

        # default parameters
        ap = adj_par.AdjustableParameter([var_list])
        self.assertIsNone(ap.v_range)
        self.assertIsNone(ap.v_init)
        self.assertEqual(ap.scale_factor, 1)
        self.assertIsNone(ap.par_name)
        self.assertIsNone(ap.gui_type)

    def test_adj_par_boolean(self):
        var = 'par2'
        v_init = False
        par_name = 'vip_par'
        gui_type = 'checkbox'

        ap = adj_par.AdjParBoolean(
            var=var, v_init=v_init,
            par_name=par_name, gui_type=gui_type
        )

        self.assertEqual([var], ap.var_list)
        self.assertEqual(v_init, ap.v_init)
        self.assertEqual(par_name, ap.par_name)
        self.assertEqual(gui_type, ap.gui_type)

        # default parameters
        ap = adj_par.AdjParBoolean(var)
        self.assertIsNone(ap.v_init)
        self.assertIsNone(ap.par_name)
        self.assertEqual(ap.gui_type, 'checkbox')

    def test_adj_par_boolean_multiple(self):
        var_list = ['par1', 'par2']
        v_init = [False, True]
        par_name = ['vip_par', 'vip_par_2']
        gui_type = 'radio_group_2'
        ap = adj_par.AdjParBooleanMultiple(
            var_list=var_list, v_init=v_init,
            par_name=par_name, gui_type=gui_type
        )
        self.assertTrue(var_list, ap.var_list)
        self.assertTrue(v_init, ap.v_init)
        self.assertTrue(par_name, ap.par_name)
        self.assertTrue(gui_type, ap.gui_type)
        # assertions
        with self.assertRaises(AssertionError):
            adj_par.AdjParBooleanMultiple(
                var_list=['var'], v_init=v_init,
                par_name=par_name, gui_type=gui_type)
        with self.assertRaises(AssertionError):
            adj_par.AdjParBooleanMultiple(
                var_list=var_list, v_init=[True],
                par_name=par_name, gui_type=gui_type)
        with self.assertRaises(AssertionError):
            adj_par.AdjParBooleanMultiple(
                var_list=['var'], v_init=v_init,
                par_name=['fef'], gui_type=gui_type)
        # default parameters
        ap = adj_par.AdjParBooleanMultiple(var_list)
        self.assertIsNone(ap.v_init)
        self.assertIsNone(ap.par_name)
        self.assertEqual(ap.gui_type, 'radio_group')

    def test_adj_par_slider(self):
        var = 'par1'
        v_range = (2, 10, 2)
        v_init = 2
        scale_factor = 1.2
        par_name = 'vip_par'
        gui_type = 'slider_2'

        ap = adj_par.AdjParSlider(
            var, v_range, v_init, scale_factor, par_name, gui_type
        )

        self.assertEqual([var], ap.var_list)
        self.assertEqual(v_range, ap.v_range)
        self.assertEqual(v_init, ap.v_init)
        self.assertEqual(scale_factor, ap.scale_factor)
        self.assertEqual(par_name, ap.par_name)
        self.assertEqual(gui_type, ap.gui_type)

        v_range = 10
        ap = adj_par.AdjParSlider(var, v_range)
        self.assertEqual((0, 10, 1), ap.v_range)
        v_range = (5, 25)
        ap = adj_par.AdjParSlider(var, v_range)
        self.assertEqual((5, 25, 2), ap.v_range)

        # default parameters
        ap = adj_par.AdjParSlider(var, 10)
        self.assertEqual((0, 10, 1), ap.v_range)
        self.assertIsNone(ap.v_init)
        self.assertEqual(ap.scale_factor, 1)
        self.assertIsNone(ap.par_name)
        self.assertEqual(ap.gui_type, 'slider')

    # noinspection DuplicatedCode
    def test_adj_par_range(self):
        var_list = ('start', 'end')
        v_range = (2, 10, 2)
        v_init = (2, 5.3)
        scale_factor = 1.2
        par_name = 'vip_par'
        gui_type = 'range_2'

        ap = adj_par.AdjParRange(
            var_list, v_range, v_init, scale_factor, par_name, gui_type
        )

        self.assertEqual(list(var_list), ap.var_list)
        self.assertEqual(v_range, ap.v_range)
        self.assertEqual(v_init, ap.v_init)
        self.assertEqual(scale_factor, ap.scale_factor)
        self.assertEqual(par_name, ap.par_name)
        self.assertEqual(gui_type, ap.gui_type)

        v_range = 10
        ap = adj_par.AdjParRange(var_list, v_range)
        self.assertEqual((0, 10, 1), ap.v_range)
        v_range = (5, 25)
        ap = adj_par.AdjParRange(var_list, v_range)
        self.assertEqual((5, 25, 2), ap.v_range)

        # default parameters
        ap = adj_par.AdjParRange(var_list, 10)
        self.assertEqual((0, 10, 1), ap.v_range)
        self.assertIsNone(ap.v_init)
        self.assertEqual(ap.scale_factor, 1)
        self.assertIsNone(ap.par_name)
        self.assertEqual(ap.gui_type, 'range')
