import unittest
import numpy as np

from bio_rtd import inlet


class TestInlet(unittest.TestCase):

    def check_parent_init(self, i, species, inlet_id, gui_title):
        self.assertEqual(i.species_list, species)
        self.assertEqual(i._n_species, len(species))
        self.assertEqual(i.uo_id, inlet_id)
        self.assertEqual(i.gui_title, gui_title)

    # noinspection DuplicatedCode
    def test_constant_inlet(self):
        t = np.linspace(0, 10, 100)
        f = 2.4
        c = np.array([2, 3.1])
        species = ['c1', 'c2']
        inlet_id = 'constant_inlet'
        gui_title = 'Constant Inlet'

        i = inlet.ConstantInlet(t, f, c, species, inlet_id, gui_title)
        self.check_parent_init(i, species, inlet_id, gui_title)

        self.assertEqual(f, i.f)
        np.testing.assert_array_equal(c, i.c)

        # make sure it's copy
        c += 1
        np.testing.assert_array_almost_equal(c, i.c + 1)
        c -= 1

        # assertions
        with self.assertRaises(AssertionError):
            inlet.ConstantInlet(t, f, c[np.newaxis, :], species, inlet_id, gui_title)
        with self.assertRaises(AssertionError):
            inlet.ConstantInlet(t, f, c, ['c1'], inlet_id, gui_title)

        # refresh
        i.f = 5
        i.c = np.array([6, 7])
        f_ref = np.ones_like(t) * i.f
        c_ref = np.ones_like(i._c_out) * i.c[:, np.newaxis]
        self.assertNotEqual(f_ref[0], i._f_out[0])
        self.assertNotEqual(c_ref[0][0], i._c_out[0][0])
        i._refresh()
        np.testing.assert_array_almost_equal(f_ref, i._f_out)
        np.testing.assert_array_almost_equal(c_ref, i._c_out)

    # noinspection DuplicatedCode,DuplicatedCode
    def test_interval_inlet(self):
        t = np.linspace(0, 10, 100)
        f = 2.4
        c_inner = np.array([2, 3.1])
        c_outer = np.array([6, 7.1])
        species = ['c1', 'c2']
        inlet_id = 'interval_inlet'
        gui_title = 'Interval Inlet'

        i = inlet.IntervalInlet(t, f, c_inner, c_outer, species, inlet_id, gui_title)
        self.check_parent_init(i, species, inlet_id, gui_title)

        self.assertEqual(f, i.f)
        np.testing.assert_array_equal(c_inner, i.c_inner)
        np.testing.assert_array_equal(c_outer, i.c_outer)

        # make sure it's copy
        c_inner += 1
        c_outer += 2
        np.testing.assert_array_almost_equal(c_inner, i.c_inner + 1)
        np.testing.assert_array_almost_equal(c_outer, i.c_outer + 2)
        c_inner -= 1
        c_outer -= 2

        # assertions
        with self.assertRaises(AssertionError):
            inlet.IntervalInlet(t, f, np.array([2]), c_outer, species, inlet_id, gui_title)
        with self.assertRaises(AssertionError):
            inlet.IntervalInlet(t, f, c_inner, np.array([2]), species, inlet_id, gui_title)

        # refresh
        i.f = 5
        i.t_start = 1.2
        i.t_end = 8.03
        i.c_inner = np.array([1, 2.2])
        i.c_outer = np.array([6, 7])
        f_ref = np.ones_like(t) * i.f
        c_ref = np.ones_like(i._c_out) * i.c_inner[:, np.newaxis]
        c_ref[:, t >= i.t_end] = i.c_outer[:, np.newaxis]
        c_ref[:, t < i.t_start] = i.c_outer[:, np.newaxis]
        self.assertNotEqual(f_ref[0], i._f_out[0])
        self.assertNotEqual(c_ref[0][0], i._c_out[0][0])
        i._refresh()
        np.testing.assert_array_almost_equal(f_ref, i._f_out)
        np.testing.assert_array_almost_equal(c_ref, i._c_out)

    def test_custom_inlet(self):
        t = np.linspace(0, 10, 100)
        f = np.random.rand(t.size)
        c = np.random.rand(2, t.size)
        species = ['c1', 'c2']
        inlet_id = 'custom_inlet'
        gui_title = 'Custom Inlet'

        i = inlet.CustomInlet(t, f, c, species, inlet_id, gui_title)
        self.check_parent_init(i, species, inlet_id, gui_title)

        np.testing.assert_array_equal(f, i.f)
        np.testing.assert_array_equal(c, i.c)

        # make sure it's copy
        c_old = c.copy()
        c += 2
        f_old = f.copy()
        f += 1
        np.testing.assert_array_almost_equal(f, i.f)
        np.testing.assert_array_almost_equal(c, i.c)
        np.testing.assert_array_almost_equal(f_old, i._f_out)
        np.testing.assert_array_almost_equal(c_old, i._c_out)
        i._refresh()
        np.testing.assert_array_almost_equal(f, i._f_out)
        np.testing.assert_array_almost_equal(c, i._c_out)

        # assertions
        with self.assertRaises(AssertionError):
            inlet.CustomInlet(t, f, c[:, 2:], species, inlet_id, gui_title)
        with self.assertRaises(AssertionError):
            inlet.CustomInlet(t, f, c, ['c1'], inlet_id, gui_title)

