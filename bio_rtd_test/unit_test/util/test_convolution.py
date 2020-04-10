import unittest

import numpy as np

from bio_rtd.utils import convolution
from bio_rtd.utils import vectors
from bio_rtd import peak_shapes
from bio_rtd_test.aux_bio_rtd_test import TestLogger


class TimeConvTest(unittest.TestCase):

    def setUp(self) -> None:
        self.t = np.linspace(0, 100, 1500)
        self.dt = self.t[1] - self.t[0]
        self.log = TestLogger()
        self.rtd_rt_mean = 50
        self.rtd = peak_shapes.emg(self.t, rt_mean=self.rtd_rt_mean, sigma=20, skew=0.5)
        self.rtd_norm = self.rtd / self.rtd.sum() / self.dt
        self.assertAlmostEqual(self.rtd_norm.sum() * self.dt, 1)
        # define pre-fill
        self.pre_fill = np.ones([3, 1])
        self.pre_fill[1] = 0
        self.pre_fill[2] = 0.23
        # define wash
        self.wash = np.ones([3, 1])
        self.wash[0] = 0.45
        self.wash[1] = 0.32
        # define in c profile
        self.c_in = np.ones([3, self.t.size])
        self.c_in[0, :] = 0.1
        self.c_in[1, :] = 0.5
        self.c_in[:, -self.rtd.size:] = 0  # clip the end in order to prevent the spill

    def test_time_conv(self):

        # empty c_in -> empty c_out
        with self.assertWarns(Warning, msg="Info: Convolution: Got empty c_in"):
            c_out = convolution.time_conv(self.dt, np.array([]), self.rtd, None, logger=self.log)
            self.assertTrue(c_out.size == 0)

        # empty bio_rtd in -> c_out == c_in
        with self.assertWarns(Warning, msg="Convolution: Got empty bio_rtd"):
            c_in = np.ones([1, 10])
            c_out = convolution.time_conv(self.dt, c_in, np.array([]), None, logger=self.log)
            np.testing.assert_array_equal(c_out, c_in)

        def calc_target_c_out(_c_in, _rtd, _pre_fill=None):
            _c_out_t = np.empty_like(_c_in)
            _c_ext = np.pad(_c_in, ((0, 0), (_rtd.size, 0)), mode="constant")
            if _pre_fill is not None:
                _c_ext[:, :_rtd.size] = _pre_fill
            for j in range(_c_out_t.shape[0]):
                _c_out_t[j] = np.convolve(_c_ext[j], _rtd)[_rtd.size:_rtd.size + _c_in.shape[1]] * self.dt
            return _c_out_t

        # no pre-fill, normalized bio_rtd
        # generate target
        c_out_target = calc_target_c_out(self.c_in, self.rtd_norm)
        # make sure target has the same sum as input
        np.testing.assert_array_equal(self.c_in.sum(1), c_out_target.sum(1))
        # make sure target has the same shape as input
        self.assertEqual(self.c_in.shape, c_out_target.shape)
        # test
        c_out = convolution.time_conv(self.dt, self.c_in, self.rtd_norm, None, logger=self.log)
        np.testing.assert_array_equal(c_out, c_out_target)
        c_out = convolution.time_conv(self.dt, self.c_in, self.rtd_norm, np.zeros_like(self.pre_fill), logger=self.log)
        np.testing.assert_array_equal(c_out, c_out_target)

        # no pre-fill and non-normalized bio_rtd
        # generate target
        c_out_target = calc_target_c_out(self.c_in, self.rtd)
        # make sure target has the same sum as input
        np.testing.assert_array_equal(self.c_in.sum(1), c_out_target.sum(1) / self.rtd.sum() / self.dt)
        # make sure target has the same shape as input
        self.assertEqual(self.c_in.shape, c_out_target.shape)
        # test
        c_out = convolution.time_conv(self.dt, self.c_in, self.rtd, None, logger=self.log)
        np.testing.assert_array_equal(c_out, c_out_target)

        # pre-fill, normalized bio_rtd
        # generate target
        c_out_target = calc_target_c_out(self.c_in, self.rtd_norm, self.pre_fill)
        # make sure target has the same sum as input + pre-fill
        np.testing.assert_array_almost_equal(
            self.pre_fill.flatten() * self.rtd_rt_mean,
            (c_out_target.sum(1) - self.c_in.sum(1)) * self.dt,
            2
        )
        # make sure target has the same shape as input
        self.assertEqual(self.c_in.shape, c_out_target.shape)
        # test
        c_out = convolution.time_conv(self.dt, self.c_in, self.rtd_norm, self.pre_fill, logger=self.log)
        np.testing.assert_array_equal(c_out, c_out_target)

        # pre-fill, non-normalized bio_rtd
        # generate target
        c_out_target = calc_target_c_out(self.c_in, self.rtd, self.pre_fill)
        # make sure target has the same sum as input + pre-fill
        np.testing.assert_array_almost_equal(
            self.c_in.sum(1) * self.dt + self.pre_fill.flatten() * self.rtd_rt_mean,
            c_out_target.sum(1) / self.rtd.sum(),
            2
        )
        # make sure target has the same shape as input
        self.assertEqual(self.c_in.shape, c_out_target.shape)
        # test
        c_out = convolution.time_conv(self.dt, self.c_in, self.rtd, self.pre_fill, logger=self.log)
        np.testing.assert_array_equal(c_out, c_out_target)

    def test_piece_wise_time_conv(self):
        # prepare
        f_in = np.ones(self.c_in.shape[1]) * 3.5
        i_f_in_delay = 20
        f_in[:i_f_in_delay] = 0
        i_f_in_stop = f_in.size - 200
        f_in[i_f_in_stop:] = 0
        t_cycle = 15.63
        i_cycle = int(round(t_cycle / self.dt))
        rt_mean = 4.54
        i_rt_mean = int(round(rt_mean / self.dt))

        # assertions
        with self.assertRaises(AssertionError):
            convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in[:-2], c_in=self.c_in,
                                                        t_cycle=t_cycle, rt_mean=rt_mean, rtd=self.rtd,
                                                        c_equilibration=None, logger=self.log)
        with self.assertRaises(AssertionError):
            convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=self.c_in, t_cycle=-1,
                                                        rt_mean=rt_mean, rtd=self.rtd, c_equilibration=None,
                                                        logger=self.log)
        with self.assertRaises(AssertionError):
            convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=self.c_in, t_cycle=t_cycle,
                                                        rt_mean=-1, rtd=self.rtd, c_equilibration=None,
                                                        logger=self.log)

        # empty c_in -> empty c_out
        with self.assertWarns(Warning, msg="Info: Convolution: Got empty c_in"):
            c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=np.array([]), c_in=np.array([[]]),
                                                                t_cycle=t_cycle, rt_mean=rt_mean, rtd=self.rtd,
                                                                c_equilibration=None, logger=self.log)
            self.assertTrue(c_out.size == 0)

        # empty bio_rtd in -> c_out == c_in
        with self.assertWarns(Warning, msg="Convolution: Got empty bio_rtd"):
            c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=self.c_in, t_cycle=t_cycle,
                                                                rt_mean=rt_mean, rtd=np.array([]), c_equilibration=None,
                                                                logger=self.log)
            np.testing.assert_array_equal(c_out, self.c_in)

        # empty bio_rtd in -> c_out == c_in
        with self.assertWarns(Warning, msg="Info: Convolution: Got empty f_in"):
            c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in * 0, c_in=self.c_in,
                                                                t_cycle=t_cycle,
                                                                rt_mean=rt_mean, rtd=self.rtd, c_equilibration=None,
                                                                logger=self.log)
            np.testing.assert_array_equal(c_out, np.zeros_like(self.c_in))

        # prepare inlet profiles
        peak_1 = np.zeros_like(self.c_in)
        peak_1[0, i_f_in_delay:i_f_in_delay + i_cycle] = 1
        c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=peak_1, t_cycle=t_cycle,
                                                            rt_mean=rt_mean, rtd=self.rtd, c_equilibration=None,
                                                            logger=self.log)
        c_out_target = convolution.time_conv(dt=self.dt, c_in=peak_1, rtd=self.rtd, c_equilibration=None,
                                             logger=self.log)
        # same as convolution without switching when the outlet for the cycle with material is 'on'
        np.testing.assert_array_almost_equal(
            c_out[:, i_f_in_delay + i_rt_mean:i_f_in_delay + i_rt_mean + i_cycle],
            c_out_target[:, i_f_in_delay + i_rt_mean:i_f_in_delay + i_rt_mean + i_cycle]
        )
        # zero around the cycle outlet
        self.assertAlmostEqual(c_out[:, :i_f_in_delay + i_rt_mean].sum(), 0)
        self.assertAlmostEqual(c_out[:, i_f_in_delay + i_rt_mean + i_cycle:].sum(), 0)

        # same test with init fill
        pre_fill = np.array([[0.12], [3.45], [0]])
        c_out_prev = c_out.copy()
        c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=peak_1, t_cycle=t_cycle,
                                                            rt_mean=rt_mean, rtd=self.rtd, c_equilibration=pre_fill,
                                                            logger=self.log)
        c_out_target = convolution.time_conv(dt=self.dt, c_in=peak_1[:, i_f_in_delay:], rtd=self.rtd,
                                             c_equilibration=pre_fill, logger=self.log)
        # same as convolution without switching when the outlet for the cycle with material is 'on'
        np.testing.assert_array_almost_equal(
            c_out[:, i_f_in_delay + i_rt_mean:i_f_in_delay + i_rt_mean + i_cycle],
            c_out_target[:, i_rt_mean:i_rt_mean + i_cycle]
        )
        # just pre fill around the cycle
        np.testing.assert_array_almost_equal(
            c_out[0, :i_f_in_delay + i_rt_mean] / pre_fill[0],
            c_out[1, :i_f_in_delay + i_rt_mean] / pre_fill[1]
        )
        np.testing.assert_array_almost_equal(
            c_out[0, i_f_in_delay + i_rt_mean + i_cycle:] / pre_fill[0],
            c_out[1, i_f_in_delay + i_rt_mean + i_cycle:] / pre_fill[1]
        )
        self.assertAlmostEqual(c_out[0, i_f_in_delay + i_rt_mean + i_cycle:].sum(), 134.88, 2)
        self.assertAlmostEqual(c_out[1].sum() / pre_fill[1, 0] * pre_fill[0, 0], c_out[0].sum() - c_out_prev[0].sum())
        self.assertEqual(c_out[2].sum(), 0)

        def calc_switch_times():
            _i_start, _i_end = vectors.true_start_and_end(f_in > 0)
            _i_switch_inlet = [int(round(i)) for i in np.arange(_i_start, _i_end, t_cycle / self.dt)]
            _i_switch_inlet_off = _i_switch_inlet[1:] + [_i_end]
            _i_switch_outlet = [min(i + i_rt_mean, f_in.size) for i in _i_switch_inlet]
            _i_switch_outlet_off = _i_switch_outlet[1:] + [min(_i_switch_outlet[-1] + i_cycle, f_in.size)]
            return _i_start, _i_end, _i_switch_inlet, _i_switch_inlet_off, _i_switch_outlet, _i_switch_outlet_off

        # test ending
        peak_2 = np.zeros_like(peak_1)
        peak_2[0, i_f_in_stop - int(i_cycle / 2):] = 1
        # calc switch times
        i_start, i_end, i_switch_inlet, i_switch_inlet_off, i_switch_outlet, i_switch_outlet_off = calc_switch_times()
        peak_2[:, :i_switch_inlet[-1]] = 0
        # target
        c_ref = peak_2.copy()
        c_ref[:, i_f_in_stop:] = 0
        c_out_target = convolution.time_conv(dt=self.dt, c_in=c_ref, rtd=self.rtd, logger=self.log)
        # calc
        c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=peak_2, t_cycle=t_cycle,
                                                            rt_mean=rt_mean, rtd=self.rtd, logger=self.log)
        # same as convolution without switching after the outlet for the cycle with material is 'on'
        np.testing.assert_array_almost_equal(
            c_out[:, i_switch_outlet[-1]:i_switch_outlet_off[-1]],
            c_out_target[:, i_switch_outlet[-1]:i_switch_outlet_off[-1]]
        )
        self.assertEqual(c_out[:, :i_switch_outlet[-1]].sum(), 0)

        # test ending with wash
        # target
        c_ref[:, i_f_in_stop:] = self.wash
        c_out_target = convolution.time_conv(dt=self.dt, c_in=c_ref, rtd=self.rtd, logger=self.log)
        # calc
        c_out = convolution.piece_wise_time_conv(dt=self.dt, f_in=f_in, c_in=peak_2, t_cycle=t_cycle,
                                                            rt_mean=rt_mean, rtd=self.rtd, c_wash=self.wash,
                                                            logger=self.log)
        # same as convolution without switching after the outlet for the cycle with material is 'on'
        np.testing.assert_array_almost_equal(
            c_out[:, i_switch_outlet[-1]:i_switch_outlet_off[-1]],
            c_out_target[:, i_switch_outlet[-1]:i_switch_outlet_off[-1]]
        )
        self.assertTrue(c_out[:, :i_switch_outlet[-1]].sum() > 0)
        np.testing.assert_array_almost_equal(
            c_out[0, :i_switch_outlet[-1]] / self.wash[0],
            c_out[1, :i_switch_outlet[-1]] / self.wash[1]
        )
