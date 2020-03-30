__all__ = ['CSTR']
__version__ = '0.1'
__author__ = 'Jure Sencar'

import numpy as _np

import bio_rtd.core as _core
import bio_rtd.utils as _utils
import bio_rtd.peak_shapes as _peak_shapes


class CSTR(_core.UnitOperation):
    """
    Simulation of CSTR with ideal mixing

    Target upper fill volume can be defined via
    - `v_void` : upper target volume
    - `v_min` : minimum target volume (if periodic_inlet == True)
    - `v_min_ratio` : the ratio between v_min and v_void (if periodic_inlet == True)
    - `rt_target` : target residence time (_target_upper_fill_volume = rt_target * _f_out_target)
    If `v_void == -1`, then the `v_min` is considered
    If `v_min == -1`, then the `v_min_ratio` is considered
    If `v_min_ratio == -1`, then the `rt_target` is considered
    If `rt_target == -1`, then the previous `_target_upper_fill_volume` is used

    Initial fill volume can be defined via
    - v_init : init fill volume (_v_init = v_init)
    - v_init_ratio : the ratio between _v_init and v_void
    If v_init < 0, then the v_init_ratio is considered
    If v_init_ratio < 0, then the _v_init = v_void
    The concentration of pre-filled part is defined by `c_init`

    """

    def __init__(self, t: _np.ndarray, uo_id: str, gui_title: str = "CSTR"):
        super().__init__(t, uo_id, gui_title)

        # CSTR volume definition (one of those should be positive)
        self.rt_target = -1
        self.v_void = -1
        self.v_min = -1
        self.v_min_ratio = -1

        # initial volume definition
        self.v_init = -1
        self.v_init_ratio = -1
        # initial concentration profile in CSTR (if pre-filled)
        self.c_init = _np.array([])
        # if True it overrides `v_init` and `v_init_ratio`
        self.starts_empty = False

    def _calc_f_out_target_and_t_cycle(self):
        # determine if it is boxed shaped
        self._is_f_in_box_shaped = self._is_flow_box_shaped()
        # calc target out flow
        if self._is_f_in_box_shaped:
            self._f_out_target = self._f.max()
            self._t_cycle = 0
        else:
            self._f_out_target, self._t_cycle = self._estimate_steady_state_mean_f()

    def _calc_v_void(self):
        # calc cstr max volume (_v_void)
        self._ensure_single_non_negative_parameter(
            self.log.WARNING, log_level_none=self.log.ERROR,
            v_void=self.v_void,
            v_min=self.v_min,
            v_min_ratio=self.v_min_ratio,
            rt_target=self.rt_target
        )
        if self.v_void >= 0:
            self._v_void = self.v_void
        elif self.v_min >= 0:
            assert self._t_cycle > 0, "`v_min_ratio` can only de defined for periodic inlet flow rate"
            f_in = self._f.max()
            f_out = self._f_out_target
            dv = f_out * self._t_cycle * (1 - f_out / f_in)
            self._v_void = self.v_min + dv
        elif self.v_min_ratio >= 0:
            assert self._t_cycle > 0, "`v_min_ratio` can only de defined for periodic inlet flow rate"
            f_in = self._f.max()
            f_out = self._f_out_target
            dv = f_out * self._t_cycle * (1 - f_out / f_in)
            self._v_void = dv / (1 - self.v_min_ratio)
        else:  # self.rt_target >= 0
            self._v_void = self.rt_target * self._f_out_target

    def _calc_v_init(self):
        # calc initial volume `_v_init`
        if self.starts_empty:
            self._v_init = 0
        else:
            if self.v_init >= 0:
                self._v_init = self.v_init
                if self.v_init_ratio >= 0:
                    self.log.w("Initial volume is already defined by `v_init` (`v_init_ratio` is ignored)")
            elif self.v_init_ratio >= 0:
                assert hasattr(self, '_v_void') and self._v_void > 0, "`_v_void` should be defined by now"
                self._v_init = self.v_init_ratio * self._v_void
            else:
                assert hasattr(self, '_v_void') and self._v_void > 0, "`_v_void` should be defined by now"
                self.log.w("Initial volume for CSTR is undefined. Using (`v_init_ratio` = 1)")
                self._v_init = self._v_void

    def _calc_c_init(self):
        # calc initial concentration `_c_init`
        if self.c_init.size == 0:
            self._c_init = _np.zeros([self._n_species, 1])
        elif self.c_init.size == self._n_species:
            self._c_init = self.c_init.reshape(self._n_species, 1)
        else:
            raise ValueError("`c_init` should have one element for each component")

    def _sim_convolution(self):
        """ Use convolution instead of iterative numerical simulation. """

        # assertion
        assert hasattr(self, '_f_out_target') and self._f_out_target > 0
        assert hasattr(self, '_v_void') and self._v_void > 0
        assert hasattr(self, '_c_init') and self._c_init.size == self._c.shape[0] == self._n_species
        assert hasattr(self, '_is_f_in_box_shaped') and self._is_f_in_box_shaped

        # rt_mean
        rt_mean = self._v_void / self._f_out_target
        # time vector
        t_rtd = _np.arange(0, min(rt_mean * 10, self._t[-1]), self._dt)
        # exponential decay probability function
        p_rtd = _peak_shapes.tanks_in_series(t_rtd, rt_mean, 1, self.log)
        # apply convolution
        self._c = _utils.convolution.time_conv(self._dt, self._c, p_rtd, self._c_init)

        # log data
        self.log.i_data(self._log_tree, "rt_mean", rt_mean)
        self.log.d_data(self._log_tree, "p_rtd", p_rtd)

    def _sim_numerical(self):

        # assertions
        assert hasattr(self, '_v_void') and self._v_void > 0
        assert hasattr(self, '_v_init') and self._v_init >= 0
        assert hasattr(self, '_c_init') and self._c_init.size == self._c.shape[0] == self._n_species
        assert hasattr(self, '_f_out_target') and self._f_out_target > 0

        # status in cstr
        v = self._v_init
        m = self._c_init.flatten() * v

        # do not turn off outlet once started
        keep_outlet_on = False

        # when the flow starts
        i_f_on = _utils.vectors.true_start(self._f > 0)
        # set c to zero when flow is off
        self._c[:, :i_f_on] = 0

        for i in range(i_f_on, self._t.size):
            # fill in
            dv = self._f[i] * self._dt
            v += dv
            m += self._c[:, i] * dv

            if v < self._f_out_target * self._dt:  # CSTR is empty
                if not keep_outlet_on:
                    self._c[:, i] = 0
                    self._f[i] = 0
                else:
                    # CSTR dry during operation -> shout it down
                    self.log.i("CSTR ran dry during operation -> shutting down")
                    self._c[:, i:] = 0
                    self._f[i:] = 0
                    return
            elif v < self._v_void and not keep_outlet_on:  # wait until filled
                self._c[:, i] = 0
                self._f[i] = 0
            else:
                keep_outlet_on = True
                # calc current concentration
                c = m / v
                # get outlet
                self._c[:, i] = c
                self._f[i] = self._f_out_target
                # subtract outlet from cstr
                v -= self._f_out_target * self._dt
                m -= self._f_out_target * self._dt * c

    def _calculate(self):

        if _np.all(self._f <= 0):
            self.log.w("Inlet flow rate is zero")
            return

        # prepare
        self._calc_f_out_target_and_t_cycle()
        self._calc_v_void()
        self._calc_v_init()
        self._calc_c_init()

        # simulation
        if self._v_init == self._v_void and self._is_f_in_box_shaped:
            self._sim_convolution()
        else:
            self._sim_numerical()

        # log data
        self.log.i_data(self._log_tree, "f_out_target", self._f_out_target)
        self.log.i_data(self._log_tree, "t_cycle", self._t_cycle)
        self.log.i_data(self._log_tree, "v_void", self._v_void)
        self.log.i_data(self._log_tree, "v_init", self._v_init)
        self.log.i_data(self._log_tree, "c_init", self._c_init)
