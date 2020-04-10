"""Surge tanks.

Unit operations that accept various flow rate profiles and provide
constant or box-shaped profile.

"""

__all__ = ['CSTR', 'TwoAlternatingCSTRs']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as _np

import bio_rtd.core as _core
import bio_rtd.utils as _utils
import bio_rtd.peak_shapes as _peak_shapes


class CSTR(_core.UnitOperation):
    """Simulation of CSTR with ideal mixing.

    Accepts constant, box-shaped or box-shaped periodic flow rate
    profiles.

    Provides constant or box-shaped flow rate profile.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "CSTR".

    Notes
    -----
    Target upper fill volume can be defined via:

    - :attr:`v_void`
    - :attr:`rt_target`
    - :attr:`v_min`
    - :attr:`v_min_ratio`

    Initial fill volume can be defined via:

    - :attr:`v_init`
    - :attr:`v_init_ratio`
    - :attr:`starts_empty` -> If `True` it overrides :attr:`v_init` and
      :attr:`v_init_ratio`

    The concentration of pre-filled part is defined by :attr:`c_init`.
    Empty array (default) means that all components are 0.

    Examples
    --------
    >>> t = _np.linspace(0, 100, 1001)  # min
    >>> cstr = CSTR(t, uo_id="sample_cstr")
    >>> # Size of the surge tank.
    >>> cstr.v_void = 140  # mL
    >>> cstr.starts_empty = True  # optional
    >>> f_in = _np.zeros_like(t)
    >>> f_in[0:800] = 12.5  # mL/min
    >>> c_in = _np.zeros([1, t.size])
    >>> c_in[0][0:800] = 2.5  # mg/mL
    >>> f_out, c_out = cstr.evaluate(f_in, c_in)
    >>> f_out
    array([0., 0., 0., ..., 0., 0., 0.])
    >>> f_out[105:115]
    array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , 12.5, 12.5, 12.5, 12.5])
    >>> c_out
    array([[0., 0., 0., ..., 0., 0., 0.]])
    >>> c_out[0][105:115]
    array([0. , 0. , 0. , 0. , 0. , 0. , 2.5, 2.5, 2.5, 2.5])
    >>> # Pre-filled with buffer (and no product).
    >>> cstr.starts_empty = False
    >>> cstr.v_init_ratio = 1  # starts completely pre-filled (default)
    >>> f_out, c_out = cstr.evaluate(f_in, c_in)
    >>> f_out
    array([12.5, 12.5, 12.5, ...,  0. ,  0. ,  0. ])
    >>> c_out
    array([[0.02232143, 0.04444445, 0.06637082, ..., 0.42450783, 0.42073445,
            0.41699166]])

    """

    def __init__(self, t: _np.ndarray, uo_id: str, gui_title: str = "CSTR"):
        super().__init__(t, uo_id, gui_title)
        # CSTR volume definition (one of those should be positive).
        self.v_void: float = -1
        """Target upper fluid volume (max fill level) in the CSTR."""
        self.rt_target: float = -1
        """Target mean residence time in the CSTR.
        
        Max fill level = `rt_target` * outlet flow rate.
        
        """
        self.v_min: float = -1
        """Target lowest fill level in the CSTR.
        
        Only valid for periodic inlet flow rate profile.
        
        Upper fill level is calculated at runtime.
        
        """
        self.v_min_ratio: float = -1
        """Ratio between max and min fill levels in the CSTR.
        
        Only valid for periodic inlet flow rate profile.
        
        Absolute fill levels are calculated at runtime.
        
        """
        # Initial volume definition.
        self.v_init: float = -1
        """Initial pre-fill level - absolute value."""
        self.v_init_ratio: float = -1
        """Initial pre-fill level as a share of max fill level."""
        self.starts_empty: bool = False
        """CSTR has a 0 initial fill level.
        
        If `True` it overrides `v_init` and `v_init_ratio`.
        
        """
        self.c_init = _np.array([])
        """Pre-fill buffer composition. Default = array([]) = all 0"""

    def _calc_f_out_target_and_t_cycle(self):
        """Estimate target outlet flow rate and inlet cycle duration.

        Target outlet flow rate (`self._f_out_target`).

        Inlet cycle duration (`self._t_cycle`).

        """
        # Determine if it is boxed shaped
        self._is_f_in_box_shaped = self._is_flow_box_shaped()
        # Calc target out flow.
        if self._is_f_in_box_shaped:
            self._f_out_target = self._f.max()
            self._t_cycle = 0
        else:
            self._f_out_target, self._t_cycle = \
                self._estimate_steady_state_mean_f()

    def _calc_v_void(self):
        """Calc cstr max volume (`self._v_void`)"""
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.WARNING, log_level_none=self.log.ERROR,
            v_void=self.v_void,
            v_min=self.v_min,
            v_min_ratio=self.v_min_ratio,
            rt_target=self.rt_target
        )
        if self.v_void >= 0:
            self._v_void = self.v_void
        elif self.v_min >= 0:
            assert self._t_cycle > 0, f"`v_min_ratio` can only de defined" \
                                      f" for periodic inlet flow rate"
            f_in = self._f.max()
            f_out = self._f_out_target
            dv = f_out * self._t_cycle * (1 - f_out / f_in)
            self._v_void = self.v_min + dv
        elif self.v_min_ratio >= 0:
            assert self._t_cycle > 0, f"`v_min_ratio` can only de defined" \
                                      f" for periodic inlet flow rate"
            f_in = self._f.max()
            f_out = self._f_out_target
            dv = f_out * self._t_cycle * (1 - f_out / f_in)
            self._v_void = dv / (1 - self.v_min_ratio)
        else:  # self.rt_target >= 0
            self._v_void = self.rt_target * self._f_out_target

    def _calc_v_init(self):
        """Calc initial volume `self._v_init`."""
        if self.starts_empty:
            self._v_init = 0
        else:
            if self.v_init >= 0:
                self._v_init = self.v_init
                if self.v_init_ratio >= 0:
                    self.log.w(f"Initial volume is already defined by"
                               f" `v_init` (`v_init_ratio` is ignored)")
            elif self.v_init_ratio >= 0:
                assert hasattr(self, '_v_void') and self._v_void > 0, \
                    "`_v_void` should be defined by now"
                self._v_init = self.v_init_ratio * self._v_void
            else:
                assert hasattr(self, '_v_void') and self._v_void > 0, \
                    "`_v_void` should be defined by now"
                self.log.w(f"Initial volume for CSTR is undefined."
                           f" Using (`v_init_ratio` = 1).")
                self._v_init = self._v_void

    def _calc_c_init(self):
        """Calc initial concentration `self._c_init`."""
        if self.c_init.size == 0:
            self._c_init = _np.zeros([self._n_species, 1])
        elif self.c_init.size == self._n_species:
            self._c_init = self.c_init.reshape(self._n_species, 1)
        else:
            raise ValueError(f"`c_init` should have one element"
                             f" for each component")

    def _sim_convolution(self):
        """Convolution instead of iterative numerical simulation."""
        assert hasattr(self, '_f_out_target') and self._f_out_target > 0
        assert hasattr(self, '_v_void') and self._v_void > 0
        assert hasattr(self, '_c_init') \
            and self._c_init.size == self._c.shape[0] == self._n_species
        assert hasattr(self, '_is_f_in_box_shaped') \
            and self._is_f_in_box_shaped
        # Calc `rt_mean`.
        rt_mean = self._v_void / self._f_out_target
        # Time vector for rtd.
        t_rtd = _np.arange(0, min(rt_mean * 10, self._t[-1]), self._dt)
        # Exponential decay probability function.
        p_rtd = _peak_shapes.tanks_in_series(t=t_rtd,
                                             rt_mean=rt_mean,
                                             n_tanks=1,
                                             logger=self.log)
        # Apply convolution.
        self._c = _utils.convolution.time_conv(dt=self._dt,
                                               c_in=self._c,
                                               rtd=p_rtd,
                                               c_equilibration=self._c_init,
                                               logger=self.log)
        # Log data.
        self.log.i_data(self._log_tree, "rt_mean", rt_mean)
        self.log.d_data(self._log_tree, "p_rtd", p_rtd)

    def _sim_numerical(self):
        """Iterative numerical simulation."""
        assert hasattr(self, '_v_void') and self._v_void > 0
        assert hasattr(self, '_v_init') and self._v_init >= 0
        assert hasattr(self, '_c_init') \
            and self._c_init.size == self._c.shape[0] == self._n_species
        assert hasattr(self, '_f_out_target') and self._f_out_target > 0
        # Status in cstr.
        v = self._v_init
        m = self._c_init.flatten() * v
        # Do not turn off outlet once started.
        keep_outlet_on = False
        # When the flow starts.
        i_f_on = _utils.vectors.true_start(self._f > 0)
        # Set c to zero when flow is off.
        self._c[:, :i_f_on] = 0

        for i in range(i_f_on, self._t.size):
            # Fill in.
            dv = self._f[i] * self._dt
            v += dv
            m += self._c[:, i] * dv

            if v < self._f_out_target * self._dt:  # CSTR is empty
                if not keep_outlet_on:
                    self._c[:, i] = 0
                    self._f[i] = 0
                else:
                    # CSTR dry during operation -> shout it down.
                    self.log.i(f"CSTR ran dry during operation"
                               f" -> shutting down")
                    self._c[:, i:] = 0
                    self._f[i:] = 0
                    return
            elif v < self._v_void and not keep_outlet_on:  # wait until filled
                self._c[:, i] = 0
                self._f[i] = 0
            else:
                keep_outlet_on = True
                # Calc current concentration.
                c = m / v
                # Get outlet.
                self._c[:, i] = c
                self._f[i] = self._f_out_target
                # Subtract outlet from cstr.
                v -= self._f_out_target * self._dt
                m -= self._f_out_target * self._dt * c

    def _calculate(self):
        # Prepare.
        self._calc_f_out_target_and_t_cycle()
        self._calc_v_void()
        self._calc_v_init()
        self._calc_c_init()
        # Simulation.
        if self._v_init == self._v_void and self._is_f_in_box_shaped:
            self._sim_convolution()
        else:
            self._sim_numerical()
        # Log data.
        self.log.i_data(self._log_tree, "f_out_target", self._f_out_target)
        self.log.i_data(self._log_tree, "t_cycle", self._t_cycle)
        self.log.i_data(self._log_tree, "v_void", self._v_void)
        self.log.i_data(self._log_tree, "v_init", self._v_init)
        self.log.i_data(self._log_tree, "c_init", self._c_init)


class TwoAlternatingCSTRs(_core.UnitOperation):
    """Simulation of Two alternating CSTRs with ideal mixing.

    Accepts constant, box-shaped or periodic box-shaped flow rate
    profiles.

    Provides constant or box-shaped flow rate profile.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "TwoAlternatingCSTRs".

    Notes
    -----
    Define how many periods a CSTR collects in one cycle
    (applies to periodic inlet only):

    - :attr:`collect_n_periods` - default = 1

    Define when the switch occurs for periodic inlet:

    - :attr:`relative_role_switch_time` - default = 0.9

    Define when the switch occurs for constant inlet
    (one should be defined):

    - :attr:`t_cycle`
    - :attr:`v_cycle`

    Define leftover volume after discharge (optional, max one):

    - :attr:`v_leftover`
    - :attr:`v_leftover_rel`

    """

    def __init__(self, t: _np.ndarray, uo_id: str,
                 gui_title: str = "TwoAlternatingCSTRs"):
        super().__init__(t, uo_id, gui_title)
        self.collect_n_periods: int = 1
        """How many periods of a periodic inlet a surge tank collects.
        
        Default = 1.
         
        Only relevant for periodic inlet flow rate profile.
        
        """
        self.relative_role_switch_time: float = 0.9
        """When CSTRs switch roles within the inlet flow rate off time.
        
        Default = 0.9.
        
        This is valid for the first cycle. In case of leftover material,
        the first two cycles are shorter, thus the switch occurs earlier 
        in following cycles.
        
        In case of inlet periodic box-shaped flow rate profile,
        the CSTRs can their switch role (providing or collecting)
        at any time when the inlet flow rate if off.
        
        In terms of RTD it is desirable to start releasing material as
        soon as collection phase ends. However additional safety margin
        might be applied in order to allow some cycle-to-cycle
        variability in a real process.
        
        """
        self.t_cycle: float = -1
        """The duration (time) of cycle after which CSTRs switch roles.
        
        In case of constant inlet profile, one of `t_cycle` and
        :attr:`v_cycle` needs to be defined.
        
        In case of periodic inlet flow rate,
        this value should not be defined.
        
        """
        self.v_cycle: float = -1
        """Collected volume after which CSTRs switch roles.
        
        In case of constant inlet profile, one of `v_cycle` and
        :attr:`t_cycle` needs to be defined.
        
        In case of periodic inlet flow rate,
        this value should not be defined.
        
        """
        self.v_leftover: float = -1
        """What amount of fluid stays in CSTR after discharge.
        
        Only one of `v_leftover` and
        :attr:`v_leftover_rel` should be defined.
        If none are defined, then the leftover is set to 0.
        
        """
        self.v_leftover_rel: float = -1
        """Relative amount of fluid that stays in CSTR after discharge.
        
        Relative to the amount of collected material in one cycle.
        
        Only one of `v_leftover_rel` and
        :attr:`v_leftover` should be defined.
        If none are defined, then the leftover is set to 0.
        
        """

    def _calc_f_out_target_and_t_cycle(self):
        """Estimate target outlet flow rate and inlet cycle duration.

        Target outlet flow rate (`self._f_out_target`).

        Inlet cycle duration (`self._t_cycle`).

        """
        if self._is_flow_box_shaped():
            # Constant inlet.
            self._f_out_target = self._f.max()
            assert (self.t_cycle > 0) is not (self.v_cycle > 0), \
                f"Exactly one of `t_cycle` and `v_cycle` needs to be defined" \
                f" for constant inlet flow rate."
            self._t_cycle = self.t_cycle if self.t_cycle > 0 \
                else self.v_cycle / self._f_out_target
        else:
            # Periodic inlet.
            self._f_out_target, self._t_cycle = \
                self._estimate_steady_state_mean_f()
            # Apply multiple periods collection.
            self._t_cycle *= self.collect_n_periods
        # Log.
        self.log.i_data(self._log_tree, "t_cycle", self._t_cycle)
        self.log.i_data(self._log_tree, "f_out_target", self._f_out_target)

    def _calc_t_leftover(self):
        assert hasattr(self, "_t_cycle")
        assert hasattr(self, "_f_out_target")
        assert self.v_leftover <= 0 or self.v_leftover_rel <= 0, \
            f"Only one of the `v_leftover` and `v_leftover_rel`" \
            f" should be defined."
        if self.v_leftover > 0:
            self._t_leftover = self.v_leftover / self._f_out_target
        elif self.v_leftover_rel > 0:
            self._t_leftover = self.v_leftover_rel * self._t_cycle
        else:
            self._t_leftover = 0

    def _calc_i_switch_list(self):
        """Determine when CSTRs switch roles (index on `t` vector)."""
        assert hasattr(self, "_t_cycle")
        assert hasattr(self, "_f_out_target")
        assert hasattr(self, "_t_leftover")
        if self._is_flow_box_shaped():
            # Constant inlet.
            self._i_switch_list = []
            # Get inlet flow rate start and end position.
            i_start, i_end = _utils.vectors.true_start_and_end(self._f > 0)
            i_cycle = self._t_cycle / self._dt
            i_leftover = self._t_leftover / self._dt
            # Manage first two cycles.
            if i_leftover > 0:
                i_start += i_cycle + i_leftover * 2
                self._i_switch_list.append(i_start)
                i_start += i_cycle + i_leftover
                self._i_switch_list.append(i_start)
            # Add following cycles.
            i_start += i_cycle
            while i_start < self._t.size:
                self._i_switch_list.append(i_start)
                i_start += i_cycle
        else:
            # Periodic inlet.
            i_flow_start_list, i_flow_on_duration, _ = \
                self._assert_periodic_flow()
            assert 1 >= self.relative_role_switch_time >= 0
            i_cycle = self._t_cycle / self._dt
            i_delay = (i_cycle - i_flow_on_duration) \
                * self.relative_role_switch_time
            i_leftover = self._t_leftover / self._dt
            i_switch = _np.arange(i_flow_start_list[0], self._t.size, i_cycle)
            assert _np.all(2 > _np.abs(
                i_switch[:len(i_flow_start_list)] - i_flow_start_list))
            # Apply delay.
            i_switch += i_flow_on_duration + i_delay
            # Apply delay due to leftover material
            assert i_delay >= 2 * i_leftover, \
                f"Leftover volume is too large to manage cycles."
            i_switch[1] -= i_leftover
            i_switch[2:] -= 2 * i_leftover
            # Remove entries over the simulation time size.
            self._i_switch_list = [i for i in i_switch if i < self._t.size]
        # Log.
        self.log.i_data(self._log_tree, "t_switch_list", self._i_switch_list)

    def _simulate_cycle_by_cycle(self):
        """Cycle-by-cycle simulation."""
        assert hasattr(self, '_t_cycle')
        assert hasattr(self, '_f_out_target')
        assert hasattr(self, '_i_switch_list')
        # Prepare vectors.
        dv_in = self._f.copy() * self._dt
        dm_in = self._c.copy() * dv_in[_np.newaxis, :]
        dv_out = self._dt * self._f_out_target
        self._f *= 0
        self._c *= 0
        # Prepare log.
        log_data_cycles = list()
        self.log.set_branch(self._log_tree, "cycles", log_data_cycles)
        # Leftover volume.
        v_leftover = self._t_leftover * self._f_out_target
        # Prepare variables.
        v_st1 = 0
        m_st1 = _np.zeros(self._c.shape[0])
        v_st2 = 0
        m_st2 = _np.zeros(self._c.shape[0])
        i_prev_switch = 0
        i_current_switch = self._i_switch_list[0]
        # Iterate over cycles.
        for i_switch_target in [*self._i_switch_list, self._t.size]:
            # Prepare log.
            self._cycle_tree = dict()
            log_data_cycles.append(self._cycle_tree)
            # Collect.
            self.log.i_data(self._cycle_tree, "v_start", v_st1)
            self.log.i_data(self._cycle_tree, "m_start", m_st1)
            i_start = int(_np.floor(i_prev_switch + 1))
            i_d_start = i_start - i_prev_switch  # fraction of fist step
            i_end = int(_np.floor(i_current_switch))
            i_d_end = i_current_switch - i_end  # fraction of last step
            if i_start < self._t.size:
                v_st1 += dv_in[i_start-1] * i_d_start
                m_st1 += dm_in[:, i_start-1] * i_d_start
            v_st1 += dv_in[i_start:i_end].sum()
            m_st1 += dm_in[:, i_start:i_end].sum(1)
            if i_end < self._t.size:
                v_st1 += dv_in[i_end] * i_d_end
                m_st1 += dm_in[:, i_end] * i_d_end
            self.log.i_data(self._cycle_tree, "v_after_collection", v_st1)
            self.log.i_data(self._cycle_tree, "m_after_collection", m_st1)
            self.log.i_data(self._cycle_tree,
                            "c_discharge",
                            m_st1 / v_st1 if v_st1 > 0 else 0)
            # Discharge duration.
            i_discharge_duration = (v_st1 - v_leftover) / dv_out
            i_next_switch = min(i_current_switch + i_discharge_duration,
                                self._t.size)
            i_discharge_duration = i_next_switch - i_current_switch
            # Discharge.
            i_start = int(_np.floor(i_current_switch + 1))
            i_d_start = i_start - i_current_switch  # fraction of fist step
            i_end = int(_np.floor(i_next_switch))
            i_d_end = i_next_switch - i_end  # fraction of last step
            if i_discharge_duration > 0 and i_start <= i_end:
                if i_start <= self._t.size and i_d_start > 0:
                    f0, c0 = self._f[i_start - 1], self._c[:, i_start - 1]
                    df = self._f_out_target * i_d_start
                    c_df = m_st1 / v_st1
                    self._f[i_start - 1] += df
                    self._c[:, i_start - 1] = (c_df * df + c0 * f0) / (df + f0)
                self._c[:, i_start:i_end] = m_st1[:, _np.newaxis] / v_st1
                self._f[i_start:i_end] = self._f_out_target
                if i_end < self._t.size:
                    self._c[:, i_end] = m_st1 / v_st1
                    self._f[i_end] += self._f_out_target * i_d_end
            elif i_discharge_duration > 0:
                assert i_start == i_end + 1
                i_d_mid = i_d_start + i_d_end - 1
                f0, c0 = self._f[i_end], self._c[:, i_end]
                df = self._f_out_target * i_d_mid
                c_df = m_st1 / v_st1
                self._f[i_end] += df
                self._c[:, i_end] = (c_df * df + c0 * f0) / (df + f0)
            if v_st1 > 0:
                v_cycle_filled = v_st1
                v_st1 -= dv_out * i_discharge_duration
                m_st1 *= v_st1 / v_cycle_filled
            self.log.i_data(self._cycle_tree, "v_after_discharge", v_st1)
            self.log.i_data(self._cycle_tree, "m_after_discharge", m_st1)
            self.log.i_data(self._cycle_tree,
                            "i_start_discharge",
                            i_current_switch)
            self.log.i_data(self._cycle_tree,
                            "i_end_discharge",
                            i_next_switch)
            # Prepare for next cycle.
            i_prev_switch = i_current_switch
            i_current_switch = i_next_switch
            if abs(i_discharge_duration) < 1e-09:
                # End of inlet flow. The surge tank ran dry.
                break
            # Ensure that the switch time does not deviate too much.
            assert abs(i_prev_switch - i_switch_target) < 2
            # Switch roles.
            v_st2, m_st2, v_st1, m_st1 = v_st1, m_st1, v_st2, m_st2

    def _ensure_box_shape(self):
        """Ensures clean box-shaped profile.

        It trims start and end positive flow rate value if it is < 99 %
        of median positive flow value.

        Asserts that flow rate variations are < 0.01 % and sets all
        positive values to median flow value.

        """
        i_start, i_end = _utils.vectors.true_start_and_end(self._f > 0)
        f_on = _np.median(self._f[i_start + 1:i_end - 1])
        self._f[i_start] = 0 if self._f[i_start] < 0.99 * f_on else f_on
        self._f[i_end - 1] = 0 if self._f[i_end - 1] < 0.9999 * f_on else f_on
        assert _np.all(self._f[i_start + 1:i_end - 1] > 0.9999 * f_on)
        assert _np.all(self._f[i_start + 1:i_end - 1] < 1.0001 * f_on)
        self._f[i_start + 1:i_end - 1] = f_on

    def _calculate(self):
        # Prepare.
        self._calc_f_out_target_and_t_cycle()
        self._calc_t_leftover()
        self._calc_i_switch_list()
        # Simulation.
        self._simulate_cycle_by_cycle()
        self._ensure_box_shape()
