"""Module with abstract classes."""

__all__ = ['Inlet', 'UnitOperation',
           'RtdModel', 'UserInterface',
           'PDF', 'ChromatographyLoadBreakthrough',
           'ParameterSetList']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from collections import OrderedDict as _OrderedDict

from bio_rtd import logger as _logger
from bio_rtd import adj_par as _adj_par
from bio_rtd import utils as _utils


class DefaultLoggerLogic(_ABC):
    # noinspection PyProtectedMember
    """Default binding of the `RtdLogger` to a class.

    The class holds a reference to a :class:`bio_rtd.logger.RtdLogger`
    instance. When the class receives the instance, it plants a data
    tree into it. If the class is asked to provide the instance before
    it received one, then an instance of
    :class:`bio_rtd.logger.DefaultLogger` is created and passed on.

    Parameters
    ----------
    logger_parent_id
        Custom unique id that belongs to the instance of the class.

        The data tree of this instance is stored in
        :class:`bio_rtd.logger.RtdLogger` under the `logger_parent_id`.

    Examples
    --------
    >>> logger_parent_id = "parent_unit_operation"
    >>> l = DefaultLoggerLogic(logger_parent_id)
    >>> isinstance(l.log, _logger.DefaultLogger)
    True
    >>> # Log error: DefaultLogger raises RuntimeError.
    >>> l.log.e("Error Description")
    Traceback (most recent call last):
    RuntimeError: Error Description
    >>> # Log waring: DefaultLogger prints it.
    >>> l.log.w("Warning Description")
    Warning Description
    >>> # Log info: DefaultLogger ignores it.
    >>> l.log.i("Info")
    >>> l.log.log_data = True
    >>> l.log.log_level = _logger.RtdLogger.DEBUG
    >>> l.log.i_data(l._log_tree, "a", 3)  # store value in logger
    >>> l.log.d_data(l._log_tree, "b", 7)  # store at DEBUG level
    >>> l.log.get_data_tree(logger_parent_id)["b"]
    7
    >>> l.log = _logger.StrictLogger()
    >>> # Log waring: StrictLogger raises RuntimeError.
    >>> l.log.w("Warning Info")
    Traceback (most recent call last):
    RuntimeError: Warning Info

    See Also
    --------
    :class:`bio_rtd.logger.DefaultLogger`

    """

    def __init__(self, logger_parent_id: str):
        self._instance_id = logger_parent_id
        self._log_entity_id = logger_parent_id
        self._logger: _typing.Union[_logger.RtdLogger, None] = None
        self._log_tree = dict()  # place to store logged data

    @property
    def log(self) -> _logger.RtdLogger:
        """Reference of the `RtdLogger` instance.

        Setter also plants instance data tree into passed logger.

        If logger is requested, but not yet set, then a
        :class:`bio_rtd.logger.DefaultLogger` is instantiated.

        """
        if self._logger is None:
            self.log = _logger.DefaultLogger()  # init default logger
        return self._logger

    @log.setter
    def log(self, logger: _logger.RtdLogger):
        self._logger = logger
        self._logger.set_data_tree(self._log_entity_id, self._log_tree)

    def set_logger_from_parent(self, parent_id: str,
                               logger: _logger.RtdLogger):
        """Inherit logger from parent.

        Parameters
        ----------
        parent_id
            Unique identifier of parent instance.
        logger
            Logger from parent instance.
        """
        self._logger = logger
        self._log_entity_id = f"{parent_id}/{self._instance_id}"
        self._logger.set_data_tree(self._log_entity_id,
                                   self._log_tree)


class Inlet(DefaultLoggerLogic, _ABC):
    """Generates starting flow rate and concentration profiles.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    species_list
        List with names of simulating process fluid species.
    inlet_id
        Unique identifier of an instance. It is stored in :attr:`uo_id`.
    gui_title
        Readable title of an instance.

    """

    def __init__(self, t: _np.ndarray, species_list: _typing.Sequence[str],
                 inlet_id: str, gui_title: str):
        super().__init__(inlet_id)  # logger
        
        # Assert proper time vector.
        assert t[0] == 0, "t should start with 0"
        assert len(t.shape) == 1, "t should be a 1D np.ndarray"
        self._t = _np.linspace(0, t[-1], t.size)
        self._dt = t[-1] / (t.size - 1)
        assert _np.all(_np.abs(self._t - t) < 0.001 * self._dt / t.size), \
            "t should have a fixed step size"
        
        # Species
        self.species_list: _typing.Sequence[str] = species_list
        """List with names of simulating process fluid species."""
        self._n_species = len(self.species_list)
        
        # Strings
        self.uo_id: str = inlet_id
        """Unique identifier of the instance."""
        self.gui_title: str = gui_title
        """Human readable title (for plots)."""
        
        # Placeholders
        self.adj_par_list: _typing.Sequence[_adj_par.AdjustableParameter] = ()
        """List of adjustable parameters exposed to the GUI."""
        
        # Outputs
        self._f_out = _np.zeros_like(t)
        self._c_out = _np.zeros([self._n_species, t.size])

    def get_t(self) -> _np.ndarray:
        """Get simulation time vector."""
        return self._t

    def get_n_species(self) -> int:
        """Get number of process fluid species."""
        return self._n_species

    @_abstractmethod
    def refresh(self):  # pragma: no cover
        """Updates output profiles.

        Internally it updates `self._f_out` and `self._c_out` based on
        instance attribute values.

        """
        pass

    def get_result(self) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Get flow rate and concentration profiles.

        Returns
        -------
        f_out
            Flow rate profile.
        c_out
            Concentration profile.
        """
        return self._f_out, self._c_out


class UnitOperation(DefaultLoggerLogic, _ABC):
    """Processes flow rate and concentration profiles.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI.

    """

    def __init__(self, t: _np.ndarray, uo_id: str, gui_title: str = ""):
        super().__init__(uo_id)  # logger
        # simulation time vector
        assert t[0] == 0, "Time vector must start with 0"
        self._t = t
        self._dt = t[-1] / (t.size - 1)  # time step
        # id and title
        self.uo_id: str = uo_id
        """Unique identifier of the instance"""
        self.gui_title: str = gui_title
        """Readable title for GUI"""

        # adjustable parameter list
        self.adj_par_list = []
        """list of :class:`bio_rtd.adj_par.AdjustableParameter`: List
        of adjustable parameters exposed to the GUI."""
        # hide unit operation from plots
        self.gui_hidden: bool = False
        """Hide the of the unit operation (default False)."""

        # start-up phase (optional initial delay)
        self.discard_inlet_until_t: float = -1
        """Discard inlet until given time."""
        self.discard_inlet_until_min_c: _np.ndarray = _np.array([])
        """Discard inlet until given concentration is reached."""
        self.discard_inlet_until_min_c_rel: _np.ndarray = _np.array([])
        """Discard inlet until given concentration relative to is reached.
        
        Specified concentration is relative to the max concentration.
        
        """
        self.discard_inlet_n_cycles: int = -1
        """Discard first n cycles of the periodic inlet flow rate profile."""
        # shout-down phase (optional early stop)
        self.discard_outlet_until_t: float = -1
        """Discard outlet until given time."""
        self.discard_outlet_until_min_c: _np.ndarray = _np.array([])
        """Discard outlet until given concentration is reached."""
        self.discard_outlet_until_min_c_rel: _np.ndarray = _np.array([])
        """Discard outlet until given concentration relative to is reached.
        
        Specified concentration is relative to the max concentration.
        
        """
        self.discard_outlet_n_cycles: int = -1
        """Discard first n cycles of the periodic outlet flow rate profile."""

        # placeholders, populated during simulation
        self._c: _np.ndarray = _np.array([])  # concentration profiles
        self._f: _np.array = _np.array([])  # flow rate profile
        self._n_species: int = 0  # number of species

    def _assert_valid_species_list(self, species: _typing.Sequence[int]):
        """Species indexes start with 0.

        List must be ordered in ascending order (to prevent bugs).
        List must have unique values (again, to prevent bugs).

        """
        if len(species) == 0:
            self.log.w("Species list is empty")
        else:
            assert max(species) < self._n_species, \
                "Index of species should be less than number of species"
            assert min(species) >= 0, \
                "Index of species should not be negative"
            assert len(set(species)) == len(species), \
                "Vector with species should not have duplicate values"
            assert list(set(species)) == species, \
                "Values in vector with species must be ascending"

    def _is_flow_box_shaped(self) -> bool:
        """Constant profile with optional leading or trailing zeros."""
        assert _np.all(self._f >= 0), "Flow rate is negative!!!"
        if _np.all(self._f == 0):
            self.log.w("Flow rate is 0!")
            return False
        max_flow_start, max_flow_end = \
            _utils.vectors.true_start_and_end(self._f == self._f.max())
        if _np.any(self._f[:max_flow_start] > 0):
            return False
        elif _np.any(self._f[max_flow_end:] > 0):
            return False
        elif _np.any(self._f[max_flow_start:max_flow_end] != self._f.max()):
            return False
        else:
            return True

    def _i_flow_on(self) -> _typing.Sequence[int]:
        """Detect when the flow rate switches from off to on.

        In case of periodic flow rate, the function returns all
        switching events.

        Returns
        -------
        i_interval_start
            Indexes of time points at which the flow rate turns on.
            Each index corresponds to a leading non-zero value.

        """
        if _np.all(self._f == 0):
            self.log.w("Flow rate is 0!")
            return []
        assert _np.all(self._f[self._f != 0] == self._f.max()), \
            "flow rate must have a constant 'on' value"
        return list(_np.argwhere(_np.diff(self._f, prepend=0) > 0).flatten())

    def _assert_periodic_flow(self) -> _typing.Tuple[_typing.Sequence[int],
                                                     float,
                                                     float]:
        """Assert and provides info about periodic flow rate.

        Only last period is allowed to be shorter than others.

        Returns
        -------
        i_flow_start_list
            Indexes of time-points at which the flow rate gets turned on
        i_flow_on_average
            Number of time-points of flow 'on' interval.
        t_cycle_duration
            Duration of average cycle ('on' + 'off' interval)

        """
        # Get all flow on positions.
        i_flow_start_list = self._i_flow_on()
        assert len(i_flow_start_list) > 0, "Flow is 0"
        assert len(i_flow_start_list) > 1, \
            "Periodic flow must have at least 2 cycles"

        # Get average cycle duration.
        t_cycle_duration = _np.mean(_np.diff(i_flow_start_list)) * self._dt
        assert _np.all(_np.abs(
            _np.diff(i_flow_start_list) - t_cycle_duration / self._dt
        ) <= 1), "Periodic flow rate must have fixed cycle duration"

        # Assert periods.
        i_flow_on_first = \
            _utils.vectors.true_start(self._f[i_flow_start_list[0]:] == 0)
        i_flow_on_total = 0.0
        n_cycles = 0
        on_mask = _np.zeros_like(self._f, dtype=bool)
        for i in range(len(i_flow_start_list)):
            # Get next stop.
            if i + 1 == len(i_flow_start_list):
                i_cycle_flow_on_duration = _utils.vectors.true_end(
                    self._f[i_flow_start_list[i]:] > 0
                )
            else:
                i_cycle_flow_on_duration = _utils.vectors.true_start(
                    self._f[i_flow_start_list[i]:] == 0
                )
                i_flow_on_total += i_cycle_flow_on_duration
                n_cycles += 1
            i_flow_off = i_flow_start_list[i] + i_cycle_flow_on_duration
            on_mask[i_flow_start_list[i]:i_flow_off] = True
            if i + 1 == len(i_flow_start_list):
                # allow last cycle to be clipped
                assert i_cycle_flow_on_duration - i_flow_on_first <= 1
            else:
                # allow to be 1 time step off the first cycle
                assert abs(i_cycle_flow_on_duration - i_flow_on_first) <= 1
        # Flow can be either off or on at the constant value.
        assert _np.all(self._f[on_mask] == self._f.max())
        assert _np.all(self._f[~on_mask] == 0)

        i_flow_on_average = i_flow_on_total / n_cycles

        return i_flow_start_list, i_flow_on_average, t_cycle_duration

    def _estimate_steady_state_mean_f(self) -> (float, float):
        """Estimate mean flow rate in a period of periodic flow rate.

        Uses `self._assert_periodic_flow()` to get data about flow rate.

        Returns
        -------
        (float, float)
            f_mean
                Mean flow rate in one cycle.
            t_cycle_duration
                Duration of a cycle ('on' + 'off' interval).

        """
        _, i_flow_on_d, t_cycle_duration = self._assert_periodic_flow()
        if i_flow_on_d < 6:
            self.log.w(
                f"Flow is turned on for {i_flow_on_d} (< 6) time points")
        f_mean = self._f.max() * i_flow_on_d * self._dt / t_cycle_duration
        return f_mean, t_cycle_duration

    def _estimate_steady_state_mean_c(
            self,
            species: _typing.Optional[_typing.Sequence[int]] = None
    ) -> _np.ndarray:
        """Estimate mean concentration after start-up phase.

        In case of box shaped flow rate, the steady-state concentration
        is estimated as an average concentration between the first and
        the last crossing of 90 % of the max concentration of the sum
        across relevant species.

        In case of periodic flow rate, the steady-state concentration
        is estimated as an average concentration during the cycle with
        the largest average concentration.

        If possible, define the model so that this function is not
        needed; estimates can in general lead to unintuitive results.

        Parameters
        ----------
        species
            List of indexes of relevant species. (indexes start with 0).
            If not specified, all species are selected.

        Returns
        -------
        c_mean_ss: ndarray
            Estimated steady-state mean concentration.

        """
        if species is None:
            species = range(self._n_species)
        else:
            self._assert_valid_species_list(species)

        assert self._c.size > 0

        if _np.all(self._c[species] == 0):
            self.log.w("Concentration is zero for all relevant components")
            load_c_ss = _np.zeros([len(species), 1])
            self.log.i_data(self._log_tree, 'load_c_ss', load_c_ss)
            return load_c_ss

        if self._is_flow_box_shaped():
            c = self._c[species][:, self._f > 0]
            i_st, i_end = _utils.vectors.true_start_and_end(
                c.sum(0) >= c.sum(0).max() * 0.9
            )
            self.log.w(f"Steady-state concentration is being estimated"
                       f" as an average concentration withing 90 %"
                       f" between the first and the last crossing of 90 %"
                       f" of the max concentration for sum across"
                       f" species: {species}")
            load_c_ss = _np.mean(c[:, i_st:i_end], 1)[:, _np.newaxis]
            self.log.i_data(self._log_tree, 'load_c_ss', load_c_ss)
            return load_c_ss
        else:
            # Get info about periodic flow rate.
            i_flow_on_start_list, i_flow_on_duration, t_cycle_duration = \
                self._assert_periodic_flow()
            self.log.w(f"Steady-state concentration is being estimated"
                       f" as an average concentration in the cycle"
                       f" with the largest average concentration"
                       f" summed across species: {species}")
            m_max = 0
            c_max = _np.zeros([len(species), 1])
            for i, i_st in enumerate(i_flow_on_start_list):
                i_flow_on_duration_int = int(round(i_flow_on_duration))
                if i_st + i_flow_on_duration_int >= self._t.size:
                    continue
                _c_max = self._c[species, i_st:i_st + i_flow_on_duration_int] \
                    .mean(1)
                if _c_max.sum() > m_max:
                    c_max = _c_max
                    m_max = _c_max.sum()
            # Add dimension: (n_species,) -> (n_species, 1)
            load_c_ss = c_max[:, _np.newaxis]
            self.log.i_data(self._log_tree, 'load_c_ss', load_c_ss)
            return load_c_ss

    def _ensure_single_non_negative_parameter(self, log_level_multiple: int,
                                              log_level_none: int, **kwargs):
        """Check if one or multiple parameters are non-negative.

        Parameters
        ----------
        log_level_multiple
            Log level at which the function reports to `RtdLogger` in
            case of multiple non-negative parameters.
        log_level_none
            Log level at which the function reports to `RtdLogger` in
            case of no non-negative parameters.

        """
        non_null_keys = [key for key in kwargs.keys() if kwargs.get(key) >= 0]
        if len(non_null_keys) > 1:
            self.log.log(
                log_level_multiple,
                f"Only one of the parameters: {non_null_keys}"
                f" should be defined!")
        elif len(non_null_keys) == 0:
            self.log.log(
                log_level_none,
                f"One of the parameters: {kwargs.keys()} should be defined!")

    def _cut_start_of_c_and_f(self, outlet: bool = False):
        """Trim beginning of flow rate and concentration profiles

        Parameters
        ----------
        outlet
            Current profiles have already been processed by this
            `UnitOperation`.

        """
        if self._f.max() == 0:
            self.log.w("Flow rate is 0!")
            self._f *= 0
            self._c *= 0
            return

        if outlet:
            discard_until_t = self.discard_outlet_until_t
            discard_n_cycles = self.discard_outlet_n_cycles
            discard_until_min_c = self.discard_outlet_until_min_c
            discard_until_min_c_rel = self.discard_outlet_until_min_c_rel
        else:
            discard_until_t = self.discard_inlet_until_t
            discard_n_cycles = self.discard_inlet_n_cycles
            discard_until_min_c = self.discard_inlet_until_min_c
            discard_until_min_c_rel = self.discard_inlet_until_min_c_rel

        def get_cycle_start(_i, _i_start_list, _i_duration):
            for _i_start in _i_start_list:
                if _i <= _i_start + _i_duration:
                    return _i_start

        i_init_cut = -1
        _aux_has_min_c = discard_until_min_c.size > 0 \
            and discard_until_min_c.max() > 0
        _aux_has_min_c_rel = discard_until_min_c_rel.size > 0 \
            and discard_until_min_c_rel.max() > 0
        if _aux_has_min_c:
            assert discard_until_min_c.size == self._n_species
            discard_until_min_c = \
                discard_until_min_c.reshape(self._n_species, 1)
        if _aux_has_min_c_rel:
            assert discard_until_min_c_rel.size == self._n_species
            discard_until_min_c_rel = \
                discard_until_min_c_rel.reshape(self._n_species, 1)
        self._ensure_single_non_negative_parameter(
            self.log.ERROR, 0,
            discard_until_t=discard_until_t,
            discard_until_min_c=1 if _aux_has_min_c else -1,
            discard_until_min_c_rel=1 if _aux_has_min_c_rel else -1,
            discard_n_cycles=discard_n_cycles,
        )
        if discard_until_t > 0:
            if self._is_flow_box_shaped():
                i_init_cut = int(discard_until_t / self._dt)
            else:
                i_start_list, i_duration, _ = self._assert_periodic_flow()
                i_init_cut = get_cycle_start(int(discard_until_t / self._dt),
                                             i_start_list, i_duration)
        elif _aux_has_min_c or _aux_has_min_c_rel:
            if _aux_has_min_c:
                lim = _np.all(self._c >= discard_until_min_c, 0) \
                    * (self._f > 0)
            else:
                c_lim = discard_until_min_c_rel \
                    * self._c.max(1)[:, _np.newaxis]
                lim = _np.all(self._c >= c_lim, 0) * (self._f > 0)
            if _np.any(lim):
                if self._is_flow_box_shaped():
                    i_init_cut = _utils.vectors.true_start(lim)
                else:
                    i_start_list, i_duration, _ = self._assert_periodic_flow()
                    i_init_cut = get_cycle_start(
                        _utils.vectors.true_start(lim),
                        i_start_list, i_duration)
            else:
                if _aux_has_min_c:
                    c = discard_until_min_c
                else:
                    c = discard_until_min_c_rel
                self.log.w(f"Threshold concentration profile"
                           f" {c} was never reached!")
                self._f[:] = 0
                self._c[:] = 0
                return
        elif discard_n_cycles > 0:
            i_flow_on_start_list, _, _ = self._assert_periodic_flow()
            if len(i_flow_on_start_list) > discard_n_cycles:
                i_init_cut = i_flow_on_start_list[discard_n_cycles]
            else:
                self.log.w(f"All the cycles {len(i_flow_on_start_list)}"
                           f" are removed!")
                self._f[:] = 0
                self._c[:] = 0
                return

        if i_init_cut > 0:
            self._f[:i_init_cut] = 0
            self._c[:, :i_init_cut] = 0

    def evaluate(self, f_in: _np.array, c_in: _np.ndarray
                 ) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Evaluate the propagation throughout the unit operation.

        Parameters
        ----------
        c_in
            Inlet concentration profile with shape
            (n_species, n_time_steps).
        f_in
            Inlet flow rate profile with shape (n_time_steps,).

        Returns
        -------
        f_out
            Outlet flow rate profile.
        c_out
            Outlet concentration profile.

        """
        # Make a copy of inlet profiles, so we don't modify the originals.
        self._f = f_in.copy()
        self._c = c_in.copy()
        # Clear log.
        self._log_tree.clear()

        # Make sure concentration is not negative.
        if _np.any(self._c < 0):
            raise AssertionError(
                f"Negative concentration at unit operation: {self.uo_id}")
        # Make sure flow is not negative.
        if _np.any(self._f < 0):
            raise AssertionError(
                f"Negative flow at unit operation: {self.uo_id}")
        # Make sure flow is still there.
        if self._f.max() == 0:
            self.log.w(f"No inlet flow at unit operation: {self.uo_id}")
            return self._f * 0, self._c * 0

        # Calculate number of tracking species.
        self._n_species = self._c.shape[0]
        # Cut off start if required.
        self._cut_start_of_c_and_f(outlet=False)
        # (Re-)calculate the outlet profiles.
        self._calculate()
        # Cutoff start of outlet if needed.
        self._cut_start_of_c_and_f(outlet=True)

        return self._f, self._c

    @_abstractmethod
    def _calculate(self):  # pragma: no cover
        """Re-calculates the c_out and f_out from c_in and f_in.

        Function should use and modify `self._f` and `self._c`.

        """
        pass

    def get_result(self) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Returns existing flow rate and concentration profiles.

        Returns
        -------
        f_out
            Outlet flow rate profile.
        c_out
            Outlet concentration profile.

        """
        return self._f, self._c


class ParameterSetList(_ABC):
    """Abstract class for asserting keys in key-value pairs.

    Key-value pairs passed to `assert_and_get_provided_kv_pairs` should
    contain all keys from (at least) one of the key groups in
    `POSSIBLE_KEY_GROUPS`. The method returns key-value pars with keys
    from that group and all passed keys that can be also found in
    `OPTIONAL_KEYS`.

    Examples
    --------
    >>> class DummyClass(ParameterSetList):
    ...    POSSIBLE_KEY_GROUPS = [['par_1'], ['par_2a', 'par_2b']]
    ...    OPTIONAL_KEYS = ['key_plus_1', 'key_plus_2']
    >>>
    >>> dc = DummyClass()
    >>> dc.assert_and_get_provided_kv_pairs(par_1=1, par_2a=2)
    {'par_1': 1}
    >>> dc.assert_and_get_provided_kv_pairs(par_2a=1, par_2b=2,
    ...                                     key_plus_1=3, key_plus_9=2)
    {'par_2a': 1, 'par_2b': 2, 'key_plus_1': 3}
    >>> dc.assert_and_get_provided_kv_pairs(
    ...     key_plus_1=1) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    KeyError: "Keys ... do not contain any of the required groups: ...

    """

    # noinspection PyPep8Naming
    @property
    @_abstractmethod
    def POSSIBLE_KEY_GROUPS(self) \
            -> _typing.Sequence[_typing.Sequence[str]]:  # pragma: no cover
        """Possible key combinations.

        Examples
        --------
        POSSIBLE_KEY_GROUPS = [['v_void'], ['f', 'rt_mean']]

        """
        raise NotImplementedError

    # noinspection PyPep8Naming
    @property
    @_abstractmethod
    def OPTIONAL_KEYS(self) -> _typing.Sequence[str]:  # pragma: no cover
        """Optional additional keys.

        Examples
        --------
        OPTIONAL_KEYS = ['skew', 't_delay']

        """
        raise NotImplementedError

    def assert_and_get_provided_kv_pairs(self, **kwargs) -> dict:
        """
        Parameters
        ----------
        **kwargs
            Inputs to `calc_pdf(**kwargs)` function

        Returns
        -------
        dict
            Filtered `**kwargs` so the keys contain first possible key
            group in :attr:`POSSIBLE_KEY_GROUPS` and any number of
            optional keys from :attr:`OPTIONAL_KEYS`.

        Raises
        ------
        ValueError
            If `**kwargs` do not contain keys from any of the groups
            in :attr:`POSSIBLE_KEY_GROUPS`.

        """
        for group in self.POSSIBLE_KEY_GROUPS:
            if any([key not in kwargs.keys() for key in group]):
                continue
            else:
                # Get keys from groups.
                d = {key: kwargs.get(key) for key in group}
                # Get optional keys.
                d_extra = {key: kwargs.get(key)
                           for key in self.OPTIONAL_KEYS
                           if key in kwargs.keys()}
                # Combine and return.
                return {**d, **d_extra}

        raise KeyError(f"Keys {list(kwargs.keys())} do not contain any of"
                       f" the required groups: {self.POSSIBLE_KEY_GROUPS}")


class PDF(ParameterSetList, DefaultLoggerLogic, _ABC):
    """Abstract class for defining probability distribution functions.

    Parameters
    ----------
    t
        Simulation time vector.
    pdf_id
        Unique identifier of the `PDF` instance.

    """

    def __init__(self, t: _np.ndarray, pdf_id: str = ""):
        super().__init__(pdf_id)

        assert t[0] == 0
        assert t[-1] > 0

        self._dt = t[-1] / (t.size - 1)
        self._t_steps_max = t.size

        # apply cutoff
        self.trim_and_normalize = True
        """Trim edges of the pdf and normalize it afterwards.
        
        Default = True.
        
        Relative threshold value is specified by
        :attr:`cutoff_relative_to_max`.
        
        Normalization is performed after the trimming.
        The area of pd == 1.
        
        """
        self.cutoff_relative_to_max = 0.0001
        """Cutoff as a share of max value of the pdf (default 0.0001).
        
        It is defined to avoid very long tails of the distribution.
        
        Cutoff is enabled if :attr:`trim_and_normalize` == True.
        
        """

        # placeholder for the result of the pdf calculation
        self._pdf: _np.array = _np.array([])

    def _apply_cutoff_and_normalize(self):
        # Get cutoff positions.
        i_start, i_end = _utils.vectors.true_start_and_end(
            self._pdf >= self.cutoff_relative_to_max * self._pdf.max()
        )
        # 0 at front.
        self._pdf[:i_start] = 0
        # Cut at end.
        self._pdf = self._pdf[:i_end]
        # Normalize.
        self._pdf = self._pdf / self._pdf[:i_end].sum() / self._dt

    def update_pdf(self, **kwargs):
        """Re-calculate PDF based on specified parameters.

        The calculated probability distribution can be obtained by
        :func:`get_p()`

        Parameters
        ----------
        **kwargs
            Should contain keys from one of the group in
            :attr:`POSSIBLE_KEY_GROUPS`.
            It may contain additional keys from :attr:`OPTIONAL_KEYS`.

        """
        kw = self.assert_and_get_provided_kv_pairs(**kwargs)
        self._pdf = self._calc_pdf(kw)
        if self.trim_and_normalize:
            self._apply_cutoff_and_normalize()

    @_abstractmethod
    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:  # pragma: no cover
        """Calculation of probability distribution.

        Evaluates pdf for a given set of parameters. The keys of the
        `kw_pars` include keys from one of the group in
        :attr:`POSSIBLE_KEY_GROUPS` and any optional subset of keys from
        :attr:`OPTIONAL_KEYS`.

        Parameters
        ----------
        kw_pars
            Key-value parameters.
            Keys contain keys from one of the groups specified in
            `self.POSSIBLE_KEY_GROUPS`.
            Keys may contain additional keys from `self.OPTIONAL_KEYS`.
        """
        raise NotImplementedError

    def get_p(self) -> _np.ndarray:
        """Get probability distribution.

        Returns
        -------
        p: ndarray
            Evaluated probability distribution function.

            Corresponding time axis starts with 0 and has a fixed step
            size (`_dt`).

            If :attr:`trim_and_normalize` == 1 then `sum(p * _dt) == 1`.

        """
        assert self._pdf.size > 0, f"PDF is empty. Make sure `update_pdf`" \
                                   f" was called before `get_p`."
        if self._pdf.size <= 5:
            self.log.e("PDF should have a length of at least 5 time steps.")

        return self._pdf


class ChromatographyLoadBreakthrough(ParameterSetList,
                                     DefaultLoggerLogic, _ABC):
    """What parts of the load bind to the column.

    Parameters
    ----------
    dt
        Time step duration.
    bt_profile_id
        Unique identifier of the PDF instance. Used for logs.

    """

    def __init__(self, dt: float,
                 bt_profile_id: str = "ChromatographyLoadBreakthrough"):
        super().__init__(bt_profile_id)
        assert dt > 0
        self._dt = dt

    def update_btc_parameters(self, **kwargs):
        """Update binding dynamics for a given set of parameters.

        Parameters
        ----------
        **kwargs
            Should contain keys from one of the group in
            :attr:`POSSIBLE_KEY_GROUPS`.
            It may contain additional keys from :attr:`OPTIONAL_KEYS`.

        """
        kw = self.assert_and_get_provided_kv_pairs(**kwargs)
        self._update_btc_parameters(kw)

    def calc_c_bound(self,
                     f_load: _np.ndarray,
                     c_load: _np.ndarray) -> _np.ndarray:
        """Calculates what parts of load bind to the column.

        The default implementation calculates cumulative mass of the
        load material and passes it to :func:`_update_btc_parameters`
        abstract method for evaluation on what shares of the load
        bind to the column. Those shares are then multiplied by `c_load`
        in order to obtain resulting `c_bound`.

        This method is meant to be overridden, if needed.

        Parameters
        ---------
        f_load
            Load flow rate profile.
        c_load
            Load concentration profile. Concentration profile should
            include only species which bind to the column.

        Returns
        -------
        c_bound: ndarray
            Parts of the load that bind to the column during the
            load step.

            `c_bound` has the same shape as `c_load`.

        """
        # All species are summed together.
        m_cum_sum = _np.cumsum(
            (c_load * f_load[_np.newaxis, :]).sum(0)) * self._dt
        if m_cum_sum[-1] == 0:  # empty concentration vector
            return _np.zeros_like(c_load)
        unbound_to_load_ratio = \
            self._calc_unbound_to_load_ratio(m_cum_sum.flatten())
        c_bound = c_load * (1 - unbound_to_load_ratio[_np.newaxis, :])
        return c_bound

    @_abstractmethod
    def _update_btc_parameters(self,
                               kw_pars: dict):  # pragma: no cover
        """Update binding dynamics for a given key-value set.

        Parameters
        ----------
        kw_pars: dict
            Key-value parameters.
            Keys contain keys from one of the groups specified in
            `self.POSSIBLE_KEY_GROUPS` and may contain additional keys
            from `self.OPTIONAL_KEYS`.
        """

        pass

    @_abstractmethod
    def _calc_unbound_to_load_ratio(self, loaded_material: _np.ndarray
                                    ) -> _np.ndarray:  # pragma: no cover
        """Calculates what share of the load binds to the column.

        Typical implementation would be just direct evaluation of the
        breakthrough curve. See the `ConstantPatternSolution` subclass.

        Parameters
        ----------
        loaded_material
            Cumulative sum of the amount of loaded material.

        Returns
        -------
        bound_to_unbound_ratio
            Ratio between the amount of captured material and the
            amount of load material. `bound_to_unbound_ratio` has
            the same size as `loaded_material`.

        """
        pass

    @_abstractmethod
    def get_total_bc(self) -> float:  # pragma: no cover
        """Total binding capacity.

        Meant e.g. for determining column utilization.

        """
        pass


class RtdModel(DefaultLoggerLogic, _ABC):
    """Combines `Inlet` and a train of `UnitOperation`-s into a model.

    The logger assigned to the instance of RtdModel is passed on to
    :class:`bio_rtd.core.Inlet` and
    :class:`bio_rtd.core.UnitOperation` instances.

    Parameters
    ----------
    inlet
        Inlet profile.
    dsp_uo_chain
        Sequence of unit operations. The sequence needs to be in order.
    logger
        Logger for sending status messages and storing intermediate
        data.
    title
        Title of the model.
    desc
        Description of the model.

    See Also
    --------
    :class:`bio_rtd.core.Inlet`
    :class:`bio_rtd.core.UnitOperation`
    :class:`bio_rtd.logger.RtdLogger`

    """

    def __init__(self,
                 inlet: Inlet,
                 dsp_uo_chain: _typing.Sequence[UnitOperation],
                 logger: _typing.Optional[_logger.RtdLogger] = None,
                 title: str = "RtdModel",
                 desc: str = ""):
        super().__init__(title)
        # Ensure unique `uo_id`s.
        ids = [uo.uo_id for uo in dsp_uo_chain]
        assert len(ids) == len(set(ids)), \
            "Each unit operation must have a unique id (`uo_id`)"
        # Bind parameters.
        self.inlet = inlet
        """:class:`bio_rtd.core.Inlet`: Inlet for `self.dsp_uo_chain`"""
        self.dsp_uo_chain = dsp_uo_chain
        """sequence of :class:`bio_rtd.core.UnitOperation`: Chain of
        unit operations in the process.
        
        Unit operations need to be in proper order.
        
        The logger in unit operations is overridden by the logger from
        this model.
        
        """
        self.title = title
        """Human readable title (mostly for plots)"""
        self.desc = desc
        """Human readable description (also mostly for plots)"""
        # Init data log tree with empty dict for each unit operation.
        self._log_tree = _OrderedDict(
            {inlet.uo_id: {}, **{uo.uo_id: {} for uo in dsp_uo_chain}})
        self.log = logger if logger is not None else self.log

    def get_dsp_uo(self, uo_id: str) -> UnitOperation:
        """Get reference to a `UnitOperation` with specified `uo_id`."""
        for uo in self.dsp_uo_chain:
            if uo.uo_id == uo_id:
                return uo
        raise KeyError(f"Unit operation `{uo_id}` not found.\n"
                       f"Available: {[uo.uo_id for uo in self.dsp_uo_chain]}")

    def recalculate(self,
                    start_at: int = -1,
                    on_update_callback: _typing.Optional[
                        _typing.Callable[[int], None]
                    ] = None):
        """Recalculate process fluid propagation.

        Parameters
        ----------
        start_at
            Index of first unit operation for re-evaluation.

            Indexing starts at 0 (-1 for the inlet). Default = -1.
        on_update_callback
            Optional callback function which receives an integer.

            The integer corresponds to the index of re-evaluated unit
            operation, starting with 0 (-1 for inlet).

            This can serve as a trigger for updating UI or any other
            post-processing after re-evaluation of unit operations.

        """
        # Evaluate inlet profile.
        if start_at == -1:
            self.inlet.refresh()
            self._notify_updated(-1, on_update_callback)
            start_at = 0
        # Get outlet of previous unit operation.
        if start_at == 0:
            f, c = self.inlet.get_result()
        else:
            f, c = self.dsp_uo_chain[start_at - 1].get_result()
        # Evaluate subsequent unit operations.
        for i in range(start_at, len(self.dsp_uo_chain)):
            f, c = self.dsp_uo_chain[i].evaluate(f, c)
            self._notify_updated(i, on_update_callback)

    def _notify_updated(self, uo_i: int, on_update_callback):
        # Show INFO log.
        if uo_i == -1:
            uo = self.inlet
            self.log.i("Inlet profile updated")
        else:
            uo = self.dsp_uo_chain[uo_i]
            self.log.i(f"Unit operation `{uo.uo_id}` updated")
        # Store profiles in DEBUG data log.
        if self.log.log_level <= self.log.DEBUG:
            f, c = uo.get_result()
            self.log.d_data(self._log_tree[uo.uo_id], "f", f)
            self.log.d_data(self._log_tree[uo.uo_id], "c", c)
        # Call callback function if specified.
        if on_update_callback:
            on_update_callback(uo_i)

    @DefaultLoggerLogic.log.setter
    def log(self, logger: _logger.RtdLogger):
        self._logger = logger
        self._logger.set_data_tree(self._log_entity_id, self._log_tree)
        # Pass logger to inlet and unit operations.
        self.inlet.log = logger
        for uo in self.dsp_uo_chain:
            uo.log = logger

    def set_logger_from_parent(self, parent_id: str,
                               logger: _logger.RtdLogger):  # pragma: no cover
        # This dummy definition is here just to maintain the right order
        # of methods in documentation.
        super().set_logger_from_parent(parent_id, logger)


class UserInterface(_ABC):
    """Wrapper around RtdModel suitable for building GUI on top of it.

    Parameters
    ----------
    rtd_model
        Residence time distribution model.

    See Also
    --------
    :class:`bio_rtd.core.RtdModel`

    """

    def __init__(self, rtd_model: RtdModel):
        self.rtd_model = rtd_model

        # default values - update them after init
        self.start_at: int = -1
        """Index of first unit operation for re-evaluation.
                
        Indexing starts at 0 (-1 for the inlet). Default = -1.
        
        """
        self.species_label: _typing.Sequence[str] = \
            self.rtd_model.inlet.species_list
        """Labels of the species in concentration array.
        
        Initially inherited from :class:`bio_rtd.core.Inlet` instance.
        
        """
        self.x_label: str = 't'
        """Label of x axis (time). Default = 't'"""
        self.y_label_c: str = 'c'
        """Label of y axis (concentration). Default = 'c'"""
        self.y_label_f: str = 'f'
        """Label of y axis (flow rate). Default = 'f'"""

    def recalculate(self, forced=False):
        """Re-evaluates the model from the `start_at` index onwards.

        Parameters
        ----------
        forced
            If true, the entire model (inlet + unit operations) is
            re-evaluated. The same can be achieved by setting
            :attr:`start_at` to -1.

        """
        start_at = -1 if forced else self.start_at

        def callback_fun(i):
            if i == -1:
                f, c = self.rtd_model.inlet.get_result()
            else:
                f, c = self.rtd_model.dsp_uo_chain[i].get_result()
            self._update_ui_for_uo(i, f, c)

        # Re-calculate the model.
        self.rtd_model.recalculate(start_at, callback_fun)

    @_abstractmethod
    def build_ui(self):  # pragma: no cover
        """Build the UI from scratch."""
        raise NotImplementedError

    @_abstractmethod
    def _update_ui_for_uo(self, uo_i, f, c):  # pragma: no cover
        """Update the UI for specific unit operation.

        Parameters
        ----------
        uo_i: int
            Index of unit operation (-1 for inlet profile).
            Indexes for DSP unit operation train start with 0.
        c: _np.ndarray
            Outlet concentration profile.
        f: _np.ndarray
            Outlet flow profile.
        """
        raise NotImplementedError
