"""Module with abstract classes."""

__all__ = ['Inlet', 'UnitOperation',
           'RtdModel', 'UserInterface',
           'PDF', 'ChromatographyLoadBreakthrough',
           'ParameterSetList']
__version__ = '0.7'
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

    The class holds a reference to a `RtdLogger` logger instance
    and plants a data tree in the logger.

    Parameters
    ----------
    logger_parent_id
        Custom unique id that belongs to the instance of the class.
        It is used to plant a data tree in the `RtdLogger`.

    Examples
    --------
    >>> logger_parent_id = "parent_unit_operation"
    >>> l = DefaultLoggerLogic(logger_parent_id)
    >>> isinstance(l.log, _logger.DefaultLogger)
    True
    >>> l.log.e("Error Description")  # log error
    Traceback (most recent call last):
    RuntimeError: Error Description
    >>> l.log.w("Warning Description")  # log waring
    Warning Description
    >>> l.log.i("Info")  # log info
    >>> l.log.log_data = True
    >>> l.log.log_level = _logger.RtdLogger.DEBUG
    >>> l.log.i_data(l._log_tree, "a", 3)  # store value in logger
    >>> l.log.d_data(l._log_tree, "b", 7)  # store at DEBUG level
    >>> l.log.get_data_tree(logger_parent_id)["b"]
    7
    >>> l.log = _logger.StrictLogger()
    >>> l.log.w("Warning Info")
    Traceback (most recent call last):
    RuntimeError: Warning Info

    Notes
    -----
    See the documentation of the `RtdLogger`.

    """

    def __init__(self, logger_parent_id: str):
        self._instance_id = logger_parent_id
        self._logger: _typing.Union[_logger.RtdLogger, None] = None
        self._log_tree = dict()  # place to store logged data

    @property
    def log(self) -> _logger.RtdLogger:
        """Logger.

        If logger is not set, then a `DefaultLogger` is instantiated.
        Setter also plants a data tree into passed logger.

        """
        if self._logger is None:
            self.log = _logger.DefaultLogger()  # init default logger
        return self._logger

    @log.setter
    def log(self, logger: _logger.RtdLogger):
        self._logger = logger
        self._logger.set_data_tree(self._instance_id, self._log_tree)

    def set_logger_from_parent(self, parent_id: str, logger: _logger.RtdLogger):
        """Inherit logger from parent.

        Parameters
        ----------
        parent_id
        logger
        """
        self._logger = logger
        self._logger.set_data_tree(f"{parent_id}/{self._instance_id}",
                                   self._log_tree)


class Inlet(DefaultLoggerLogic, _ABC):
    """Generates starting flow rate and concentration profiles.

    Parameters
    ----------
    t
        Simulation time vector

        Starts with 0 and has a constant time step.
    species_list
        List with names of simulating process fluid species.
    inlet_id
        Unique identifier of an instance. It is stored in :attr:`uo_id`.
    gui_title
        Readable title of an instance.

    Attributes
    ----------
    species_list : list of str
        List with names of simulating process fluid species.
    uo_id : str
        Unique identifier of the :class:`Inlet` instance.
    gui_title : str
        Readable title of the :class:`Inlet` instance.
    adj_par_list : list of :class:`bio_rtd.adj_par.AdjustableParameter`
        List of adjustable parameters exposed to the GUI.

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
        self.species_list = species_list
        self._n_species = len(self.species_list)
        
        # Strings
        self.uo_id: str = inlet_id
        self.gui_title = gui_title
        
        # Placeholders
        self.place_inlet_before_uo_id: _typing.Optional[str] = None
        self.adj_par_list: _typing.Sequence[_adj_par.AdjustableParameter] = ()
        
        # Outputs
        self._f_out = _np.zeros_like(t)
        self._c_out = _np.zeros([self._n_species, t.size])

    @_abstractmethod
    def _refresh(self):  # pragma: no cover
        """Re-calculates `self._f_out` and `self._c_out`."""
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

    def get_t(self) -> _np.ndarray:
        """Get simulation time vector."""
        return self._t

    def get_n_species(self) -> int:
        """Get number of process fluid species."""
        return self._n_species


class UnitOperation(DefaultLoggerLogic, _ABC):
    """Processes flow rate and concentration profiles.

    Parameters
    ----------
    t
        Global simulation time vector.
        It must starts with 0 and have a constant time step.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI.

    Attributes
    ----------
    adj_par_list
        List of adjustable parameters exposed to the GUI.
    gui_hidden
        Hide the of the unit operation (default False).
    discard_inlet_until_t
        Discard inlet until given time.
    discard_inlet_until_min_c
        Discard inlet until given concentration is reached.
    discard_inlet_until_min_c_rel
        Discard inlet until given concentration relative to is reached.
        Specified concentration is relative to the max concentration.
    discard_inlet_n_cycles
        Discard first n cycles of the periodic inlet flow rate profile.
    discard_outlet_until_t
        Discard outlet until given time.
    discard_outlet_until_min_c
        Discard outlet until given concentration is reached.
    discard_outlet_until_min_c_rel
        Discard outlet until given concentration relative to is reached.
        Specified concentration is relative to the max concentration.
    discard_outlet_n_cycles
        Discard first n cycles of the periodic outlet flow rate profile.

    """

    def __init__(self, t: _np.ndarray, uo_id: str, gui_title: str = ""):
        super().__init__(uo_id)  # logger
        # simulation time vector
        assert t[0] == 0, "Time vector must start with 0"
        self._t = t
        self._dt = t[-1] / (t.size - 1)  # time step
        # id and title
        self.uo_id = uo_id
        self.gui_title = gui_title

        # hide unit operation from plots
        self.gui_hidden = False
        # adjustable parameter list
        self.adj_par_list: _typing.Sequence[_adj_par.AdjustableParameter] = ()
        """Settings"""

        # start-up phase (optional initial delay)
        self.discard_inlet_until_t = -1
        self.discard_inlet_until_min_c: _np.ndarray = _np.array([])
        self.discard_inlet_until_min_c_rel: _np.ndarray = _np.array([])
        self.discard_inlet_n_cycles = -1
        # shout-down phase (optional early stop)
        self.discard_outlet_until_t = -1
        self.discard_outlet_until_min_c: _np.ndarray = _np.array([])
        self.discard_outlet_until_min_c_rel: _np.ndarray = _np.array([])
        self.discard_outlet_n_cycles = -1

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
        i_interval_start : Sequence[int]
            Indexes of time points at which the flow rate turns on.
            Each index corresponds to a leading non-zero value.

        """
        if _np.all(self._f == 0):
            self.log.w("Flow rate is 0!")
            return []
        assert _np.all(self._f[self._f != 0] == self._f.max()), \
            "flow rate must have a constant 'on' value"
        return list(_np.argwhere(_np.diff(self._f, prepend=0) > 0).flatten())

    def _assert_periodic_flow(self) -> (_typing.Sequence[int], int, float):
        """Assert and provides info about periodic flow rate.

        Only last period is allowed to be shorter than others.

        Returns
        -------
        i_flow_start_list : Sequence[int]
            Indexes of time-points at which the flow rate gets turned on
        i_flow_on_duration : int
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
        i_flow_on_duration = \
            _utils.vectors.true_start(self._f[i_flow_start_list[0]:] == 0)
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
            i_flow_off = i_flow_start_list[i] + i_cycle_flow_on_duration
            on_mask[i_flow_start_list[i]:i_flow_off] = True
            if i + 1 == len(i_flow_start_list):
                # allow last cycle to be clipped
                assert i_cycle_flow_on_duration - i_flow_on_duration <= 1
            else:
                # allow to be 1 time step off the first cycle
                assert abs(i_cycle_flow_on_duration - i_flow_on_duration) <= 1
        # Flow can be either off or on at the constant value.
        assert _np.all(self._f[on_mask] == self._f.max())
        assert _np.all(self._f[~on_mask] == 0)

        return i_flow_start_list, i_flow_on_duration, t_cycle_duration

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
        species : list of int
            List of indexes of relevant species. (indexes start with 0).
            If not specified, all species are selected.

        Returns
        -------
        (float, float)
            f_mean
                Mean flow rate in one cycle.
            t_cycle_duration
                Duration of a cycle ('on' + 'off' interval).

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
                if i_st + i_flow_on_duration >= self._t.size:
                    continue
                _c_max = self._c[species, i_st:i_st + i_flow_on_duration] \
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
        log_level_multiple : int
            Log level at which the function reports to `RtdLogger` in
            case of multiple non-negative parameters.
        log_level_none : int
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
        """Evaluate the propagation through the unit operation.

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
    `_possible_key_groups`. The method returns key-value pars with keys
    from that group and all passed keys that can be also found in
    `_optional_keys`.

    Examples
    --------
    >>> class DummyClass(ParameterSetList):
    ...    _possible_key_groups = [['par_1'], ['par_2a', 'par_2b']]
    ...    _optional_keys = ['key_plus_1', 'key_plus_2']
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

    @property
    @_abstractmethod
    def _possible_key_groups(self) \
            -> _typing.Sequence[_typing.Sequence[str]]:  # pragma: no cover
        """Possible key combinations.

        To override it in a new class, simply define an attribute, e.g.:
        _possible_key_groups = [['v_void'], ['f', 'rt_mean']]

        """
        raise NotImplementedError

    @property
    @_abstractmethod
    def _optional_keys(self) -> _typing.Sequence[str]:  # pragma: no cover
        """Optional additional keys.

        To override it in a new class, simply define an attribute, e.g.:
        _optional_keys = ['skew', 't_delay']

        """
        raise NotImplementedError

    def assert_and_get_provided_kv_pairs(self, **kwargs) -> dict:
        """
        Parameters
        ----------
        kwargs
            Inputs to `calc_pdf(**kwargs)` function

        Returns
        -------
        dict
            Filtered `kwargs` so the keys contain first possible key
            group in `_possible_key_groups` and any number of optional
            keys from `_optional_keys`.

        Raises
        ------
        ValueError
            If `**kwargs` do not contain keys from any of the groups
            in `_possible_key_groups`.

        """
        for group in self._possible_key_groups:
            if any([key not in kwargs.keys() for key in group]):
                continue
            else:
                # Get keys from groups.
                d = {key: kwargs.get(key) for key in group}
                # Get optional keys.
                d_extra = {key: kwargs.get(key)
                           for key in self._optional_keys
                           if key in kwargs.keys()}
                # Combine and return.
                return {**d, **d_extra}

        raise KeyError(f"Keys {list(kwargs.keys())} do not contain any of"
                       f" the required groups: {self._possible_key_groups}")


class PDF(ParameterSetList, DefaultLoggerLogic, _ABC):
    """Abstract class for defining probability distribution functions.

    Parameters
    ----------
    t
        Simulation time vector.
    pdf_id
        Unique identifier of the PDF instance.

    Attributes
    ----------
    trim_and_normalize
        Trim edges of the peak by the threshold at the relative signal
        specified by `cutoff_relative_to_max`. Default = True.
    cutoff_relative_to_max
        Cutoff as a share of max value of the pdf (default 0.0001).
        It is defined to avoid very long tails of the distribution.

    Methods
    -------
    get_p()
        Get calculated PDF.
    update_pdf(**kwargs)
        Re-calculate PDF based on specified parameters.

    Abstract Methods
    ----------------
    _calc_pdf(kw_pars: dict)
        Calculate new pdf for a given set of parameters. The keys of the
        `kw_pars` include keys from one of the group in
        `_possible_key_groups` and any optional subset of keys from
        `_optional_keys`.

    """

    def __init__(self, t: _np.ndarray, pdf_id: str = ""):
        super().__init__(pdf_id)

        assert t[0] == 0
        assert t[-1] > 0

        self._dt = t[-1] / (t.size - 1)
        self._t_steps_max = t.size

        # apply cutoff
        self.trim_and_normalize = True
        self.cutoff_relative_to_max = 0.0001

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

        Parameters
        ----------
        kwargs
            Should contain keys from one of the group in
            `self._possible_key_groups`.
            It may contain additional keys from `self._optional_keys`.

        """
        kw = self.assert_and_get_provided_kv_pairs(**kwargs)
        self._pdf = self._calc_pdf(kw)
        if self.trim_and_normalize:
            self._apply_cutoff_and_normalize()

    @_abstractmethod
    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:  # pragma: no cover
        """Calculation of probability distribution.

        Parameters
        ----------
        kw_pars: dict
            Key-value parameters.
            Keys contain keys from one of the groups specified in
            `self._possible_key_groups`.
            Keys may contain additional keys from `self._optional_keys`.
        """
        raise NotImplementedError

    def get_p(self) -> _np.ndarray:
        """Get probability distribution.

        Returns
        -------
        p: np.ndarray
            Evaluated probability distribution function.
            `sum(p * self._dt) == 1`
            Corresponding time axis starts with 0 and has a fixed step
            of self._dt.

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
    bt_profile_id
        Unique identifier of the PDF instance. Used for logs.

    Notes
    -----
    See docstring of `ParameterSetList` for information about key
    groups.

    """

    def __init__(self, dt: float,
                 bt_profile_id: str = "ChromatographyLoadBreakthrough"):
        super().__init__(bt_profile_id)
        assert dt > 0
        self._dt = dt

    def update_btc_parameters(self, **kwargs) -> None:
        """Update binding dynamics for a given set of parameters."""
        kw = self.assert_and_get_provided_kv_pairs(**kwargs)
        self._update_btc_parameters(kw)

    def calc_c_bound(self,
                     f_load: _np.ndarray,
                     c_load: _np.ndarray) -> _np.ndarray:
        """Calculates what parts of load bin to the column.

        Parameters
        ---------
        f_load
            Load flow rate profile.
        c_load
            Load concentration profile. Concentration profile should
            include only species which bind to the column.

        Returns
        -------
        c_bound
            Parts of the load that binds to the column during the
            load step. `c_bound` has the same shape as `c_load`.

        Notes
        -----
        This is default implementation. The user is welcome to override
        this function in a custom child class.

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
                               kw_pars: dict) -> None:  # pragma: no cover
        """Update binding dynamics for a given key-value set.

        Parameters
        ----------
        kw_pars: dict
            Key-value parameters.
            Keys contain keys from one of the groups specified in
            `self._possible_key_groups` and may contain additional keys
            from `self._optional_keys`.
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
        """Total binding capacity

        Useful for determining column utilization.

        """
        pass


class RtdModel(DefaultLoggerLogic, _ABC):
    """Combines inlet and a train of unit operations into a model.

    The logger assigned to the instance of RtdModel is propagated
    throughout `inlet` and unit operations in `dsp_uo_chain`.

    Parameters
    ----------
    inlet
        Inlet profile.
    dsp_uo_chain
        Sequence of unit operations. The sequence needs to be in order.
    logger
        Logger that the model uses for sending status messages and
        storing intermediate data.
    title
        Title of the model.
    desc
        Description of the model.

    Methods
    -------
    recalculate(start_at, on_update_callback)
        Recalculates the process fluid propagation, starting at
        `start_at` unit operation (-1 for inlet and entire process).
        Callback function can be specified. It receives the index of
        the just updated unit operation.

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
        self.dsp_uo_chain = dsp_uo_chain
        self.title = title
        self.desc = desc
        # Init data log tree with empty dict for each unit operation.
        self._log_tree = _OrderedDict(
            {inlet.uo_id: {}, **{uo.uo_id: {} for uo in dsp_uo_chain}})
        self.log = logger if logger is not None else self.log

    @DefaultLoggerLogic.log.setter
    def log(self, logger: _logger.RtdLogger):
        self._logger = logger
        self._logger.set_data_tree(self._instance_id, self._log_tree)
        # Pass logger to inlet and unit operations.
        self.inlet.log = logger
        for uo in self.dsp_uo_chain:
            uo.log = logger

    def get_dsp_uo(self, uo_id: str) -> UnitOperation:
        """Get reference to a `UnitOperation` with specified `uo_id`."""
        for uo in self.dsp_uo_chain:
            if uo.uo_id == uo_id:
                return uo
        raise KeyError(f"Unit operation `{uo_id}` not found.\n"
                       f"Available: {[uo.uo_id for uo in self.dsp_uo_chain]}")

    def recalculate(self,
                    start_at: int = 0,
                    on_update_callback: _typing.Optional[
                        _typing.Callable[[int], None]
                    ] = None):
        """Recalculate process fluid propagation.

        Parameters
        ----------
        start_at
            The index of first unit operation that needs to be
            re-evaluated. Default = 0.
            If -1, then the inlet profile is also re-evaluated.
        on_update_callback
            Optional callback function which receives an integer.

            The integer corresponds to the index of re-evaluated unit
            operation (-1 for inlet).
            This can serve as a trigger for updating UI after
            re-evaluation of individual unit operations.

        """
        # Evaluate inlet profile.
        if start_at == -1:
            self.inlet._refresh()
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


class UserInterface(_ABC):
    """Wrapper around RtdModel suitable for building GUI on top of it.

    Parameters
    ----------
    rtd_model : RtdModel
        RTD model.

    Attributes
    ----------
    species_label : list of str
        Labels of the species in concentration array.
    x_label
        Label of x axis (time). Default = 't'
    y_label_c
        Label of y axis (concentration). Default = 'c'
    y_label_f
        Label of y axis (flow rate). Default = 'f'
    start_at : int
        The index of unit operation (starting with 0) at which the
        re-evaluation starts. The value of -1 means that the inlet
        profile is also reevaluated.

    """

    def __init__(self, rtd_model: RtdModel):
        self.rtd_model = rtd_model

        # default values - update them after init
        self.start_at = -1
        self.species_label = self.rtd_model.inlet.species_list
        self.x_label = 't'
        self.y_label_c = 'c'
        self.y_label_f = 'f'

    def recalculate(self, forced=False):
        """Re-evaluates the model from the `start_at` index onwards.

        Parameters
        ----------
        forced
            If true, the entire model (inlet + unit operations) is
            re-evaluated. The same can be achieved by setting
            `self.start_at` to -1.

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
