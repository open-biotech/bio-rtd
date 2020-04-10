"""Semi continuous unit operations.

Unit operations that accept and provide
constant or box-shaped flow rate profile.

Box-shaped flow rate profile is a profile with constant value with
optional trailing zeros at front or at end.

"""

__all__ = ['Dilution', 'Concentration', 'BufferExchange',
           'FlowThrough', 'FlowThroughWithSwitching']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.core as _core
import bio_rtd.utils as _utils


class Dilution(_core.UnitOperation):
    """Process fluid stream dilution by constant ratio.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    dilution_ratio
        Dilution ratio. Must be >= 1.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "Dilution".


    Attributes
    ----------
    c_add_buffer
        Concentration of species in dilution buffer.

    """

    def __init__(self, t: _np.ndarray,
                 dilution_ratio: float,
                 uo_id: str,
                 gui_title: str = "Dilution"):
        super().__init__(t, uo_id, gui_title)
        assert dilution_ratio >= 1
        self.dilution_ratio: float = dilution_ratio
        """Dilution ratio. Must be >= 1.
        
        Dilution ratio of 1.2 means adding 20 % of dilution buffer.
        
        """
        self.c_add_buffer: _np.ndarray = _np.array([])
        """Concentration of species in dilution buffer.
        
        Default = empty array (= all components are 0).
        
        If defined, it must have a value for each process fluid specie.
        
        """

    def _calc_c_add_buffer(self):
        """Get concentration of species in dilution buffer."""
        assert self.c_add_buffer.size == 0 \
            or self.c_add_buffer.size == self._n_species

        if self.c_add_buffer.size == 0:
            self._c_add_buffer = _np.zeros([self._n_species, 1])
        else:
            self._c_add_buffer = \
                self.c_add_buffer.reshape(self._n_species, 1)

    def _calculate(self):
        assert self.dilution_ratio >= 1
        assert hasattr(self, "_c_add_buffer")
        if self.dilution_ratio == 1:
            self.log.w("Dilution ratio is set to 1.")
        # Apply dilution.
        self._f = self._f * self.dilution_ratio
        self._c = (self._c + (self.dilution_ratio - 1) * self._c_add_buffer) \
            / self.dilution_ratio


class Concentration(_core.UnitOperation):
    """Concentrate process fluid stream.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    flow_reduction
        Flow reduction (== volume reduction). Must be > 1.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "Concentration".


    Attributes
    ----------
    non_retained_species
        List of indexes of non-retained species. Indexing starts with 0.
    relative_losses
        Relative losses of retained species.

    """

    def __init__(self, t: _np.ndarray,
                 flow_reduction: float,  # f_out = f_in / flow_reduction
                 uo_id: str, gui_title: str = "Concentration"):
        super().__init__(t, uo_id, gui_title)
        assert flow_reduction > 1
        self.flow_reduction: float = flow_reduction
        """Flow reduction (== volume reduction). Must be > 1.
        
        `outlet flow rate` = `inlet flow rate` / `flow_reduction`.
        
        """
        self.non_retained_species: _typing.Sequence[int] = []
        """Indexes of non-retained species. Indexing starts with 0."""
        self.relative_losses: float = 0.
        """Relative losses of retained species."""

    # noinspection DuplicatedCode
    def _calculate(self):
        if len(self.non_retained_species) > 0:
            assert len(self.non_retained_species) < self._n_species, \
                f"All species cannot be `non_retained_species`."
            assert list(self.non_retained_species) \
                == list(set(self.non_retained_species)), \
                f"Indexes must be unique and in order."
            assert min(self.non_retained_species) >= 0
            assert max(self.non_retained_species) < self._n_species, \
                f"Index for species starts with 0."
        assert 1 >= self.relative_losses >= 0
        assert self.flow_reduction >= 1
        retained_species = [i for i in range(self._n_species)
                            if i not in self.non_retained_species]
        self._f = self._f / self.flow_reduction
        self._c[retained_species] *= \
            self.flow_reduction * (1 - self.relative_losses)


class BufferExchange(_core.UnitOperation):
    """Buffer exchange.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    exchange_ratio
        Exchange ratio (== efficiency). Must be > 0 and <= 1.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "BufferExchange".


    Attributes
    ----------
    non_retained_species
        List of indexes of non-retained species. Indexing starts with 0.
    c_exchange_buffer
        Exchange buffer composition.
    relative_losses
        Relative losses of retained species.

    Notes
    -----
    Can be combined with `Concentration` and one of the `FlowThrough`
    or `FlowThroughWithSwitching` steps
    in order to simulate unit operations such as SPTFF or UFDF
    """

    def __init__(self, t: _np.ndarray,
                 exchange_ratio: float,
                 uo_id: str, gui_title: str = "BufferExchange"):
        super().__init__(t, uo_id, gui_title)
        assert 1 >= exchange_ratio > 0
        self.exchange_ratio: float = exchange_ratio
        """Exchange ratio (== efficiency). Must be > 0 and <= 1.
        
        Share of exchange buffer in outlet buffer.
        
        """
        self.non_retained_species: _typing.Sequence[int] = []
        """Indexes of non-retained species. Indexing starts with 0."""
        self.c_exchange_buffer = _np.array([])  # dilution buffer composition
        """Concentration of species in exchange buffer.
        
        Default = empty array (= all components are 0).
        
        If defined, it must have a value for each process fluid specie.
        
        """
        self.relative_losses: float = 0.
        """Relative losses of retained species during dilution."""

    def _calc_c_exchange_buffer(self):
        """Calc concentration of species in exchange buffer."""
        assert self.c_exchange_buffer.size == 0 \
            or self.c_exchange_buffer.size == self._n_species
        if self.c_exchange_buffer.size == 0:
            self._c_exchange_buffer = _np.zeros([self._n_species, 1])
        else:
            self._c_exchange_buffer = \
                self.c_exchange_buffer.reshape(self._n_species, 1)

    # noinspection DuplicatedCode
    def _calculate(self):
        if len(self.non_retained_species) > 0:
            assert len(self.non_retained_species) < self._n_species, \
                f"All species cannot be `non_retained_species`."
            assert list(self.non_retained_species) \
                == list(set(self.non_retained_species)), \
                f"Indexes must be unique and in order."
            assert min(self.non_retained_species) >= 0
            assert max(self.non_retained_species) < self._n_species, \
                f"Index for species starts with 0."
        assert 1 >= self.exchange_ratio >= 0
        assert 1 >= self.relative_losses >= 0
        self._calc_c_exchange_buffer()
        if self.exchange_ratio == 0:
            self.log.w(f"Exchange ratio is set to 0")
        retentate_mask = _np.ones(self._n_species, dtype=bool)
        retentate_mask[self.non_retained_species] = False
        # Apply buffer exchange.
        self._c[retentate_mask] *= 1 - self.relative_losses
        self._c[~retentate_mask] *= 1 - self.exchange_ratio
        self._c += self._c_exchange_buffer * self.exchange_ratio


class FlowThrough(_core.UnitOperation):
    """Fully continuous unit operation without life cycle.

    `FlowThrough` has a constant PDF, which depends on `flow rate` and
    `void volume`.

    If `initial volume` < `void volume`,
    then the unit operation is first filled up.
    During the fill-up an ideal mixing is assumed.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    pdf
        PDF that described propagation of process fluid through the
        unit operation.

        `pdf` is updated with `void volume` and `flow rate` at runtime.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "FlowThrough".

    Notes
    -----
    **Attributes with short description:**

    Void volume (one should be defined):

    * :attr:`v_void`
    * :attr:`rt_target` - `void volume` = `rt_target` * `flow rate`

    Initial fill volume (one can be defined, optional):

    * :attr:`v_init`
    * :attr:`v_init_ratio` - `init volume` = `v_init_ratio` *
      `void volume`
    * If none are defined, then `init volume` = `void volume`

    Init fill volume concentration (optional):

    * :attr:`c_init` - must have values for each specie
    * In undefined, then all species are 0.

    Losses (optional):

    * :attr:`losses_share` - 0.1 == 10 % losses

    Losses species list (required if :attr:`losses_share` is defined):

    * :attr:`losses_species_list` - Indexes of species affected by
      losses. Indexing starts at 0.

    Examples
    --------
    A vessel is half-empty at the beginning of the process,
    then :attr:`v_init_ratio = 0.5`.
    During simulation, the vessel gets fully filled in first part.
    Ideal mixing is assumed during the first part.
    Fully filled vessel serves then as an initial state for
    the rest of the simulation.

    """

    def __init__(self, t: _np.ndarray,
                 pdf: _core.PDF,
                 uo_id: str, gui_title: str = "FlowThrough"):
        super().__init__(t, uo_id, gui_title)
        self.pdf: _core.PDF = pdf
        """Probability distribution function for describing RTD.
        
        PDF that described propagation of process fluid through the
        unit operation.

        `pdf` is updated with `void volume` and `flow rate` at runtime.
        
        """

        # void volume definition (one of those should be positive)
        self.v_void: float = -1
        """Void volume. This is effective void volume.
        
        Either `v_void` or :attr:`rt_target` should be defined.
        
        """
        self.rt_target: float = -1
        """Target residence time.
        
        Used to determine (effective) void volume of the unit operation.
        
        Either `rt_target` or :attr:`v_void` should be defined.
        
        """
        self.v_init: float = -1
        """Initial fill volume.
        
        One of `v_init` or :attr:`v_init_ratio` may be defined,
        but not both. If both are undefined, then the init volume is
        the same as void volume.
        
        """
        self.v_init_ratio: float = -1
        """Initial fill volume relative to void volume.
        
        One of `v_init` or :attr:`v_init_ratio` may be defined,
        but not both. If both are undefined, then the init volume is
        the same as void volume.
        
        """
        self.c_init: _np.ndarray = _np.array([])
        """Buffer composition in initial fill volume.
        
        E.g. equilibration buffer composition.
        
        If undefined (default) then all components are set to 0.
        
        If defined then it has to have a value for each specie.
        
        """
        self.losses_share: float = 0
        """Relative losses."""
        self.losses_species_list: _typing.Sequence[int] = []
        """Indexes of process fluid components affected by losses.
        
        Must be defined if :attr:`losses_share` > 0.
        
        """

    @_core.UnitOperation.log.setter
    def log(self, logger: _core._logger.RtdLogger):
        self._logger = logger
        # propagate logger across other elements with logging
        self._logger.set_data_tree(self._log_entity_id, self._log_tree)
        self.pdf.set_logger_from_parent(self.uo_id, logger)

    def _calc_v_void(self):
        """Void volume of the  unit operation.

        This is so-called "effective" void volume
        (dead zones are excluded).

        """
        assert self.rt_target > 0 or self.v_void > 0, \
            "Void volume must be defined."
        if self.rt_target > 0 and self.v_void > 0:
            self.log.w(f"Void volume is defined in two ways:"
                       f" By `rt_target` and `v_void`. `v_void` is used.")
        self._v_void = self.v_void if self.v_void > 0 \
            else self._f.max() * self.rt_target
        # Log.
        self.log.i_data(self._log_tree, 'v_void', self._v_void)

    def _calc_v_init(self):
        """Initial fill level in unit operation."""
        if self.v_init >= 0:
            self._v_init = self.v_init
            if self.v_init_ratio >= 0:
                self.log.w(f"Initial volume is already defined by `v_init`"
                           f" (`v_init_ratio` is ignored).")
        elif self.v_init_ratio >= 0:
            assert hasattr(self, '_v_void') and self._v_void > 0
            self._v_init = self.v_init_ratio * self._v_void
        else:
            assert hasattr(self, '_v_void') and self._v_void > 0
            self._v_init = self._v_void
        # Log.
        self.log.i_data(self._log_tree, 'v_init', self._v_init)

    def _calc_c_init(self):
        """Calc composition of equilibration buffer."""
        assert self.c_init.size == 0 or self.c_init.size == self._n_species
        if self.c_init.size == 0:
            self._c_init = _np.zeros([self._n_species, 1])
        else:
            self._c_init = self.c_init.reshape(self._n_species, 1)
        # Log.
        self.log.i_data(self._log_tree, 'c_init', self._c_init)

    def _calc_p(self):
        """Updates and evaluates flow-through PDF."""
        assert hasattr(self, '_v_void')
        self.pdf.update_pdf(v_void=self._v_void,
                            f=self._f.max(),
                            rt_mean=self._v_void / self._f.max())
        self._p = self.pdf.get_p()
        # Log.
        self.log.d_data(self._log_tree, 'p', self._p)

    def _pre_calc(self):
        """Prepare for fill-up and convolution."""
        self._calc_v_void()
        self._calc_v_init()
        self._calc_c_init()
        self._calc_p()

    # affects `_c_init`
    def _sim_init_fill_up(self):
        """Initial fill-up of the unit operation.

        This step is applicable is `v_init < v_void`.

        It changes :attr:`_c_init`, thus one needs to be careful not to
        define it after this step.

        """
        assert hasattr(self, '_v_void')
        assert hasattr(self, '_v_init')
        assert hasattr(self, '_c_init')
        # Fill up the unit operation.
        if self._v_void > self._v_init:
            fill_up_phase_i = _utils.vectors.true_start(
                _np.cumsum(self._f) * self._dt >= self._v_void - self._v_init
            )
            fill_up_volume = _np.sum(self._f[:fill_up_phase_i]) * self._dt
            fill_up_amount = _np.sum(self._f[:fill_up_phase_i]
                                     * self._c[:, :fill_up_phase_i],
                                     1) * self._dt
            self._c_init = \
                (self._c_init * self._v_init + fill_up_amount[:, _np.newaxis])\
                / (self._v_init + fill_up_volume)
            self._c[:, :fill_up_phase_i] = 0
            self._f[:fill_up_phase_i] = 0
        # Log.
        self.log.i_data(self._log_tree, 'c_init_after_fill_up', self._c_init)
        self.log.d_data(self._log_tree, 'c_after_fill_up', self._c)

    def _sim_convolution(self):
        assert self._is_flow_box_shaped(), \
            f"Inlet flow rate must be constant (or box shaped)"
        assert hasattr(self, '_c_init')
        assert hasattr(self, '_p')
        # Convolution with initial concentration.
        self._c[:, self._f > 0] = _utils.convolution.time_conv(
            dt=self._dt,
            c_in=self._c[:, self._f > 0],
            rtd=self._p,
            c_equilibration=self._c_init,
            logger=self.log
        )
        self._c[:, self._f <= 0] = 0

    def _sim_losses(self):
        if len(self.losses_species_list) == 0 or self.losses_share == 0:
            return
        assert 1 >= self.losses_share > 0
        assert min(self.losses_species_list) >= 0
        assert max(self.losses_species_list) < self._n_species
        assert list(self.losses_species_list) \
            == list(set(self.losses_species_list))
        # Apply losses.
        self._c[self.losses_species_list] *= 1 - self.losses_share

    def _calculate(self):
        # Prepare.
        self._pre_calc()
        # Simulate.
        self._sim_init_fill_up()
        self._sim_convolution()
        self._sim_losses()


class FlowThroughWithSwitching(FlowThrough):
    """Fully continuous unit operation with inline switching.

    `FlowThroughWithSwitching` has a constant PDF, which depends on
    `flow rate` and `void volume`.

    First cycle starts when the inlet flow rate is turned on.
    Cycle duration might represent a flow-through column lifetime.

    If `initial volume` < `void volume`,
    then the unit operation is first filled up.
    During the fill-up an ideal mixing is assumed.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    pdf
        PDF that described propagation of process fluid through the
        unit operation.

        `pdf` is updated with `void volume` and `flow rate` at runtime.
    uo_id
        Unique identifier.
    gui_title
        Readable title for GUI. Default = "FlowThroughWithSwitching".

    Notes
    -----
    Cycle duration (e.g. column lifetime) should be defined by one of
    the following attributes:

    * :attr:`t_cycle`
    * :attr:`v_cycle`
    * :attr:`v_cycle_relative`

    """

    def __init__(self, t: _np.ndarray,
                 pdf: _core.PDF,
                 uo_id: str,
                 gui_title: str = "FlowThroughWithSwitching"):
        super().__init__(t, pdf, uo_id, gui_title)
        self.t_cycle: float = -1
        """Cycle duration (time).
        
        E.g. lifecycle of flow-through column.
        
        Only one cycle duration definition is expected. Available:
        
        * :attr:`t_cycle` (this one)
        * :attr:`v_cycle`
        * :attr:`v_cycle_relative`
        
        """
        self.v_cycle: float = -1
        """Cycle duration (volume).
        
        Cycle duration (time) = `v_cycle` / `inlet flow rate`.
        
        E.g. lifecycle of flow-through column.
        
        Only one cycle duration definition is expected. Available:
        
        * :attr:`t_cycle`
        * :attr:`v_cycle` (this one)
        * :attr:`v_cycle_relative`
        
        """
        self.v_cycle_relative: float = -1
        """Cycle duration (relative volume).
        
        Cycle duration (time) = `v_cycle_relative` * `void volume`
        / `inlet flow rate`.
        
        
        E.g. lifecycle of flow-through column.
        
        Only one cycle duration definition is expected. Available:
        
        * :attr:`t_cycle`
        * :attr:`v_cycle`
        * :attr:`v_cycle_relative` (this one)
        
        """

    def _calc_t_cycle(self):
        assert hasattr(self, '_v_void')
        # Get cycle duration.
        if self.t_cycle > 0:
            self._t_cycle = self.t_cycle
            if self.v_cycle > 0:
                self.log.w(f"Cycle duration defined in more than one way."
                           f" `v_cycle` is ignored.")
            if self.v_cycle_relative > 0:
                self.log.w(f"Cycle duration defined in more than one way."
                           f" `v_cycle_relative` is ignored.")
        elif self.v_cycle > 0:
            self._t_cycle = self.v_cycle / self._f.max()
            if self.v_cycle_relative > 0:
                self.log.w(f"Cycle duration defined in more than one way."
                           f" `v_cycle_relative` is ignored.")
        elif self.v_cycle_relative > 0:
            self._t_cycle = \
                self.v_cycle_relative * self._v_void / self._f.max()
        else:
            raise AssertionError("Cycle duration must be defined")
        # Log.
        self.log.i_data(self._log_tree, 't_cycle', self._t_cycle)

    def _sim_piece_wise_time_convolution(self):
        assert self._is_flow_box_shaped(), \
            "Inlet flow rate must be constant (or box shaped)."
        assert hasattr(self, '_v_void')
        assert hasattr(self, '_c_init')
        assert hasattr(self, '_p')
        assert hasattr(self, '_t_cycle')
        # Convolution.
        self._c = _utils.convolution.piece_wise_time_conv(
            dt=self._dt, f_in=self._f, c_in=self._c,
            t_cycle=self._t_cycle, rt_mean=self._v_void / self._f.max(),
            rtd=self._p, c_wash=self._c_init, logger=self.log
        )

    def _calculate(self):
        # Prepare.
        self._pre_calc()
        self._calc_t_cycle()
        # Simulate.
        self._sim_init_fill_up()
        self._sim_piece_wise_time_convolution()
        self._sim_losses()
