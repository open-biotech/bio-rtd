"""
Fully continuous unit operations.
They expect steady flow rate at the inlet and produce steady flow rate at the outlet.
The flow rate can have an initial delay and/or it can stop early.
"""

__all__ = ['Dilution', 'Concentration', 'BufferExchange', 'FlowThrough', 'FlowThroughWithSwitching']
__version__ = '0.2'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.core as _core
import bio_rtd.utils as _utils


class Dilution(_core.UnitOperation):
    """
    Dilute process fluid stream in constant ratio.

    Attributes
    ----------
    dilution_ratio: int
        Dilution ratio. Must be >= 1. Dilution ratio of 1.2 means adding 20 % of dilution buffer.
        It is applied before the `delay_inlet`.
    c_add_buffer: np.ndarray
        Concentration of species in dilution buffer.
    """

    def __init__(self, t: _np.ndarray,
                 dilution_ratio: float,  # 1 == no dilution, 1.2 == 20 % addition of dilution buffer
                 uo_id: str, gui_title: str = "Dilution"):
        super().__init__(t, uo_id, gui_title)

        assert dilution_ratio > 1

        self.dilution_ratio = dilution_ratio

        self.c_add_buffer: _np.ndarray = _np.array([])

    def _calc_c_add_buffer(self):
        """ Concentration of species in dilution buffer """

        assert self.c_add_buffer.size == 0 or self.c_add_buffer.size == self._n_species

        if self.c_add_buffer.size == 0:
            self._c_add_buffer = _np.zeros([self._n_species, 1])
        else:
            self._c_add_buffer = self.c_add_buffer.reshape(self._n_species, 1)

    def _calculate(self):

        assert self.dilution_ratio >= 1
        assert hasattr(self, "_c_add_buffer")

        if self.dilution_ratio == 1:
            self.log.w("Dilution ratio is set to 1")

        # apply dilution
        self._f = self._f * self.dilution_ratio
        self._c = (self._c + (self.dilution_ratio - 1) * self._c_add_buffer) / self.dilution_ratio


class Concentration(_core.UnitOperation):
    """
    Concentrate process fluid stream

    `Concentration` step can be used before or after the `FlowThrough` or `FlowThroughWithSwitching` steps
    in order to simulate unit operations such as SPTFF or UFDF
    """

    def __init__(self, t: _np.ndarray,
                 flow_reduction: float,  # f_out = f_in / flow_reduction
                 uo_id: str, gui_title: str = "Concentration"):
        super().__init__(t, uo_id, gui_title)

        assert flow_reduction > 1

        self.flow_reduction = flow_reduction

        self.non_retained_species = []
        self.relative_losses = 0.

    # noinspection DuplicatedCode
    def _calculate(self):
        if len(self.non_retained_species) > 0:
            assert len(self.non_retained_species) < self._n_species, \
                "All species cannot be `non_retained_species`."
            assert list(self.non_retained_species) \
                == list(set(self.non_retained_species)), \
                "Indexes must be unique and in order"
            assert min(self.non_retained_species) >= 0
            assert max(self.non_retained_species) < self._n_species, \
                "Index for species starts with 0"
        assert 1 >= self.relative_losses >= 0
        assert self.flow_reduction >= 1

        retained_species = [i for i in range(self._n_species)
                            if i not in self.non_retained_species]
        self._f = self._f / self.flow_reduction
        self._c[retained_species] *= \
            self.flow_reduction * (1 - self.relative_losses)


class BufferExchange(_core.UnitOperation):
    """
    Buffer exchange

    Can be combined with `Concentration` and one of the `FlowThrough` or `FlowThroughWithSwitching` steps
    in order to simulate unit operations such as SPTFF or UFDF
    """

    def __init__(self, t: _np.ndarray,
                 exchange_ratio: float,  # 0 <= exchange_ratio <= 1
                 uo_id: str, gui_title: str = "BufferExchange"):
        super().__init__(t, uo_id, gui_title)

        assert 1 >= exchange_ratio > 0

        self.exchange_ratio = exchange_ratio  # share of inlet buffer in outlet buffer

        self.non_retained_species = []
        self.c_exchange_buffer = _np.array([])  # dilution buffer composition
        self.relative_losses = 0  # relative losses during dilution

    def _calc_c_exchange_buffer(self):
        """ Concentration of species in exchange buffer """

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
                "All species cannot be `non_retained_species`."
            assert list(self.non_retained_species) \
                == list(set(self.non_retained_species)), \
                "Indexes must be unique and in order"
            assert min(self.non_retained_species) >= 0
            assert max(self.non_retained_species) < self._n_species, \
                "Index for species starts with 0"

        assert 1 >= self.exchange_ratio >= 0
        assert 1 >= self.relative_losses >= 0

        self._calc_c_exchange_buffer()

        if self.exchange_ratio == 0:
            self.log.w("Exchange ratio is set to 0")

        retentate_mask = _np.ones(self._n_species, dtype=bool)
        retentate_mask[self.non_retained_species] = False

        self._c[retentate_mask] = self._c[retentate_mask] * (1 - self.relative_losses)
        self._c[~retentate_mask] = self._c[~retentate_mask] * (1 - self.exchange_ratio)
        self._c += self._c_exchange_buffer * self.exchange_ratio


class FlowThrough(_core.UnitOperation):
    """
    Fully continuous unit operation without inline switching

    FlowThrough has a constant PDF, which depends on process parameters.
    It assumes a constant inlet flow rate (apart from initial delay or early stop).
    It does not depend on concentration.

    If initial volume (`v_init`) < void volume (`v_void`), then the unit operation is first filled up.
    During the fill-up an ideal mixing is assumed.

    Attributes
    ----------
    v_void: float
        Effective void volume of the unit operations (v_void = rt_target / f)

        Values > 0 are accepted.
    rt_target: float
        Specified flow-through time as alternative way to define `v_void` (v_void = rt_target / f)
        Values > 0 are accepted in v_void <= 0
    v_init: float
        Effective void volume of the unit operations (v_void = rt_target / f)
        Values (v_void >= v_init >= 0) are accepted. Values > v_void result in error.
        If v_init and v_init_ratio are both undefined or out of range, v_init = v_void is assumed.
    v_init_ratio: float
        Specified flow-through time as alternative way to define `v_void` (v_void = rt_target / f)
        Values (0 >= v_init_ratio >= 1) are accepted if v_init if < 0. Values > 1 result in error.
        If v_init and v_init_ratio are both < 0, v_init == v_void is assumed.
    c_init: np.ndarray
        Concentration in v_init for each process fluid component.
        If left blank it is assumed that all components are 0.
    pdf: PDF
        Steady-state probability distribution function (see PFD class).
        pdf is updated based on passed `v_void` and `f` at runtime
    """

    def __init__(self, t: _np.ndarray, pdf: _core.PDF, uo_id: str, gui_title: str = "FlowThrough"):
        super().__init__(t, uo_id, gui_title)

        # void volume definition (one of those should be positive)
        self.v_void = -1
        self.rt_target = -1

        # initial volume definition (if both are negative, v_init == v_void is assumed)
        self.v_init = -1
        self.v_init_ratio = -1

        # initial concentration in pre-filled buffer
        # empty array means that the initial concentration is 0 for all species
        self.c_init: _np.ndarray = _np.array([])

        self.losses_share = 0
        self.losses_species_list = []

        # PDF function
        self.pdf: _core.PDF = pdf

    @_core.UnitOperation.log.setter
    def log(self, logger: _core._logger.RtdLogger):
        self._logger = logger
        # propagate logger across other elements with logging
        self._logger.set_data_tree(self._instance_id, self._log_tree)
        self.pdf.set_logger_from_parent(self.uo_id, logger)

    def _calc_v_void(self):
        """
        Void volume of the  unit operation.

        This is so-called "effective" void volume (dead zones are excluded).
        """

        assert self.rt_target > 0 or self.v_void > 0, "Void volume must be defined"

        if self.rt_target > 0 and self.v_void > 0:
            self.log.w("Void volume is defined in two ways: By `rt_target` and `v_void`. `v_void` is used.")

        self._v_void = self.v_void if self.v_void > 0 else self._f.max() * self.rt_target

        self.log.i_data(self._log_tree, 'v_void', self._v_void)

    def _calc_v_init(self):
        """
        The initial fill level in unit operation.

        Default:
            init fill level == void volume

        Use case:
            A vessel is half-empty at the beginning of the process, then `v_init_ratio = 0.5`.
            During simulation, the vessel gets fully filled in first part.
            Ideal mixing is assumed during the first part.
            Fully filled vessel serves then as an initial state for the rest of the simulation.
        """

        if self.v_init >= 0:
            self._v_init = self.v_init
            if self.v_init_ratio >= 0:
                self.log.w("Initial volume is already defined by `v_init` (`v_init_ratio` is ignored)")
        elif self.v_init_ratio >= 0:
            assert hasattr(self, '_v_void') and self._v_void > 0, "`_v_void` should be defined by now"
            self._v_init = self.v_init_ratio * self._v_void
        else:
            assert hasattr(self, '_v_void') and self._v_void > 0, "`_v_void` should be defined by now"
            self._v_init = self._v_void

        self.log.i_data(self._log_tree, 'v_init', self._v_init)

    def _calc_c_init(self):
        """ Composition of equilibration buffer """

        assert self.c_init.size == 0 or self.c_init.size == self._n_species

        # calc initial concentration of v_init
        if self.c_init.size == 0:
            self._c_init = _np.zeros([self._n_species, 1])
        else:
            self._c_init = self.c_init.reshape(self._n_species, 1)

        self.log.i_data(self._log_tree, 'c_init', self._c_init)

    def _calc_p(self):
        """ Evaluates flow-through PDF """

        assert hasattr(self, '_v_void')

        self.pdf.update_pdf(v_void=self._v_void, f=self._f.max(), rt_mean=self._v_void / self._f.max())
        self._p = self.pdf.get_p()
        self.log.d_data(self._log_tree, 'p', self._p)

    def _pre_calc(self):
        self._calc_v_void()
        self._calc_v_init()
        self._calc_c_init()
        self._calc_p()

    # affects `_c_init`
    def _sim_init_fill_up(self):
        """
        Initial fill-up of the unit operation

        This step is applicable is `v_init < v_void`.
        """

        assert hasattr(self, '_v_void')
        assert hasattr(self, '_v_init')
        assert hasattr(self, '_c_init')

        # fill up the unit operation
        if self._v_void > self._v_init:
            fill_up_phase_i = _utils.vectors.true_start(
                _np.cumsum(self._f) * self._dt >= self._v_void - self._v_init
            )
            fill_up_volume = _np.sum(self._f[:fill_up_phase_i]) * self._dt
            fill_up_amount = _np.sum(self._f[:fill_up_phase_i] * self._c[:, :fill_up_phase_i], 1) * self._dt
            self._c_init = (self._c_init * self._v_init + fill_up_amount[:, _np.newaxis]) / \
                           (self._v_init + fill_up_volume)
            self._c[:, :fill_up_phase_i] = 0
            self._f[:fill_up_phase_i] = 0

        self.log.i_data(self._log_tree, 'c_init_after_fill_up', self._c_init)
        self.log.d_data(self._log_tree, 'c_after_fill_up', self._c)

    def _sim_convolution(self):

        assert self._is_flow_box_shaped(), "Inlet flow rate must be constant (or box shaped)"
        assert hasattr(self, '_c_init')
        assert hasattr(self, '_p')

        # convolution with initial concentration
        self._c[:, self._f > 0] = _utils.convolution.time_conv(
            self._dt, self._c[:, self._f > 0], self._p, self._c_init, logger=self.log
        )
        self._c[:, self._f <= 0] = 0

    def _sim_losses(self):

        if len(self.losses_species_list) == 0 or self.losses_share == 0:
            return

        assert 1 >= self.losses_share > 0
        assert min(self.losses_species_list) >= 0
        assert max(self.losses_species_list) < self._n_species
        assert list(self.losses_species_list) == list(set(self.losses_species_list))

        # apply losses
        self._c[self.losses_species_list] *= 1 - self.losses_share

    def _calculate(self):
        # prepare
        self._pre_calc()
        # simulate
        self._sim_init_fill_up()
        self._sim_convolution()
        self._sim_losses()


class FlowThroughWithSwitching(FlowThrough):
    """
    Fully continuous unit operation with inline switching (= piece-wise FC UO)

    FlowThroughWithSwitching has a constant PDF, which depends on process parameters.
    It assumes a constant inlet flow rate (apart from initial delay or early stop).
    Its operation does not depend on concentration values
    It is periodically interrupted

    If initial volume (`v_init`) < void volume (`v_void`), then the unit operation is first filled up.
    During the fill-up an ideal mixing is assumed.
    First cycle starts when the inlet flow rate is turned on (possible initial delays are covered in UnitOperation)

    Attributes
    ----------
    t_cycle: float
        Duration of the cycle between switches
    v_cycle: float
        Alternative way to define `t_cycle`
        `t_cycle = v_cycle / f`
    v_cycle_relative: float
        Alternative way to define `t_cycle`
        `t_cycle = v_cycle_relative * v_void / f = v_cycle_relative * rt_mean`
    pdf: PDF
        Steady-state probability distribution function (see PFD class).
        pdf is updated based on passed `v_void` and `f` at runtime

    Parameters
    ----------
    v_void: float
        Effective void volume of the unit operations (v_void = rt_target / f)

        Values > 0 are accepted.
    rt_target: float
        Specified flow-through time as alternative way to define `v_void` (v_void = rt_target / f)
        Values > 0 are accepted in v_void <= 0
    c_init: np.ndarray
        Concentration in v_init for each process fluid component.
        If left blank it is assumed that all components are 0.
    """

    def __init__(self, t: _np.ndarray,
                 pdf: _core.PDF,
                 uo_id: str,
                 gui_title: str = "FlowThroughWithSwitching"):
        super().__init__(t, pdf, uo_id, gui_title)

        self.t_cycle = -1  # defines cycle duration
        self.v_cycle = -1  # defines cycle duration (_t_cycle = v_cycle / self._f[-1])
        self.v_cycle_relative = -1  # defines cycle duration (_t_cycle = v_cycle * self._v_void)

    def _calc_t_cycle(self):

        assert hasattr(self, '_v_void')

        # get cycle duration
        if self.t_cycle > 0:
            self._t_cycle = self.t_cycle
            if self.v_cycle > 0:
                self.log.w("Cycle duration defined in more than one way. `v_cycle` is ignored.")
            if self.v_cycle_relative > 0:
                self.log.w("Cycle duration defined in more than one way. `v_cycle_relative` is ignored.")
        elif self.v_cycle > 0:
            self._t_cycle = self.v_cycle / self._f.max()
            if self.v_cycle_relative > 0:
                self.log.w("Cycle duration defined in more than one way. `v_cycle_relative` is ignored.")
        elif self.v_cycle_relative > 0:
            self._t_cycle = self.v_cycle_relative * self._v_void / self._f.max()
        else:
            raise AssertionError("Cycle duration must be defined")

        self.log.i_data(self._log_tree, 't_cycle', self._t_cycle)

    def _sim_piece_wise_convolution(self):

        assert self._is_flow_box_shaped(), "Inlet flow rate must be constant (or box shaped)"
        assert hasattr(self, '_v_void')
        assert hasattr(self, '_c_init')
        assert hasattr(self, '_p')
        assert hasattr(self, '_t_cycle')

        # convolution with initial concentration
        self._c = _utils.convolution.piece_wise_conv_with_init_state(
            dt=self._dt, f_in=self._f, c_in=self._c,
            t_cycle=self._t_cycle, rt_mean=self._v_void / self._f.max(),
            rtd=self._p, c_wash=self._c_init, logger=self.log
        )

    def _calculate(self):
        # prepare
        self._pre_calc()
        self._calc_t_cycle()
        # simulate
        self._sim_init_fill_up()
        self._sim_piece_wise_convolution()
        self._sim_losses()
