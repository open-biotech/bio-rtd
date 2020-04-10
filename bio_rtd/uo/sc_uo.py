"""Semi continuous unit operations.

Unit operations that accept constant or box-shaped flow rate profile
and provide periodic flow rate profile.

"""
__all__ = ['AlternatingChromatography', 'ACC', 'PCC', 'PCCWithWashDesorption']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np
import scipy.interpolate as _interp

from bio_rtd.chromatography import bt_load as _bt_load
import bio_rtd.utils as _utils
import bio_rtd.core as _core
import bio_rtd.pdf as _pdf


class AlternatingChromatography(_core.UnitOperation):
    """Simulation of alternating chromatography.

    This class implements logic common to various types of alternating
    chromatography. It has a role of a base class for
    specific types of alternating chromatography to extend.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    load_bt
        Load breakthrough logic.
    peak_shape_pdf
        Elution peak shape.
    gui_title
        Readable title for GUI. Default = "AC".

    Notes
    -----
    **Quick description of which attributes are available:**

    Non-binding species (optional):

    * :attr:`non_binding_species`

    Column volume (exactly one required):

    * :attr:`cv`
    * :attr:`ft_mean_retentate` and :attr:`column_porosity_retentate`

    Column porosity for binding species (required in case of
    :attr:`ft_mean_retentate` or wash or load recycling):

    * :attr:`column_porosity_retentate`

    Equilibration step duration (optional, if both, the values are
    added together):

    * :attr:`equilibration_cv`
    * :attr:`equilibration_t`

    Equilibration step flow rate (exactly one needed):

    * :attr:`equilibration_f` - absolute, has priority if defined
    * :attr:`equilibration_f_rel` - relative, default = 1

    Load step duration:

    * :attr:`load_cv` - preferred
    * :attr:`load_c_end_ss` - concentration limit for breakthrough; also
      requires :attr:`load_recycle_pdf`
    * :attr:`load_c_end_relative_ss` - concentration limit for
      breakthrough relative to steady-state load concentration; also
      requires :attr:`load_recycle_pdf`

    Iterative optimization of estimation of load step duration
    (ignored if :attr:`load_cv` is defined):

    * :attr:`load_c_end_estimate_with_iterative_solver` - default = True
    * :attr:`load_c_end_estimate_with_iter_solver_max_iter` - default =
      1000

    Extension of first load step (optional; ignored if no recycling):

    * :attr:`load_extend_first_cycle` - default = `False`
    * :attr:`load_extend_first_cycle_cv` and
      :attr:`load_extend_first_cycle_t` - added together if both defined

    Load linear velocity - only for column height determination
    (optional):

    * :attr:`load_target_lin_velocity`

    Wash step duration (optional, if both, the values are
    added together):

    * :attr:`wash_cv`
    * :attr:`wash_t`

    Wash step flow rate (exactly one needed):

    * :attr:`wash_f` - absolute, has priority if defined
    * :attr:`wash_f_rel` - relative, default = 1

    Unaccounted losses - applied before peak cut (optional):

    * :attr:`unaccounted_losses_rel` - relative, default = 1

    Elution step duration (optional, if both, the values are
    added together):

    * :attr:`elution_cv`
    * :attr:`elution_t`

    Elution step flow rate (exactly one needed):

    * :attr:`elution_f` - absolute, has priority if defined
    * :attr:`elution_f_rel` - relative, default = 1

    Elution buffer composition (optional):

    * :attr:`elution_buffer_c`


    Elution peak position duration - first momentum
    (optional, if both, the values are added together):

    * :attr:`elution_peak_position_cv`
    * :attr:`elution_peak_position_t`

    Elution peak cut start (one is required):

    * :attr:`elution_peak_cut_start_t`
    * :attr:`elution_peak_cut_start_cv`
    * :attr:`elution_peak_cut_start_c_rel_to_peak_max`
    * :attr:`elution_peak_cut_start_peak_area_share`

    Elution peak cut end (one is required):

    * :attr:`elution_peak_cut_end_t`
    * :attr:`elution_peak_cut_end_cv`
    * :attr:`elution_peak_cut_end_c_rel_to_peak_max`
    * :attr:`elution_peak_cut_end_peak_area_share`

    Regeneration step duration (optional, if both, the values are
    added together):

    * :attr:`regeneration_cv`
    * :attr:`regeneration_t`

    Regeneration step flow rate (exactly one needed):

    * :attr:`regeneration_f` - absolute, has priority if defined
    * :attr:`regeneration_f_rel` - relative, default = 1

    Wash desorption (optional, also check if class supports it):

    * :attr:`wash_desorption` - default = `False`

    Load breakthrough recycle (optional):

    * :attr:`load_recycle` - default = `False`

    Load breakthrough propagation dynamics
    (required if :attr:`load_recycle` is `True`
    or :attr:`load_c_end_ss` is defined or
    or :attr:`load_c_end_relative_ss` is defined):

    * :attr:`load_recycle_pdf`

    Wash recycle (optional):

    * :attr:`wash_recycle` - default = `False`

    Duration of wash recycling
    (optional; ignored if :attr:`wash_recycle` is `False`):

    * :attr:`wash_recycle_duration_cv` and
      :attr:`wash_recycle_duration_t` - summed together if both defined.
    * Entire wash step if
      :attr:`wash_recycle_duration_cv` and
      :attr:`wash_recycle_duration_t` are not defined.


    Please note that subclasses might introduce new attributes or change
    the default values of existing attributes.

    """

    def __init__(self,
                 t: _np.ndarray,
                 uo_id: str,
                 load_bt: _core.ChromatographyLoadBreakthrough,
                 peak_shape_pdf: _core.PDF,
                 gui_title: str = "AC"):
        super().__init__(t, uo_id, gui_title)
        # Bind parameters.
        self.load_bt: _core.ChromatographyLoadBreakthrough = load_bt
        """Determines what part of load material binds to the column."""
        self.elution_peak_shape: _core.PDF = peak_shape_pdf
        """Elution peak shape."""
        self.non_binding_species: _typing.Sequence[int] = []
        """Process buffer species that are NOT binding to the column.
        
        Indexing starts with 0.
        
        """
        self.cv: float = -1
        """Column volume.
        
        Column volume should be defined by exactly one of the following
        attribute groups:
        
        * :attr:`cv` (this one)
        * :attr:`ft_mean_retentate`
          and :attr:`column_porosity_retentate`
        
        """
        self.ft_mean_retentate: float = -1
        """Flow-through time of retentate under non-binding conditions.
        
        Used to define column volume (independently of scale).
        
        Column volume should be defined by exactly one of the following
        attribute groups:
        
        * :attr:`cv`
        * :attr:`ft_mean_retentate` (this one) and 
          :attr:`column_porosity_retentate`
        
        """
        self.column_porosity_retentate: float = -1
        """Column porosity for retentate under non-binding conditions.
        
        Required in case :attr:`ft_mean_retentate` is used to define
        column volume.
        
        Required in case :attr:`load_c_end_ss` or
        :attr:`load_c_end_relative_ss` are used to estimate
        load step duration.
        
        Required in case of load or wash recycling.
        
        """
        self.equilibration_cv: float = -1
        """Duration of equilibration step.
        
        The values of :attr:`equilibration_t` and 
        :attr:`equilibration_cv` are added together.
        
        """
        self.equilibration_t: float = -1
        """Duration of equilibration step.
        
        The values of :attr:`equilibration_t` and 
        :attr:`equilibration_cv` are added together.
        
        """
        self.equilibration_f: float = -1
        """Equilibration step flow rate.
        
        Equilibration step flow rate should be defined by
        exactly one of the following attributes:
        
        * :attr:`equilibration_f` (this one)
        * :attr:`equilibration_f_rel`
        
        """
        self.equilibration_f_rel: float = 1
        """Equilibration step flow rate relative to load flow rate.
        
        Default = 1.
        
        Equilibration step flow rate = :attr:`equilibration_f_rel`
        * `load flow rate`
        
        Equilibration step flow rate should be defined by
        exactly one of the following attributes:
        
        * :attr:`equilibration_f`
        * :attr:`equilibration_f_rel` (this one)
        
        """
        # Duration of the load phase.
        self.load_cv: float = -1  # load duration in CV
        """Load phase duration in CV.
        
        This is preferable way to define the duration of the load step
        as it does not require any estimations about steady state.
        
        Load phase duration should be defined by exactly one of
        the following attribute groups:
        
        * :attr:`load_cv` (this one)
        * :attr:`load_c_end_ss`
        * :attr:`load_c_end_relative_ss`
        
        Notes
        -----
        First load step can be extended by setting
        :attr:`load_extend_first_cycle`,
        :attr:`load_extend_first_cycle_cv` and
        :attr:`load_extend_first_cycle_t`.
        
        """
        self.load_c_end_ss: _typing.Optional[_np.ndarray] = None
        """Load phase switch based on target product breakthrough conc.
        
        Load phase duration is estimated from simulating steady state
        operation and determining when the breakthrough reaches
        specified concentration.
        
        Steady state simulation requires
        :attr:`column_porosity_retentate`
        :attr:`load_recycle_pdf`.
        
        Load phase duration should be defined by exactly one of
        the following attribute groups:
        
        * :attr:`load_cv` (preferred)
        * :attr:`load_c_end_ss` (this one)
        * :attr:`load_c_end_relative_ss`
        
        Notes
        -----
        First load step can be extended by setting
        :attr:`load_extend_first_cycle`,
        :attr:`load_extend_first_cycle_cv` and
        :attr:`load_extend_first_cycle_t`.
        
        """
        self.load_c_end_relative_ss: float = -1
        """Load phase switch based on relative breakthrough conc.
        
        Load phase duration is estimated from simulating steady state
        operation and determining when the product (binding species)
        in the breakthrough reaches specified relative concentration
        (relative to load concentration in steady-state operation).
        
        Steady state simulation requires
        :attr:`column_porosity_retentate`
        :attr:`load_recycle_pdf`.
        
        Load phase duration should be defined by exactly one of
        the following attribute groups:
        
        * :attr:`load_cv` (preferred)
        * :attr:`load_c_end_ss`
        * :attr:`load_c_end_relative_ss` (this one)
        
        Notes
        -----
        First load step can be extended by setting
        :attr:`load_extend_first_cycle`,
        :attr:`load_extend_first_cycle_cv` and
        :attr:`load_extend_first_cycle_t`.
        
        """
        self.load_c_end_estimate_with_iterative_solver: bool = True
        """Finer optimization of cycle length estimation.
        
        Default = `True`.
        
        In case load step duration is estimated based of breakthrough
        criteria (i.e. by :attr:`load_c_end_ss` or
        :attr:`load_c_end_relative_ss`), the model needs to simulate
        steady-state operation in order to determine fixed load time.
        This parameters enables iterative solver that allows more
        precise estimation but might slow down the simulation.
        
        Notes
        -----
        Max number of iteration steps is defined by
        :attr:`load_c_end_estimate_with_iter_solver_max_iter`.
        
        """
        self.load_c_end_estimate_with_iter_solver_max_iter: int = 1000
        """Max steps for optimization of cycle length estimation.
        
        Default = 1000.
        
        See Also
        --------
        :attr:`load_c_end_estimate_with_iterative_solver`
        
        """
        self.load_extend_first_cycle: bool = False
        """Extend first load phase to achieve a faster steady-state.
        
        Only relevant in case wash or load is recycled.
        
        The duration of extension is defined by:
        
        * :attr:`load_extend_first_cycle_cv` or
        * :attr:`load_extend_first_cycle_t` or
        * is determined automatically.
        
        """
        self.load_extend_first_cycle_cv: float = -1
        """Duration of first load phase extension in column volumes.
        
        Only relevant if :attr:`load_extend_first_cycle` is `True`.
        
        If the duration if defined by
        :attr:`load_extend_first_cycle_cv` and
        :attr:`load_extend_first_cycle_t`
        then the values are added together.
        
        """
        self.load_extend_first_cycle_t: float = -1
        """Duration of first load phase extension (time).
        
        Only relevant if :attr:`load_extend_first_cycle` is `True`.
        
        If the duration if defined by
        :attr:`load_extend_first_cycle_cv` and
        :attr:`load_extend_first_cycle_t`
        then the values are added together.
        
        """
        self.load_target_lin_velocity: float = -1
        """Target load linear velocity.
        
        It is used to provide information about required column height.
        
        It does not have any impact on the rest of the model.
        
        Units need to match other units in the model.
        
        """
        self.wash_cv: float = -1
        """Duration of wash step.

        The values of :attr:`wash_t` and 
        :attr:`wash_cv` are added together.

        """
        self.wash_t: float = -1
        """Duration of wash step.

        The values of :attr:`wash_t` and 
        :attr:`wash_cv` are added together.

        """
        self.wash_f: float = -1
        """Wash step flow rate.

        Wash step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`wash_f` (this one)
        * :attr:`wash_f_rel`

        """
        self.wash_f_rel: float = 1
        """Wash step flow rate relative to load flow rate. Default = 1.

        Wash step flow rate = :attr:`wash_f_rel`
        * `load flow rate`

        Wash step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`wash_f`
        * :attr:`wash_f_rel` (this one)

        """
        self.unaccounted_losses_rel: float = 0
        """Unaccounted losses as a share of bound material.
        
        Elution peak is scaled down by 1 - `unaccounted_losses_rel`
        before applying peak cut criteria.
        
        """
        self.elution_cv: float = -1
        """Duration of elution step.

        The values of :attr:`elution_t` and 
        :attr:`elution_cv` are added together.

        """
        self.elution_t: float = -1
        """Duration of elution step.

        The values of :attr:`elution_t` and 
        :attr:`elution_cv` are added together.

        """
        self.elution_f: float = -1
        """Elution step flow rate.

        Elution step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`elution_f` (this one)
        * :attr:`elution_f_rel`

        """
        self.elution_f_rel: float = 1
        """Elution step flow rate relative to load flow rate.

        Default = 1.

        Elution step flow rate = :attr:`elution_f_rel`
        * `load flow rate`

        Elution step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`elution_f`
        * :attr:`elution_f_rel` (this one)

        """
        self.elution_buffer_c: _np.ndarray = _np.array([])
        """Elution buffer composition.
        
        Default = empty array (= all components are 0).
        
        If defined it must have a value for each specie.
        
        """
        self.elution_peak_position_cv: float = -1
        """Position (cv) of elution peak in the elution step.
        
        This is for 1st moment or mean residence time (and not
        necessarily peak max position).

        The values of :attr:`elution_peak_position_t` and 
        :attr:`elution_peak_position_cv` are added together.

        """
        self.elution_peak_position_t: float = -1
        """Position (time) of elution peak in the elution step.
        
        This is for 1st moment or mean residence time (and not
        necessarily peak max position).

        The values of :attr:`elution_peak_position_t` and 
        :attr:`elution_peak_position_cv` are added together.

        """
        self.elution_peak_cut_start_t: float = -1
        """Elution peak cut start (time).
        
        Exactly one peak cut start criteria should be defined.
        
        """
        self.elution_peak_cut_start_cv: float = -1
        """Elution peak cut start (cv).
        
        Exactly one peak cut start criteria should be defined.
        
        """
        self.elution_peak_cut_start_c_rel_to_peak_max: float = -1
        """Elution peak cut start (signal relative to peak max).
        
        Exactly one peak cut start criteria should be defined.
        
        """
        self.elution_peak_cut_start_peak_area_share: float = -1
        """Elution peak cut start (share of total peak area).
        
        Exactly one peak cut start criteria should be defined.
        
        """
        self.elution_peak_cut_end_t: float = -1
        """Elution peak cut end (time).
        
        Exactly one peak cut end criteria should be defined.
        
        """
        self.elution_peak_cut_end_cv: float = -1
        """Elution peak cut end (cv).
        
        Exactly one peak cut end criteria should be defined.
        
        """
        self.elution_peak_cut_end_c_rel_to_peak_max: float = -1
        """Elution peak cut end (signal relative to peak max).
        
        Exactly one peak cut end criteria should be defined.
        
        """
        self.elution_peak_cut_end_peak_area_share: float = -1
        """Elution peak cut end (share of total peak area).
        
        Exactly one peak cut end criteria should be defined.
        
        """
        self.regeneration_cv: float = -1
        """Duration of regeneration step.

        The values of :attr:`regeneration_t` and 
        :attr:`regeneration_cv` are added together.

        """
        self.regeneration_t: float = -1
        """Duration of regeneration step.

        The values of :attr:`regeneration_t` and 
        :attr:`regeneration_cv` are added together.

        """
        self.regeneration_f: float = -1
        """Regeneration step flow rate.

        Regeneration step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`regeneration_f` (this one)
        * :attr:`regeneration_f_rel`

        """
        self.regeneration_f_rel: float = 1
        """Regeneration step flow rate relative to load flow rate.
        
        Default = 1.

        Regeneration step flow rate = :attr:`regeneration_f_rel`
        * `load flow rate`

        Regeneration step flow rate should be defined by
        exactly one of the following attributes:

        * :attr:`regeneration_f`
        * :attr:`regeneration_f_rel` (this one)

        """
        self.wash_desorption: bool = False
        """Enable wash desorption.
        
        Make sure the class implements the desorption dynamics.
        
        """
        self.load_recycle: bool = False
        """Recycle load breakthrough. Default = False."""
        self.load_recycle_pdf: _typing.Optional[_core.PDF] = None
        """PDF of wash and/or unbound load traveling through the column.
        
        The unbound (not captured) part and desorbed part are propagated
        through the column by :attr:`load_recycle_pdf`.

        Void volume for :attr:`load_recycle_pdf` is defined as
        :attr:`column_porosity_retentate` * `column volume`.
        
        """
        self.wash_recycle: bool = False
        """Recycle wash. Default = False.
        
        Wash is recycled onto 3rd column while the 2nd is on load step.
        After the wash recycle, the 3rd column is connected to 2nd
        column to recycle load breakthrough material.
        
        """
        self.wash_recycle_duration_cv: float = -1
        """Duration of wash recycle (cv).
        
        Relevant if :attr:`wash_recycle` is `True`.
        
        If both (`wash_recycle_duration_cv` and
        :attr:`wash_recycle_duration_t`) are defined, then the values
        are added together. If none of those is defined, then the
        entire wash step is recycled.
        
        """
        self.wash_recycle_duration_t: float = -1
        """Duration of wash recycle (time).
        
        Relevant if :attr:`wash_recycle` is `True`.
        
        If both (`wash_recycle_duration_t` and
        :attr:`wash_recycle_duration_cv`) are defined, then the values
        are added together. If none of those is defined, then the
        entire wash step is recycled.
        
        """

    @_core.UnitOperation.log.setter
    def log(self, logger: _core._logger.RtdLogger):
        """Propagates logger across other elements that support it."""
        # Default logic.
        self._logger = logger
        self._logger.set_data_tree(self._log_entity_id, self._log_tree)
        # Propagate logger across other elements with logging.
        if self.load_recycle_pdf is not None:
            self.load_recycle_pdf.set_logger_from_parent(self.uo_id, logger)
        if self.load_recycle_pdf is not None:
            self.elution_peak_shape.set_logger_from_parent(self.uo_id, logger)
        if self.load_recycle_pdf is not None:
            self.load_bt.set_logger_from_parent(self.uo_id, logger)

    def _get_flow_value(self,
                        step_name: str, var_name: str,
                        flow: float, rel_flow: float) -> float:
        """Calc flow rate of chromatographic step.

        If `flow` is specified, `flow` is used.
        Otherwise `rel_flow` == flow rate relative to load flow rate is
        used.

        If none are positive, then the load flow rate is used
        and a warning is logged.

        Parameters
        ----------
        step_name
            Step name (e.g. "Wash") for log messages.
        var_name
            Step variable name (e.g. "wash_t") for log data.
        flow
            Flow rate.
        rel_flow
            Flow rate relative to load flow rate.

        Returns
        -------
        float
            Flow rate.

        """
        if flow > 0:
            self.log.i_data(self._log_tree, var_name, flow)
        elif rel_flow > 0:
            flow = rel_flow * self._load_f
            self.log.i_data(self._log_tree, var_name, flow)
        else:
            self.log.w(f"{step_name} step flow rate is not defined,"
                       f" using load flow rate instead.")
            flow = self._load_f
        return flow

    def _get_time_value(self,
                        step_name: str, var_name: str,
                        t: float, cv: float, flow: float) -> float:
        """Calc duration of chromatographic step.

        If the step duration is specified in cv and in t, then the
        value are added together.

        Parameters
        ----------
        step_name
            Step name (e.g. "Wash") for log messages.
        var_name
            Step variable name (e.g. "wash_t") for log data.
        t
            Duration (time).
        cv
            Duration (cv).
        flow
            Flow rate (required if `cv` > 0).

        Returns
        -------
        float
            Total step duration (time).

        """
        # Calc.
        t_sum = max(t, 0)
        if cv > 0:
            assert flow > 0, f"{step_name}: Flow rate must be defined (> 0)" \
                             f" if the duration is specified in CVs."
            assert self._cv > 0, f"CV must be determined (by `calc_cv`)" \
                                 f" before calculating duration based on CVs."
            t_sum += cv * self._cv / flow  # sum
        # Log.
        if t <= 0 and cv <= 0:
            self.log.w(step_name + " time is not defined")
        else:
            self.log.i_data(self._log_tree, var_name, t_sum)
        return t_sum

    def _assert_non_binding_species(self):
        """Make sure binding species list is valid."""
        if len(self.non_binding_species) > 0:
            assert max(self.non_binding_species) < self._n_species, \
                "Index of non_binding_species too large (indexes start with 0)"
            assert list(set(self.non_binding_species)) \
                == list(self.non_binding_species), \
                "List of non_binding_species should have ascending order"
            assert len(self.non_binding_species) < self._n_species, \
                "All species cannot be non-binding."
        # Log.
        self.log.i_data(self._log_tree,
                        'non_binding_species',
                        self.non_binding_species)

    def _calc_load_f(self):
        """Determine load flow rate (when on)."""
        assert self._is_flow_box_shaped(), "Inlet flow must be box shaped."
        self._load_f = self._f.max()
        self.log.d_data(self._log_tree, 'load_f', self._load_f)

    def _calc_cv(self):
        """Determine column volume."""
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.ERROR,
            cv=self.cv,
            ft_mean_retentate=self.ft_mean_retentate,
        )
        if self.cv > 0:
            self._cv = self.cv
        else:  # `self.ft_mean_retentate` > 0.
            assert self.column_porosity_retentate > 0, \
                f"porosity_retentate must be defined to calc CV from " \
                f" `self.ft_mean_retentate`."
            assert self._load_f > 0, f"Load flow rate must be defined to" \
                                     f" calc CV from `self.ft_mean_retentate`."
            self._cv = self.ft_mean_retentate * self._load_f \
                / self.column_porosity_retentate
        # Log.
        self.log.i_data(self._log_tree, 'cv', self._cv)

    def _report_column_dimensions(self):
        """Report column dimensions based on load linear velocity."""
        if self.load_target_lin_velocity > 0:
            self._col_h = self._cv * self.load_target_lin_velocity \
                          / self._load_f
            self.log.i_data(self._log_tree, "column_h", self._col_h)
            self.log.i_data(self._log_tree,
                            "column_d",
                            (self._cv / self._col_h / _np.pi) ** 0.5 * 2)

    def _calc_equilibration_t(self):
        """Determine equilibration step duration."""
        if self.equilibration_cv > 0:
            # Flow rate.
            eq_f = self._get_flow_value("Equilibration",
                                        "equilibration_f",
                                        self.equilibration_f,
                                        self.equilibration_f_rel)
            # Duration.
            self._equilibration_t = self._get_time_value("Equilibration",
                                                         "equilibration_t",
                                                         self.equilibration_t,
                                                         self.equilibration_cv,
                                                         eq_f)
        else:
            # Duration.
            self._equilibration_t = max(self.equilibration_t, 0)
        # Log.
        self.log.i_data(self._log_tree,
                        'equilibration_t',
                        self._equilibration_t)

    def _calc_wash_t_and_f(self):
        """Determine wash step flow rate and duration."""
        # Flow rate.
        self._wash_f = self._get_flow_value("Wash",
                                            "wash_f",
                                            self.wash_f,
                                            self.wash_f_rel)
        # Duration.
        self._wash_t = self._get_time_value("Wash",
                                            "wash_t",
                                            self.wash_t,
                                            self.wash_cv,
                                            self._wash_f)

    def _calc_elution_t_and_f(self):
        """Determine elution step flow rate and duration."""
        # Flow rate.
        self._elution_f = self._get_flow_value("Elution",
                                               "elution_f",
                                               self.elution_f,
                                               self.elution_f_rel)
        # Duration.
        self._elution_t = self._get_time_value("Elution",
                                               "elution_t",
                                               self.elution_t,
                                               self.elution_cv,
                                               self._elution_f)

    def _calc_elution_peak_t(self):
        """Determine elution peak mean position (1st momentum)."""
        self._elution_peak_t = self._get_time_value(
            "elution peak position",
            "elution_peak_position_t",
            self.elution_peak_position_t,
            self.elution_peak_position_cv,
            self._elution_f
        )

    def _update_elution_peak_pdf(self):
        """Update elution peak PDF."""
        assert self._elution_peak_t > 0
        assert self._elution_f > 0
        # Calc elution peak shape.
        self.elution_peak_shape.update_pdf(
            rt_mean=self._elution_peak_t,
            v_void=self._elution_peak_t * self._elution_f,
            f=self._elution_f
        )
        self._p_elution_peak = \
            self.elution_peak_shape.get_p() * (1 - self.unaccounted_losses_rel)
        self.log.d_data(self._log_tree,
                        "p_elution_peak",
                        self._p_elution_peak)

    def _calc_elution_peak_cut_i_start_and_i_end(self):
        """Calc elution peak cut start and end in form of time steps.

        Values are relative to the beginning of the elution step.

        """
        elution_peak_pdf: _np.ndarray = self._p_elution_peak.copy()
        # Peak cut start.
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            elution_peak_cut_start_peak_area_share=self
            .elution_peak_cut_start_peak_area_share,
            elution_peak_cut_start_c_rel_to_peak_max=self
            .elution_peak_cut_start_c_rel_to_peak_max,
            elution_peak_cut_start_cv=self.elution_peak_cut_start_cv,
            elution_peak_cut_start_t=self.elution_peak_cut_start_t
        )
        # Calc `elution_peak_cut_start_i`.
        if self.elution_peak_cut_start_peak_area_share >= 0:
            elution_peak_cut_start_i = _utils.vectors.true_start(
                _np.cumsum(elution_peak_pdf * self._dt)
                >= self.elution_peak_cut_start_peak_area_share
            )
        elif self.elution_peak_cut_start_c_rel_to_peak_max >= 0:
            elution_peak_cut_start_i = _utils.vectors.true_start(
                elution_peak_pdf
                >= self.elution_peak_cut_start_c_rel_to_peak_max
                * elution_peak_pdf.max()
            )
        elif self.elution_peak_cut_start_cv >= 0:
            elution_peak_cut_start_i = \
                int(self.elution_peak_cut_start_cv
                    * self._cv / self._elution_f / self._dt)
        elif self.elution_peak_cut_start_t >= 0:
            elution_peak_cut_start_i = \
                int(self.elution_peak_cut_start_t / self._dt)
        else:
            self.log.w(f"Elution peak cut start is not defined."
                       f" Now collecting from the beginning"
                       f" of the elution phase.")
            elution_peak_cut_start_i = 0
        # Log.
        self.log.i_data(self._log_tree,
                        "elution_peak_cut_start_i",
                        elution_peak_cut_start_i)
        self.log.i_data(self._log_tree,
                        "elution_peak_cut_start_t",
                        elution_peak_cut_start_i * self._dt)
        # Peak cut end.
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            elution_peak_cut_end_peak_area_share=self
            .elution_peak_cut_end_peak_area_share,
            elution_peak_cut_end_c_rel_to_peak_max=self
            .elution_peak_cut_end_c_rel_to_peak_max,
            elution_peak_cut_end_cv=self.elution_peak_cut_end_cv,
            elution_peak_cut_end_t=self.elution_peak_cut_end_t,
        )
        # Calc `elution_peak_cut_end_i`.
        if self.elution_peak_cut_end_peak_area_share >= 0:
            elution_peak_cut_end_i = _utils.vectors.true_start(
                _np.cumsum(elution_peak_pdf * self._dt)
                >= (1 - self.elution_peak_cut_end_peak_area_share)
            )
        elif self.elution_peak_cut_end_c_rel_to_peak_max >= 0:
            elution_peak_cut_end_i = _utils.vectors.true_end(
                elution_peak_pdf
                >= self.elution_peak_cut_end_c_rel_to_peak_max
                * elution_peak_pdf.max()
            )
        elif self.elution_peak_cut_end_cv >= 0:
            elution_peak_cut_end_i = \
                int(self.elution_peak_cut_end_cv
                    * self._cv / self._elution_f / self._dt)
        elif self.elution_peak_cut_end_t >= 0:
            elution_peak_cut_end_i = \
                _utils.vectors.true_end(self._t < self.elution_peak_cut_end_t)
        else:
            self.log.w(f"Elution peak cut end is not defined."
                       f" Now collecting to the end of the elution phase.")
            elution_peak_cut_end_i = elution_peak_pdf.size
        self._elution_peak_cut_start_i = elution_peak_cut_start_i
        self._elution_peak_cut_end_i = elution_peak_cut_end_i
        # Log.
        self.log.i_data(self._log_tree,
                        "elution_peak_cut_end_i",
                        elution_peak_cut_end_i)
        self.log.i_data(self._log_tree,
                        "elution_peak_cut_end_t",
                        elution_peak_cut_end_i * self._dt)
        if self._elution_peak_cut_end_i * self._dt < self._elution_peak_t:
            self.log.w(f"Peak end is cut before its maximum.")
        if self._elution_peak_cut_end_i * self._dt > self._elution_t:
            self.log.w(f"Peak cut end exceeds elution step duration.")

    def _calc_elution_peak_mask(self):
        """Calc where the elution peak gets collected."""
        self._elution_peak_mask = \
            _np.ones(int(round(self._elution_t / self._dt)), dtype=bool)
        self._elution_peak_mask[self._elution_peak_cut_end_i:] = False
        self._elution_peak_mask[:self._elution_peak_cut_start_i] = False
        self.log.d_data(self._log_tree,
                        "elution_peak_interval",
                        self._elution_peak_mask)

    def _update_load_btc(self):
        """Update load breakthrough profile."""
        assert self._cv > 0, "CV must be defined by now."
        self.load_bt.update_btc_parameters(cv=self._cv)

    def _calc_regeneration_t(self):
        """Calc regeneration step duration."""
        if self.regeneration_cv > 0:
            eq_f = self._get_flow_value("Regeneration",
                                        "regeneration_f",
                                        self.regeneration_f,
                                        self.regeneration_f_rel)
            self._regeneration_t = self._get_time_value("Regeneration",
                                                        "regeneration_t",
                                                        self.regeneration_t,
                                                        self.regeneration_cv,
                                                        eq_f)
        else:
            self._regeneration_t = max(self.regeneration_t, 0)
        # Log.
        self.log.i_data(self._log_tree, 'regeneration_t', self._regeneration_t)

    def _update_load_recycle_pdf(self, flow):
        """Update pdf that describes propagation of recycled material.

        Recycled material si composed of unbound (load) and desorbed
        (wash) material throughout the column.

        `self.load_recycle_pdf` gets updated.

        """
        assert self.load_recycle_pdf is not None, \
            f"`load_recycle_pdf` must be defined by now."
        assert self.column_porosity_retentate > 0, \
            f"Retentate porosity must be defined by now."
        assert self._cv > 0, "CV must be defined by now."
        v_void = self._cv * self.column_porosity_retentate
        self.load_recycle_pdf.update_pdf(v_void=v_void,
                                         f=flow,
                                         rt_mean=v_void / flow)
        self._p_load_recycle_pdf = self.load_recycle_pdf.get_p()

    def _calc_load_recycle_wash_i(self):
        """Calculate wash recycle duration in form of time steps."""
        if self.wash_recycle_duration_t > 0 \
                or self.wash_recycle_duration_cv > 0:
            self._wash_recycle_i_duration = int(self._get_time_value(
                "Wash recycle", "load_wash_recycle_t",
                self.wash_recycle_duration_t,
                self.wash_recycle_duration_cv,
                self._wash_f
            ) / self._dt)
        else:
            # Same as wash duration.
            assert self._wash_t > 0
            self._wash_recycle_i_duration = int(round(self._wash_t / self._dt))

    def _get_load_bt_cycle_switch_criteria(self,
                                           load_c_ss: _np.ndarray
                                           ) -> _np.ndarray:
        """Get steady-state cycle switch (== end of load) criteria.

        Parameters
        ----------
        load_c_ss
            Load concentration during steady state operation.

        Returns
        -------
        ndarray
            Threshold concentration for load breakthrough.

        """
        assert self.load_c_end_ss is not None \
            or self.load_c_end_relative_ss > 0, \
            f"Load step duration should be defined!"
        if self.load_c_end_ss is not None:
            load_c_end_ss = self.load_c_end_ss
            if self.load_c_end_relative_ss > 0:
                self.log.w(f"Cycle time defined by `load_c_end_ss`"
                           f" and `load_c_end_relative_ss`."
                           f" Simulation is using `load_c_end_ss`.")
        else:  # self.load_c_end_relative_ss > 0
            load_c_end_ss = self.load_c_end_relative_ss * load_c_ss
        # Log.
        self.log.i_data(self._log_tree,
                        'load_c_end_ss',
                        load_c_end_ss)
        return load_c_end_ss

    # noinspection DuplicatedCode
    def _calc_cycle_t(self):
        """Calculates cycle time (== load time for a single column).

        Optional delay of first cycle is not part of this calculation.

        """
        assert self._cv > 0
        assert self._load_f > 0
        if self.load_cv > 0:
            t_cycle = self.load_cv * self._cv / self._load_f
            if self.load_c_end_ss is not None \
                    or self.load_c_end_relative_ss > 0:
                self.log.w(f"Cycle time defined in more than one way."
                           f" Simulation is using `load_cv`.")
        else:
            # Get bt profile for constant inlet.
            # Inlet conc.
            binding_species = [i for i in range(self._n_species)
                               if i not in self.non_binding_species]
            load_c_ss = self._estimate_steady_state_mean_c(binding_species)
            # Simulate first cycle at constant load concentration.
            f_first_load = self._load_f * _np.ones(self._t.size)
            c_first_load = load_c_ss * _np.ones([len(binding_species),
                                                 self._t.size])
            bt_first_load: _np.ndarray = \
                load_c_ss - self.load_bt.calc_c_bound(f_first_load,
                                                      c_first_load)
            # Propagate breakthrough.
            bt_first_load_out, bt_first_wash_out = \
                self._sim_c_recycle_propagation(f_first_load,
                                                bt_first_load,
                                                None)
            # Calc cycle duration.
            load_c_end_ss = self._get_load_bt_cycle_switch_criteria(load_c_ss)
            # noinspection PyTypeChecker
            i_t_first_cycle = _utils.vectors.true_start(
                bt_first_load_out.sum(0) >= load_c_end_ss.sum())
            t_cycle = i_t_first_cycle * self._dt
            # Wash desorption.
            if self.wash_desorption and self.wash_recycle:
                c_wash_desorbed = self._sim_c_wash_desorption(
                    f_first_load[:i_t_first_cycle],
                    c_first_load[:, :i_t_first_cycle]
                    - bt_first_load[:, :i_t_first_cycle])
            else:
                c_wash_desorbed = None
            bt_first_load_out, bt_first_wash_out = \
                self._sim_c_recycle_propagation(
                    f_first_load[:i_t_first_cycle],
                    bt_first_load[:, :i_t_first_cycle],
                    c_wash_desorbed)
            if self.load_recycle:
                if not self.load_c_end_estimate_with_iterative_solver:
                    self.log.w(f"Estimating cycle duration:"
                               f" Assuming sharp breakthrough profile.")
                i_load_recycle_start = self._wash_recycle_i_duration \
                    if self.wash_recycle else 0
                m_load_recycle = \
                    bt_first_load_out[
                        :,
                        i_load_recycle_start:i_t_first_cycle
                    ].sum() * self._load_f * self._dt
                _t_diff = m_load_recycle / self._load_f / load_c_ss.sum()
                t_cycle -= _t_diff
                self._load_recycle_m_ss = m_load_recycle
                self.log.i_data(self._log_tree,
                                'm_load_recycle_ss',
                                m_load_recycle)
                self.log.i_data(self._log_tree,
                                'shorten_cycle_t_due_to_bt_recycle',
                                _t_diff)
            if self.wash_recycle:
                if not self.load_c_end_estimate_with_iterative_solver:
                    self.log.w(f"Estimating cycle duration:"
                               f" Assuming sharp breakthrough profile.")
                m_wash_recycle = bt_first_wash_out[
                                     :,
                                     :self._wash_recycle_i_duration
                                 ].sum() * self._wash_f * self._dt
                _t_diff = m_wash_recycle / self._load_f / load_c_ss.sum()
                t_cycle -= _t_diff
                self._wash_recycle_m_ss = m_wash_recycle
                self.log.i_data(self._log_tree,
                                'm_wash_recycle_ss',
                                m_wash_recycle)
                self.log.i_data(self._log_tree,
                                'shorten_cycle_t_due_to_wash_recycle',
                                _t_diff)

            if self.load_c_end_estimate_with_iterative_solver \
                    and (self.wash_recycle or self.load_recycle):
                c_load_fist_cycle = load_c_ss * _np.ones([len(binding_species),
                                                          i_t_first_cycle * 2])

                def sim_cycle(f_load: _np.ndarray,
                              c_load: _np.ndarray,
                              i_prev_cycle: int) -> _typing.Tuple[_np.ndarray,
                                                                  _np.ndarray,
                                                                  int]:
                    """Simulates load-wash cycle. Calc load duration.

                    Load duration is determined based on breakthrough
                    criteria.

                    Parameters
                    ----------
                    f_load
                        Load flow rate profile.
                    c_load
                        Load conc profile.
                    i_prev_cycle
                        Previous cycle duration in time steps.

                    Returns
                    -------
                    f_load_next_cycle
                        Load and wash breakthrough flow rate profile.
                    c_load_next_cycle
                        Load and wash breakthrough conc profile.
                    i_cycle
                        Current cycle duration in time steps.

                    """
                    # Load.
                    bt_load: _np.ndarray = \
                        c_load - self.load_bt.calc_c_bound(f_load, c_load)
                    # Propagate breakthrough.
                    bt_load_out, _ = self._sim_c_recycle_propagation(
                        f_load,
                        bt_load,
                        None)
                    # 'Stop' load at specified breakthrough criteria.
                    # noinspection PyTypeChecker
                    i_cycle_duration = _utils.vectors.true_start(
                        bt_load_out.sum(0) >= load_c_end_ss.sum())
                    # Cut load at specified time.
                    bt_load = bt_load[:, :i_cycle_duration]
                    # Wash desorption.
                    if self.wash_desorption and self.wash_recycle:
                        c_first_wash_desorbed = self._sim_c_wash_desorption(
                            f_load[:i_cycle_duration],
                            c_load[:, :i_cycle_duration]
                            - bt_load[:, :i_cycle_duration])
                    else:
                        c_first_wash_desorbed = None
                    # Propagate load and wash leftovers.
                    bt_load_out, bt_wash_out = self._sim_c_recycle_propagation(
                        f_load[:i_cycle_duration],
                        bt_load,
                        c_first_wash_desorbed)
                    # Construct load for next cycle.
                    # Recycle load.
                    if self.load_recycle:
                        rec_load = bt_load_out[:,
                                               i_prev_cycle:i_cycle_duration]
                    else:
                        rec_load = _np.zeros_like(
                            bt_load_out[:, i_prev_cycle:i_cycle_duration])
                    # Next load profiles.
                    c_next_load = _np.concatenate((rec_load,
                                                   c_load_fist_cycle),
                                                  axis=1)
                    f_next_load = self._load_f * _np.ones(c_next_load.shape[1])
                    wash_recycle_i_duration = self._wash_recycle_i_duration \
                        if self.wash_recycle else 0
                    # Log.
                    m_load_recycle_ss = \
                        bt_first_load_out[
                            :,
                            wash_recycle_i_duration:i_t_first_cycle
                        ].sum() * self._load_f * self._dt
                    self._load_recycle_m_ss = m_load_recycle_ss
                    self.log.i_data(self._log_tree,
                                    'm_load_recycle_ss',
                                    m_load_recycle_ss)
                    # Recycle wash.
                    if self.wash_recycle:
                        c_next_load[:, :self._wash_recycle_i_duration] = \
                            bt_wash_out[:, :self._wash_recycle_i_duration]
                        f_next_load[:self._wash_recycle_i_duration] = \
                            self._wash_f
                        m_wash_recycle_ss = \
                            bt_wash_out[:,
                                        :self._wash_recycle_i_duration
                                        ].sum() * self._wash_f * self._dt
                        self._wash_recycle_m_ss = m_wash_recycle_ss
                        self.log.i_data(self._log_tree,
                                        'm_wash_recycle_ss',
                                        m_wash_recycle_ss)
                    # Return next load and cycle duration.
                    return f_next_load, c_next_load, \
                        i_cycle_duration - i_prev_cycle

                f_load_cycle = \
                    self._load_f * _np.ones(c_load_fist_cycle.shape[1])
                c_load_cycle = c_load_fist_cycle
                i_t_cycle_prev = i_t_first_cycle
                i_t_cycle_estimate = 0
                # Loop until cycle duration converges.
                for i in range(
                        self.load_c_end_estimate_with_iter_solver_max_iter):
                    if abs(i_t_cycle_prev - i_t_cycle_estimate) <= 1:
                        self.log.i_data(self._log_tree,
                                        "t_cycle_optimization_loop_iter",
                                        i)
                        break
                    i_t_cycle_prev = i_t_cycle_estimate
                    f_load_cycle, c_load_cycle, i_t_cycle_estimate = \
                        sim_cycle(f_load_cycle, c_load_cycle, i_t_cycle_prev)
                    # print([i, i_t_cycle_prev, i_t_cycle_estimate])
                if abs(i_t_cycle_prev - i_t_cycle_estimate) > 1:
                    self.log.w("Cycle duration estimator did not converge.")
                t_cycle = i_t_cycle_estimate * self._dt
            elif self.load_c_end_estimate_with_iterative_solver:
                self.log.i(f"No need to use iterative solver in case of"
                           f" no recycling of load and/or wash.")
        self._cycle_t = t_cycle
        self.log.i_data(self._log_tree, 'cycle_t', t_cycle)

    # noinspection DuplicatedCode
    def _calc_first_cycle_extension_t(self):
        """Calc extension of first load.

        First load step might be extended for processes with load and/or
        wash recycle in order to get faster into steady-state regime.

        """
        if not self.load_recycle and not self.wash_recycle:
            self.log.w(f"Estimation of first cycle extension requested"
                       f" on a process without load recycle.")
            self._first_cycle_extension_t = 0
            return
        elif not self.load_extend_first_cycle:
            self.log.w(f"Estimation of first cycle extension requested"
                       f" on a process without extended first cycle.")
            self._first_cycle_extension_t = 0
            return
        elif self.load_extend_first_cycle_t > 0:
            self._first_cycle_extension_t = self.load_extend_first_cycle_t
            return
        elif self.load_extend_first_cycle_cv >= 0:
            assert self._cv > 0, "CV should be defined by now."
            assert self._load_f > 0, "Load flow rate should be defined by now."
            self._first_cycle_extension_t = \
                self.load_extend_first_cycle_cv * self._cv / self._load_f
        elif self.load_cv > 0:
            raise NotImplementedError(
                f"Estimation of first cycle extension is only supported"
                f" if the cycle length is defined by breakthrough cutoff"
                f" criteria. This is due to the fact that if all the"
                f" breakthrough material gets recycles,"
                f" there is no single steady-state.")
        else:
            binding_species = [i for i in range(self._n_species)
                               if i not in self.non_binding_species]
            load_c_ss = self._estimate_steady_state_mean_c(binding_species)
            #  simulate first cycle at constant load concentration
            f_first_load = self._load_f * _np.ones(self._t.size)
            c_first_load = load_c_ss * _np.ones([len(binding_species),
                                                 self._t.size])
            bt_first_load: _np.ndarray = \
                load_c_ss - self.load_bt.calc_c_bound(f_first_load,
                                                      c_first_load)

            # propagate breakthrough
            bt_first_load_out, _ = \
                self._sim_c_recycle_propagation(f_first_load,
                                                bt_first_load,
                                                None)
            load_c_end_ss = self._get_load_bt_cycle_switch_criteria(load_c_ss)
            # noinspection PyTypeChecker
            i_t_first_cycle = _utils.vectors.true_start(
                bt_first_load_out.sum(0) >= load_c_end_ss.sum())
            dm = 0
            if self.load_recycle:
                assert hasattr(self, "_load_recycle_m_ss"), \
                    f"Function `_calc_cycle_t()` should already be called."
                dm += self._load_recycle_m_ss
            if self.wash_recycle:
                assert hasattr(self, "_wash_recycle_m_ss"), \
                    f"Function `_calc_cycle_t()` should already be called."
                dm += self._wash_recycle_m_ss
            di = 0
            if dm > 0:
                m_ext_bt = _np.cumsum(
                    bt_first_load_out.sum(0)[i_t_first_cycle:]
                ) * self._load_f * self._dt
                di += _utils.vectors.true_start(m_ext_bt >= dm)
            self._first_cycle_extension_t = di * self._dt

    def _calc_cycle_start_i_list(self):
        """Calculate load switch positions in form of time steps."""
        assert self._cycle_t > 0, \
            f"Cycle length must have been determined" \
            f" (by `_calc_cycle_t()`) by now"
        flow_i_start, flow_i_end = \
            _utils.vectors.true_start_and_end(self._f > 0)
        if self.load_extend_first_cycle:
            assert self._first_cycle_extension_t >= 0, \
                f"Prolong of first load cycle is set to `True`," \
                f" but the length is undefined."
            if self._first_cycle_extension_t == 0:
                self.log.w(f"Prolong of first load cycle is set to `True`,"
                           f" but the length of the extension is 0.")
            load_extend_first_cycle_t = self._first_cycle_extension_t
            self.log.i_data(self._log_tree,
                            "load_extend_first_cycle_t",
                            load_extend_first_cycle_t)
        else:
            load_extend_first_cycle_t = 0
        cycle_start_t_list = _np.arange(
            self._t[flow_i_start] + load_extend_first_cycle_t,
            self._t[flow_i_end - 1],
            self._cycle_t
        )
        cycle_start_t_list[0] = self._t[flow_i_start]
        self._cycle_start_i_list = _np.rint(
            cycle_start_t_list / self._dt).astype(_np.int32)
        self.log.i_data(self._log_tree,
                        "cycle_start_t_list",
                        cycle_start_t_list)

    def _prepare_simulation(self):
        """Prepare everything before cycle-by-cycle simulation."""
        self._assert_non_binding_species()
        self._calc_load_f()
        self._calc_cv()  # might depend on load_f
        self._report_column_dimensions()  # optional
        # Equilibration.
        self._calc_equilibration_t()
        # Wash.
        self._calc_wash_t_and_f()
        # Elution.
        self._calc_elution_t_and_f()
        self._calc_elution_peak_t()
        self._update_elution_peak_pdf()
        self._calc_elution_peak_cut_i_start_and_i_end()
        self._calc_elution_peak_mask()
        # Regeneration.
        self._calc_regeneration_t()
        # Prepare for estimation of cycle length.
        self._update_load_btc()
        if self.load_recycle:
            self._update_load_recycle_pdf(self._wash_f)
            if self.wash_recycle:
                self._calc_load_recycle_wash_i()
        # Cycle time.
        self._calc_cycle_t()
        if self.load_extend_first_cycle:
            self._calc_first_cycle_extension_t()
        # Cycle start positions == column load switch time points.
        self._calc_cycle_start_i_list()
        # Make sure cycle duration is long enough.
        _t_cycle_except_load = self._equilibration_t + self._wash_t \
            + self._elution_t + self._regeneration_t
        if self._cycle_t < _t_cycle_except_load:
            self.log.e(f"Load step ({self._cycle_t}) should not be shorter"
                       f" than eq_t + wash_t + elution_t + regeneration_t"
                       f" ({_t_cycle_except_load: .6})!")

    def _sim_c_load_binding(self,
                            f_load: _np.ndarray,
                            c_load: _np.ndarray
                            ) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Determine what part of load binds.

        Load in this context might also contain wash and load recycle
        from previous steps.

        Parameters
        ----------
        f_load
            Load flow rate profile.
        c_load
            Load concentration profile.

        Returns
        -------
        c_bound
            Conc profile of bound material.
        c_unbound
            Conc profile of unbound material = `c_load` - `c_bound`.

        """
        assert f_load.size == c_load.shape[1], \
            "f_load and c_load must have the same length"
        assert c_load.shape[0] == \
            self._n_species - len(self.non_binding_species), \
            "c_load must contain all binding species"
        c_bound = self.load_bt.calc_c_bound(f_load, c_load)
        # Returns bound and unbound part.
        return c_bound, c_load - c_bound

    def _sim_c_wash_desorption(self,
                               f_load: _np.ndarray,
                               c_bound: _np.ndarray) -> _np.ndarray:
        """Get conc profile of desorbed material during wash step.

        The step has no default logic.
        Thus it raises `NotImplementedError` if called.

        Parameters
        ----------
        f_load
            Flow rate profile during 'effective load' step.

            The step includes wash recycle, load recycle and load step
            as a column sees it in a single cycle.
        c_bound
            Conc profile of captured material.

        Returns
        -------
        ndarray
            Conc profile of desorbed material during wash step.

        Raises
        ------
        NotImplementedError
            This method has no default implementation. Thus it being
            called it will raise the error.

        """
        # Not implemented in core this class, as there is
        # no consensus on typical dynamics and the way to describe it.
        raise NotImplementedError("Function not implemented in this class")

    def _sim_c_recycle_propagation(
            self,
            f_unbound: _np.ndarray,
            c_unbound: _np.ndarray,
            c_wash_desorbed: _typing.Optional[_np.ndarray]
    ) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Propagate unbound and desorbed material through the column.

        Unbound (breakthrough during load) and desorbed (during wash)
        sections might have a different flow rates as they come from
        different steps - load and wash.

        Parameters
        ----------
        f_unbound
            Flow rate profile during 'total load' step for a cycle.

            The step includes wash recycle, load recycle and load step.
        c_unbound
            Conc profile of overloaded material during load step
            (plus previous wash and load recycle).
        c_wash_desorbed
            Conc profile of desorbed material during wash step.

        Returns
        -------
        c_unbound_propagated
            Propagated conc profile of overloaded material.
        c_wash_desorbed_propagated
            Propagated conc profile of desorbed material.

        """
        assert hasattr(self, "_wash_f") and self._wash_f > 0
        assert hasattr(self, "_wash_t") and self._wash_t > 0
        assert self.load_recycle_pdf is not None
        assert c_unbound.shape[0] == \
            self._n_species - len(self.non_binding_species)
        assert c_unbound.shape[1] == f_unbound.size
        if c_wash_desorbed is None or c_wash_desorbed.size == 0:
            c_wash_desorbed = _np.zeros([
                self._n_species - len(self.non_binding_species),
                int(round(self._wash_t / self._dt))])
        else:
            assert c_wash_desorbed.shape[0] == \
                   self._n_species - len(self.non_binding_species)
            assert c_wash_desorbed.shape[1] == \
                int(round(self._wash_t / self._dt))
        # Combine on volumetric scale.
        v_load = self._dt * f_unbound.cumsum()
        v_wash = v_load[-1] + \
            self._dt * _np.arange(1, c_wash_desorbed.shape[1] + 1) \
            * self._wash_f
        min_flow = min(f_unbound.min(), self._wash_f)
        dv = min_flow * self._dt
        v = _np.arange(dv,
                       (v_wash[-1] if v_wash.size > 0 else v_load[-1]) + dv,
                       dv)
        c_v_combined = _interp.interp1d(
            _np.concatenate((v_load, v_wash), axis=0),
            _np.concatenate((c_unbound, c_wash_desorbed), axis=1),
            fill_value="extrapolate"
        )(v)
        c_v_combined[c_v_combined < 0] = 0
        # Simulate traveling of leftover material through the column.
        self._update_load_recycle_pdf(min_flow)
        c_v_combined_propagated = _utils.convolution.time_conv(
            self._dt, c_v_combined, self._p_load_recycle_pdf)
        # Split back on time scale.
        c_combined_propagated = _interp.interp1d(
            v,
            c_v_combined_propagated,
            fill_value="extrapolate"
        )(_np.concatenate((v_load, v_wash), axis=0))
        c_combined_propagated[c_combined_propagated < 0] = 0
        c_unbound_propagated = c_combined_propagated[:, :v_load.size]
        c_wash_desorbed_propagated = c_combined_propagated[:, v_load.size:]
        return c_unbound_propagated, c_wash_desorbed_propagated

    def _sim_c_elution_desorption(self,
                                  m_bound: _np.ndarray
                                  ) -> _typing.Tuple[_np.ndarray,
                                                     _np.ndarray]:
        """Simulate elution step.

        Parameters
        ----------
        m_bound
            Vector with amount of product being bound to the column.

            `m_bound.size == n_species`

        Returns
        -------
        c_elution
            Outlet concentration profile during the elution.
        b_elution_peak
            Boolean vector. Peak is collected where the value is `True`.

        """
        assert self._elution_f > 0
        assert self._elution_t > 0
        i_elution_duration = int(round(self._elution_t / self._dt))
        # Multiply elution peak with the amount of captured product.
        c_elution = \
            self._p_elution_peak[_np.newaxis, :i_elution_duration] * \
            m_bound[:, _np.newaxis] / self._elution_f
        # Pad with zeros to cover the entire elution step duration.
        if c_elution.shape[1] < i_elution_duration:
            c_elution = _np.pad(c_elution,
                                ((0, 0),
                                 (0, i_elution_duration - c_elution.shape[1])),
                                mode="constant")
        # Boolean mask - `True` where peak is being collected.
        b_elution_peak = self._elution_peak_mask
        return c_elution, b_elution_peak

    def _sim_c_elution_buffer(self, n_time_steps: int) -> _np.ndarray:
        """Get elution buffer composition at the outlet of the column.

        By default the buffer composition is constant throughout the
        elution step.

        Feel free to override this function if you want to simulate
        linear gradient or if the transient phenomena at the beginning
        of peak cut needs to be considered.

        Parameters
        ----------
        n_time_steps
            Duration of elution step in number of time steps.

        Returns
        -------
        ndarray
            Buffer concentration profile at the outlet of the column
            during the elution step.

        """
        # Elution buffer composition.
        elution_buffer_composition = \
            self.elution_buffer_c.reshape(self.elution_buffer_c.size, 1)
        assert elution_buffer_composition.size == 0 \
            or elution_buffer_composition.size == self._n_species, \
            f"Elution buffer composition must be either empty or have" \
            f" a concentration value for each specie."
        assert _np.all(elution_buffer_composition >= 0), \
            "Concentration values in elution buffer must be >= 0"
        if elution_buffer_composition.size == 0:
            elution_buffer_composition = _np.zeros([self._n_species, 1])
        self.log.i_data(self._log_tree,
                        "elution_buffer_composition",
                        elution_buffer_composition)
        # Constant profile.
        c_elution_buffer = elution_buffer_composition \
            * _np.ones_like(self._t[:n_time_steps])
        return c_elution_buffer

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _sim_c_regeneration(self,
                            m_bound: _np.ndarray
                            ) -> _typing.Optional[_np.ndarray]:
        """Simulate regeneration step.

        Parameters
        ----------
        m_bound
            Vector with amount of product being bound to the column at
            the beginning of the regeneration step.

            `m_bound.size == n_species`.

        Returns
        -------
        Optional[ndarray]
            Outlet concentration profile during regeneration step.
            E.g. regeneration peak.

        """
        # No default implementation.
        c_regeneration = None
        return c_regeneration

    def _sim_c_out_cycle(self,
                         f_load: _np.ndarray,
                         c_load: _np.ndarray
                         ) -> _typing.Tuple[_typing.Optional[_np.ndarray],
                                            _typing.Optional[_np.ndarray],
                                            _np.ndarray,
                                            _np.ndarray,
                                            _typing.Optional[_np.ndarray]]:
        """Simulates load-wash-elution-regeneration steps.

        Regeneration is optional.

        This function can be replaced in case user wants to use some
        other variation of bind-elution dynamics.

        Elution peak cut is applied in this function.
        Elution peak shape must be defined by now.

        Return profiles that are `None` are considered being zero.

        Parameters
        ----------
        f_load
            Inlet (recycle + load) flow rate profile for a cycle.

            The flow rate might be different during wash recycle.
        c_load
            Inlet (recycle + load) concentration profile.

        Returns
        -------
        c_load
            Conc profile at the outlet of the column during load.
        c_wash
            Conc profile at the outlet of the column during wash.
        c_elution
            Conc profile at the outlet of the column during elution.
        b_elution
            Boolean mask for elution step. `True` where peak is being
            collected.
        c_regeneration
            Conc profile at the outlet of the column during
            regeneration.

        """
        assert self._load_f > 0
        assert self._wash_f > 0
        assert self._wash_t > 0
        assert self._elution_f > 0
        assert self._elution_t > 0
        assert self._load_f > 0
        assert self._cv > 0
        # Evaluate binding.
        c_bound, c_unbound = self._sim_c_load_binding(f_load, c_load)
        # Log.
        m_load = (c_load * f_load[_np.newaxis, :]).sum(1) * self._dt
        m_bound = (c_bound * f_load[_np.newaxis, :]).sum(1) * self._dt
        self.log.i_data(self._cycle_tree,
                        "column_utilization",
                        m_bound / self._cv / self.load_bt.get_total_bc())
        self.log.i_data(self._cycle_tree, "m_load", m_load)
        self.log.i_data(self._cycle_tree, "m_bound", m_bound)
        self.log.i_data(self._cycle_tree, "m_unbound", m_load - m_bound)
        self.log.d_data(self._cycle_tree, "f_load", f_load)
        self.log.d_data(self._cycle_tree, "c_load", c_load)
        self.log.d_data(self._cycle_tree, "c_bound", c_bound)
        self.log.d_data(self._cycle_tree, "c_unbound", c_unbound)
        # Evaluate desorption during wash.
        c_wash_desorbed = None
        if self.wash_desorption:
            c_wash_desorbed = self._sim_c_wash_desorption(f_load, c_bound)
            if c_wash_desorbed.size > 0:
                # Subtract desorbed material from bound material.
                m_bound -= c_wash_desorbed.sum(1)
            # Log.
            self.log.i_data(self._cycle_tree,
                            "m_wash_desorbed",
                            c_wash_desorbed.sum(1) * self._wash_f * self._dt)
            self.log.d_data(self._cycle_tree,
                            "c_wash_desorbed",
                            c_wash_desorbed)
        # Propagate unbound and desorbed material throughout the column.
        c_out_load = c_unbound
        c_out_wash = c_wash_desorbed
        if self.load_recycle or self.wash_recycle:
            c_out_load, c_out_wash = \
                self._sim_c_recycle_propagation(f_load,
                                                c_unbound,
                                                c_wash_desorbed)
        # Get elution peak.
        c_out_elution, elution_peak_mask = \
            self._sim_c_elution_desorption(m_bound)
        # Log.
        m_elution_peak = (c_out_elution * elution_peak_mask[_np.newaxis, :]
                          ).sum(1) * self._elution_f * self._dt
        m_elution = c_out_elution.sum(1) * self._elution_f * self._dt
        self.log.i_data(self._cycle_tree,
                        "m_elution_peak", m_elution_peak)
        self.log.i_data(self._cycle_tree,
                        "m_elution", m_elution)
        self.log.i_data(self._cycle_tree,
                        "m_elution_peak_cut_loss", m_elution - m_elution_peak)
        # Get regeneration peak.
        c_out_regeneration = self._sim_c_regeneration(
            m_bound - c_out_elution.sum(1) * self._elution_f * self._dt)

        return c_out_load, c_out_wash, c_out_elution, \
            elution_peak_mask, c_out_regeneration

    def _calculate(self):
        # Pre calculate parameters and repetitive profiles.
        self._prepare_simulation()
        # Assert proper list of binding species.
        binding_species = [i for i in range(self._n_species)
                           if i not in self.non_binding_species]
        assert len(binding_species) > 0
        # Copy inlet vectors.
        c_in_load = self._c[binding_species].copy()
        f_in_load = self._f.copy()
        f_in_i_end = min(_utils.vectors.true_end(f_in_load > 0), self._t.size)
        c_in_load[:, f_in_i_end:] = 0
        # Clear for results.
        self._c[:] = 0
        self._f[:] = 0
        # Prepare logger.
        log_data_cycles = list()
        self.log.set_branch(self._log_tree, "cycles", log_data_cycles)
        # Variable to store wash recycle to.
        previous_c_bt_wash: _typing.Optional[_np.ndarray] = None
        # Loop across cycles.
        for i in range(self._cycle_start_i_list.size):
            # Load-wash-elution-regeneration-equilibration steps for a column.
            # Load step starts at `self._cycle_start_i_list[i]`.

            # Prepare logger for this cycle.
            self._cycle_tree = dict()
            log_data_cycles.append(self._cycle_tree)
            # Load start and end time as the column sees it.
            if i > 0 and self.load_recycle:
                # Column sees leftovers from previous load during recycling.
                cycle_load_i_start = self._cycle_start_i_list[i - 1]
            else:
                cycle_load_i_start = self._cycle_start_i_list[i]
            # Calc cycle end (either next cycle or end or simulation time).
            if i + 1 < self._cycle_start_i_list.size:
                cycle_load_i_end = self._cycle_start_i_list[i + 1]
            else:
                cycle_load_i_end = f_in_i_end - 1
            # Log results.
            self.log.i_data(self._cycle_tree,
                            "i_cycle_load_start",
                            cycle_load_i_start)
            self.log.i_data(self._cycle_tree,
                            "i_cycle_load_step_start",
                            self._cycle_start_i_list[i])
            self.log.i_data(self._cycle_tree,
                            "i_cycle_load_end",
                            cycle_load_i_end)
            # Calc profiles at column outlet.
            c_out_load, c_out_wash, c_out_elution, \
                b_out_elution, c_out_regeneration = self._sim_c_out_cycle(
                    f_in_load[cycle_load_i_start:cycle_load_i_end],
                    c_in_load[:, cycle_load_i_start:cycle_load_i_end]
                )
            self.log.d_data(self._cycle_tree,
                            "c_out_load", c_out_load)
            self.log.d_data(self._cycle_tree,
                            "c_out_wash", c_out_wash)
            self.log.d_data(self._cycle_tree,
                            "c_out_elution", c_out_elution)
            self.log.d_data(self._cycle_tree,
                            "b_out_elution", b_out_elution)
            self.log.d_data(self._cycle_tree,
                            "c_out_regeneration", c_out_regeneration)
            # Load recycle.
            if self.load_recycle:
                # Recycle load during the load step.
                i_load_start_rel = self._cycle_start_i_list[i] \
                                   - cycle_load_i_start
                c_load_recycle = c_out_load[:, i_load_start_rel:]
                c_in_load[:, self._cycle_start_i_list[i]:cycle_load_i_end] = \
                    c_load_recycle
                self.log.i_data(self._cycle_tree, "m_load_recycle",
                                c_load_recycle.sum(1)
                                * self._load_f * self._dt)
                self.log.d_data(self._cycle_tree, "c_load_recycle",
                                c_load_recycle)
                # Losses during load == bt through 2nd column.
                c_loss_bt_2nd_column = c_out_load[:, i_load_start_rel]
                self.log.i_data(self._cycle_tree, "m_loss_bt_2nd_column",
                                c_loss_bt_2nd_column.sum()
                                * self._dt * self._load_f)
                self.log.d_data(self._cycle_tree, "c_loss_bt_2nd_column",
                                c_loss_bt_2nd_column)
            else:
                # report losses during load
                m_loss_load = c_out_load.sum() * self._dt * self._load_f
                self.log.i_data(self._cycle_tree, "m_loss_load", m_loss_load)
            # Wash recycle.
            if self.wash_recycle:
                if previous_c_bt_wash is not None \
                        and previous_c_bt_wash.size > 0:
                    # Clip wash recycle duration if needed.
                    i_wash_duration = min(
                        self._wash_recycle_i_duration,
                        self._t.size - self._cycle_start_i_list[i])
                    # Log losses due to discarding load bt during wash recycle.
                    s = c_in_load[
                        :,
                        self._cycle_start_i_list[i]:self._cycle_start_i_list[i]
                        + i_wash_duration]
                    self.log.i_data(self._cycle_tree,
                                    "m_loss_load_bt_during_wash_recycle",
                                    s.sum() * self._dt * self._load_f)
                    self.log.d_data(self._cycle_tree,
                                    "c_lost_load_during_wash_recycle", s)
                    self.log.d_data(self._cycle_tree, "c_wash_recycle",
                                    previous_c_bt_wash[:, :i_wash_duration])
                    self.log.i_data(
                        self._cycle_tree, "m_wash_recycle",
                        previous_c_bt_wash[:, :i_wash_duration].sum(1)
                        * self._dt * self._wash_f)
                    # Apply previous wash recycle onto the inlet profile.
                    s[:] = previous_c_bt_wash[:, :i_wash_duration]
                    f_in_load[self._cycle_start_i_list[i]:
                              self._cycle_start_i_list[i]
                              + i_wash_duration] = self._wash_f
                # Save wash from this cycle to be used during the next cycle.
                previous_c_bt_wash = c_out_wash
            else:
                # Report losses during wash.
                if c_out_wash is None:
                    c_out_wash = _np.zeros(
                        [len(binding_species),
                         int(round(self._wash_t / self._dt))])
                m_loss_wash = c_out_wash.sum() * self._dt * self._load_f
                self.log.i_data(self._cycle_tree, "m_loss_wash", m_loss_wash)
            # Elution.
            [i_el_rel_start, i_el_rel_end] = \
                _utils.vectors.true_start_and_end(b_out_elution)
            i_el_start = min(
                self._t.size,
                cycle_load_i_end + c_out_wash.shape[1] + i_el_rel_start)
            i_el_end = min(
                self._t.size,
                cycle_load_i_end + c_out_wash.shape[1] + i_el_rel_end)
            i_el_rel_end = i_el_rel_start + i_el_end - i_el_start
            # Log.
            self.log.i_data(self._cycle_tree, "i_elution_start", i_el_start)
            self.log.i_data(self._cycle_tree, "i_elution_end", i_el_end)
            # Write to global outlet.
            self._f[i_el_start:i_el_end] = self._elution_f
            self._c[binding_species, i_el_start:i_el_end] = \
                c_out_elution[:, i_el_rel_start:i_el_rel_end]


class ACC(AlternatingChromatography):
    """Alternating column chromatography without recycling.

    Alternating load-bind-elution twin-column chromatography without
    recycling of overloaded or washed out material.

    This class offers no dynamics for desorption during wash step.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    load_bt
        Load breakthrough logic.
    peak_shape_pdf
        Elution peak shape.
    gui_title
        Readable title for GUI. Default = "ACC".

    Notes
    -----
    For list of attributes refer to :class:`AlternatingChromatography`.

    See Also
    --------
    :class:`AlternatingChromatography`

    Examples
    --------

    >>> dt = 0.5  # min
    >>> t = _np.arange(0, 24.1 * 60, dt)
    >>> load_bt = _bt_load.ConstantPatternSolution(dt, dbc_100=50, k=0.12)
    >>> peak_shape_pdf = _pdf.ExpModGaussianFixedRelativeWidth(t, 0.15, 0.3)
    >>> acc_pro_a = ACC(
    ...     t,
    ...     load_bt=load_bt,
    ...     peak_shape_pdf=peak_shape_pdf,
    ...     uo_id="pro_a_acc",
    ...     gui_title="ProteinA ACC",
    ... )
    >>> acc_pro_a.cv = 100  # mL
    >>> # Equilibration step.
    >>> acc_pro_a.equilibration_cv = 1.5
    >>> # Equilibration flow rate is same as load flow rate.
    >>> acc_pro_a.equilibration_f_rel = 1
    >>> # Load 10 CVs.
    >>> acc_pro_a.load_cv = 20
    >>> # Define wash step.
    >>> acc_pro_a.wash_cv = 5
    >>> # Elution step.
    >>> acc_pro_a.elution_cv = 3
    >>> # 1st momentum of elution peak from data from above.
    >>> acc_pro_a.elution_peak_position_cv = 1.2
    >>> acc_pro_a.elution_peak_cut_start_c_rel_to_peak_max = 0.05
    >>> acc_pro_a.elution_peak_cut_end_c_rel_to_peak_max = 0.05
    >>> # Regeneration step.
    >>> acc_pro_a.regeneration_cv = 1.5
    >>> # Inlet flow rate profile.
    >>> f_in = _np.ones_like(t) * 15  # mL/min
    >>> c_in = _np.ones([1, t.size]) * 2.5  # mg/mL
    >>> # Simulate ACC.
    >>> f_out, c_out = acc_pro_a.evaluate(f_in, c_in)

    """

    def __init__(self,
                 t: _np.ndarray,
                 uo_id: str,
                 load_bt: _core.ChromatographyLoadBreakthrough,
                 peak_shape_pdf: _core.PDF,
                 gui_title: str = "ACC"):
        super().__init__(t, uo_id, load_bt, peak_shape_pdf, gui_title)

    def _sim_c_wash_desorption(self,
                               f_load: _np.ndarray,
                               c_bound: _np.ndarray) -> _np.ndarray:
        """Desorbed material during wash step is not supported by ACC.

        Raises
        ------
        NotImplementedError
            Raises exception when function if called.

        """
        raise NotImplementedError("Function not implemented in this class.")


class PCC(AlternatingChromatography):
    """Alternating column chromatography with recycling of load.

    Alternating load-bind-elution twin-column chromatography with
    optional recycling of overloaded or washed out material.

    This class offers no dynamics for desorption during wash step.

    PCC uses :attr:`load_bt` to determine what parts of the load (and
    recycled material) bind to the column. The unbound (not captured)
    part is propagated through the column by :attr:`load_recycle_pdf`.

    Void volume for :attr:`load_recycle_pdf` is defined as
    :attr:`column_porosity_retentate` * `column volume`.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    load_bt
        Load breakthrough logic.
    load_recycle_pdf
        Propagation of load breakthrough and/or washed out material
        through the column.
    column_porosity_retentate
        Porosity of the column for binding species (protein).
    peak_shape_pdf
        Elution peak shape.
    gui_title
        Readable title for GUI. Default = "PCC".

    Notes
    -----
    For list of additional attributes refer to
    :class:`AlternatingChromatography`.

    See Also
    --------
    :class:`AlternatingChromatography`

    Examples
    --------
    >>> dt = 0.5  # min
    >>> t = _np.arange(0, 24.1 * 60, dt)
    >>> load_bt = _bt_load.ConstantPatternSolution(dt, dbc_100=50, k=0.12)
    >>> peak_shape_pdf = _pdf.ExpModGaussianFixedRelativeWidth(t, 0.15, 0.3)
    >>> load_recycle_pdf = _pdf.GaussianFixedDispersion(t, 2 * 2 / 30)
    >>> pcc_pro_a = PCC(
    ...     t,
    ...     load_bt=load_bt,
    ...     peak_shape_pdf=peak_shape_pdf,
    ...     load_recycle_pdf=load_recycle_pdf,
    ...     # Porosity of the column for protein.
    ...     column_porosity_retentate=0.64,
    ...     uo_id="pro_a_pcc",
    ...     gui_title="ProteinA PCC",
    ... )
    >>> pcc_pro_a.cv = 100  # mL
    >>> # Equilibration step.
    >>> pcc_pro_a.equilibration_cv = 1.5
    >>> # Equilibration flow rate is same as load flow rate.
    >>> pcc_pro_a.equilibration_f_rel = 1
    >>> # Load until 70 % breakthrough.
    >>> pcc_pro_a.load_c_end_relative_ss = 0.7
    >>> # Automatically prolong first cycle to faster achieve a steady-state.
    >>> pcc_pro_a.load_extend_first_cycle = True
    >>> # Define wash step.
    >>> # There is no desorption during wash step in this example.
    >>> pcc_pro_a.wash_cv = 5
    >>> pcc_pro_a.wash_recycle = True
    >>> pcc_pro_a.wash_recycle_duration_cv = 2
    >>> # Elution step.
    >>> pcc_pro_a.elution_cv = 3
    >>> # 1st momentum of elution peak from data from above.
    >>> pcc_pro_a.elution_peak_position_cv = 1.2
    >>> pcc_pro_a.elution_peak_cut_start_c_rel_to_peak_max = 0.05
    >>> pcc_pro_a.elution_peak_cut_end_c_rel_to_peak_max = 0.05
    >>> # Regeneration step.
    >>> pcc_pro_a.regeneration_cv = 1.5
    >>> # Inlet flow rate profile.
    >>> f_in = _np.ones_like(t) * 15  # mL/min
    >>> c_in = _np.ones([1, t.size]) * 2.5  # mg/mL
    >>> # Simulate ACC.
    >>> f_out, c_out = pcc_pro_a.evaluate(f_in, c_in) # doctest: +ELLIPSIS
    pro_a_pcc: Steady-state concentration is being estimated ...
    pro_a_pcc: Steady-state concentration is being estimated ...

    """

    def __init__(self,
                 t: _np.ndarray,
                 uo_id: str,
                 load_bt: _core.ChromatographyLoadBreakthrough,
                 load_recycle_pdf: _core.PDF,
                 column_porosity_retentate: float,
                 peak_shape_pdf: _core.PDF,
                 gui_title: str = "PCC"):
        super().__init__(t, uo_id, load_bt, peak_shape_pdf, gui_title)
        self.load_recycle = True
        """Recycle load breakthrough. Default = `True`."""
        self.wash_recycle = False
        """Recycle wash. Default = False."""
        self.column_porosity_retentate = column_porosity_retentate
        """Column porosity for binding species.
        
        See Also
        --------
        :class:`PCC`
        
        Examples
        --------
        `column_porosity_retentate` is a mean residence time of the
        product (protein) traveling through the column during
        non-binding conditions (in CVs).
        
        """
        self.load_recycle_pdf = load_recycle_pdf
        """PDF of wash and/or unbound load traveling through the column.
        
        See Also
        --------
        :class:`PCC`
        
        """

    def _sim_c_wash_desorption(self,
                               f_load: _np.ndarray,
                               c_bound: _np.ndarray) -> _np.ndarray:
        """Desorbed material during wash step is not supported by PCC.

        Raises
        ------
        NotImplementedError
            Raises exception when function if called.

        """
        raise NotImplementedError("Function not implemented in this class.")


class PCCWithWashDesorption(PCC):
    """Alternating column chromatography with recycling of load.

    Alternating load-bind-elution twin-column chromatography with
    optional recycling of overloaded or washed out material.

    The material desorption during wash step is defined by exponential
    half life time

    * :attr:`wash_desorption_tail_half_time_cv`

    and the amount of desorbable material which is defined by

    * :attr:`wash_desorption_desorbable_material_share` or
    * :attr:`wash_desorption_desorbable_above_dbc`.

    PCC uses :attr:`load_bt` to determine what parts of the load (and
    recycled material) bind to the column.

    The unbound (not captured) part and desorbed part are propagated
    through the column by :attr:`load_recycle_pdf`.

    Void volume for :attr:`load_recycle_pdf` is defined as
    :attr:`column_porosity_retentate` * `column volume`.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    uo_id
        Unique identifier.
    load_bt
        Load breakthrough logic.
    load_recycle_pdf
        Propagation of load breakthrough and/or washed out material
        through the column.
    column_porosity_retentate
        Porosity of the column for binding species (protein).
    peak_shape_pdf
        Elution peak shape.
    gui_title
        Readable title for GUI. Default = "PCCWithWashDesorption".

    Notes
    -----
    During wash step, weaker binding isoforms might be desorbed and
    recycled. In turn they are again desorbed and recycled during next
    cycle and so on; resulting in increasing amount of desorbed material
    during wash step (even in steady-state). This is not considered by
    this class. Furthermore, it is not a favorable case in terms of RTD
    as the weakly bound material propagates from column to column for
    many cycles.

    For list of additional attributes refer to
    :class:`PCC` and :class:`AlternatingChromatography`.

    See Also
    --------
    :class:`PCC`
    :class:`AlternatingChromatography`

    """

    def __init__(self,
                 t: _np.ndarray,
                 uo_id: str,
                 load_bt: _core.ChromatographyLoadBreakthrough,
                 load_recycle_pdf: _core.PDF,
                 column_porosity_retentate: float,
                 peak_shape_pdf: _core.PDF,
                 gui_title: str = "PCCWithWashDesorption"):
        super().__init__(t, uo_id, load_bt, load_recycle_pdf,
                         column_porosity_retentate, peak_shape_pdf, gui_title)

        self.load_recycle = True
        """Recycle load breakthrough. Default = `True`."""
        self.wash_recycle = True
        """Recycle wash. Default = `True`."""
        self.wash_desorption = True
        """Simulate desorption during wash step. Default = `True`."""
        self.wash_desorption_tail_half_time_cv = -1
        """Wash desorption rate.
        
        Required if :attr:`wash_desorption` is `True`.
        
        Wash desorption is simulated as exponential decay with half-life
        :attr:`wash_desorption_tail_half_time_cv`.
        
        """
        self.wash_desorption_desorbable_material_share = -1
        """Share of material that can be desorbed during wash step.
        
        Wash desorption is simulated as exponential decay. Only part of
        adsorbed material is subjected to that exponential decay. That
        part can be defined by:
        
        * :attr:`wash_desorption_desorbable_material_share` (this one)
          or
        * :attr:`wash_desorption_desorbable_above_dbc`.
        
        """
        self.wash_desorption_desorbable_above_dbc = -1
        """Share of material that can be desorbed during wash step.
        
        Share is defined as a share of material loaded onto the column
        that exceeds specified `wash_desorption_desorbable_above_dbc`
        binding capacity.
        
        Wash desorption is simulated as exponential decay. Only part of
        adsorbed material is subjected to that exponential decay. That
        part can be defined by:
        
        * :attr:`wash_desorption_desorbable_material_share` (this one)
          or
        * :attr:`wash_desorption_desorbable_above_dbc`.
        
        """

    def _sim_c_wash_desorption(self,
                               f_load: _np.ndarray,
                               c_bound: _np.ndarray) -> _np.ndarray:
        """Get conc profile of desorbed material during wash step.

        `self.wash_desorption_tail_half_time_cv` needs to be defined.

        One of `self.wash_desorption_desorbable_material_share` and
        `self.wash_desorption_desorbable_above_dbc` needs to be defined.

        Parameters
        ----------
        f_load
            Flow rate profile during 'effective load' step.

            The step includes wash recycle, load recycle and load step
            as a column sees it in a single cycle.
        c_bound
            Conc profile of captured material.

        Returns
        -------
        ndarray
            Conc profile of desorbed material during wash step.

        """
        assert self.wash_desorption_tail_half_time_cv > 0
        assert self._load_f > 0
        assert self._wash_f > 0
        assert self._wash_t > 0
        assert self._cv > 0
        assert self.wash_desorption_desorbable_material_share > 0 \
            or self.wash_desorption_desorbable_above_dbc > 0
        assert f_load.size == c_bound.shape[1]
        assert c_bound.shape[0] \
            == self._n_species - len(self.non_binding_species)
        m_bound = (c_bound * f_load[_np.newaxis, :]).sum(1)[:, _np.newaxis] \
            * self._dt
        # Calc share of desorbable material.
        k = -1
        if self.wash_desorption_desorbable_material_share > 0:
            k = self.wash_desorption_desorbable_material_share
        if self.wash_desorption_desorbable_above_dbc > 0:
            if k > 0:
                self.log.w(
                    f"Share of desorbable material defined twice!!"
                    f" Using `load_recycle_wash_desorbable_material_share`")
            else:
                k = max(0,
                        1 - self.wash_desorption_desorbable_above_dbc
                        * self._cv / m_bound.sum())
        assert 1 >= k >= 0, f"Share of desorbable material {k}" \
                            f" must be >= 0 and <= 1."
        i_wash_duration = int(round(self._wash_t / self._dt))
        # Generate exponential tail.
        exp_pdf = _pdf.TanksInSeries(self._t[:i_wash_duration],
                                     n_tanks=1,
                                     pdf_id=f"wash_desorption_exp_drop")
        exp_pdf.allow_open_end = True
        exp_pdf.trim_and_normalize = False
        tau = self.wash_desorption_tail_half_time_cv \
            * self._cv / self._wash_f / _np.log(2)
        exp_pdf.update_pdf(rt_mean=tau)
        p = exp_pdf.get_p()[_np.newaxis, :i_wash_duration]
        # Scale desorbed material conc due to differences in flow rate.
        c_desorbed = m_bound * k * p / self._wash_f
        # Pad with zeros if needed.
        c_desorbed = _np.pad(c_desorbed,
                             ((0, 0),
                              (0, i_wash_duration - c_desorbed.shape[1])),
                             mode="constant")
        # Log.
        self.log.d_data(self._cycle_tree if hasattr(self, "_cycle_tree")
                        else self._log_tree,
                        "p_desorbed",
                        p)
        return c_desorbed
