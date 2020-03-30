__all__ = ['AlternatingChromatography', 'ACC', 'PCC', 'PCCWithWashDesorption']
__version__ = '0.2'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np
import scipy.interpolate as _interp

import bio_rtd.utils as _utils
import bio_rtd.core as _core
import bio_rtd.pdf as _pdf


class AlternatingChromatography(_core.UnitOperation):

    def __init__(self,
                 t: _np.ndarray,
                 uo_id: str,
                 load_bt: _core.ChromatographyLoadBreakthrough,
                 peak_shape_pdf: _core.PDF,
                 gui_title: str = "ACC"):
        super().__init__(t, uo_id, gui_title)
        # get load breakthrough profile
        self.load_bt = load_bt
        # get elution peak pdf
        self.elution_peak_shape = peak_shape_pdf

        # process buffer species that are not binding to the column
        self.non_binding_species = []

        # column volume
        self.cv: float = -1  # column volume
        """Column volume"""
        self.column_porosity_retentate = -1  # column porosity for product
        """Column rtm"""
        self.ft_mean_retentate = -1  # flow-through time for product (cv = f * ft_mean_retentate / porosity_retentate)

        # duration of equilibration step
        self.equilibration_cv = -1
        self.equilibration_t = -1

        # equilibration step flow rate
        self.equilibration_f = -1
        self.equilibration_f_rel = 1  # equilibration flow rate relative to the load flow rate

        # duration of the load phase
        self.load_cv = -1  # load duration in CV
        self.load_c_end_ss: _typing.Optional[_np.ndarray] = None  # breakthrough concentration upper limit (CV << load)
        self.load_c_end_relative_ss = -1  # relative breakthrough concentration upper limit (CV << load)
        self.load_c_end_estimate_with_iterative_solver = False  # finer optimization of cycle length determination
        self.load_c_end_estimate_with_iterative_solver_max_iter = 1000

        # extend first load phase (to achieve a faster steady-state)
        self.load_extend_first_cycle = False
        self.load_extend_first_cycle_cv = -1
        self.load_extend_first_cycle_t = -1

        # target load linear velocity (in order to estimate column length / diameter ratio)
        self.load_target_lin_velocity = -1  # units need to match other units chosen for the model

        # duration of the wash step
        self.wash_cv = -1
        self.wash_t = -1

        # wash step flow rate
        self.wash_f = -1
        self.wash_f_rel = 1  # wash flow rate relative to the load flow rate

        # Elution peak is scaled down by 1 - `unaccounted_losses_rel`.
        # Peak cut criteria is applied after the scale down.
        self.unaccounted_losses_rel = 0

        # duration of elution step
        self.elution_cv = -1
        self.elution_t = -1

        # elution flow rate
        self.elution_f = -1
        self.elution_f_rel = 1  # elution flow rate relative to the load flow rate
        self.elution_buffer_c: _np.ndarray = _np.array([])
        # peak position within elution step (1st moment, not max)
        self.elution_peak_position_cv = -1
        self.elution_peak_position_t = -1
        # elution peak cut
        # start (one of them needs to be defined)
        self.elution_peak_cut_start_t = -1
        self.elution_peak_cut_start_cv = -1
        self.elution_peak_cut_start_c_rel_to_peak_max = -1
        self.elution_peak_cut_start_peak_area_share = -1
        # end (one of them needs to be defined)
        self.elution_peak_cut_end_t = -1
        self.elution_peak_cut_end_cv = -1
        self.elution_peak_cut_end_c_rel_to_peak_max = -1
        self.elution_peak_cut_end_peak_area_share = -1

        # duration of regeneration step
        self.regeneration_cv = -1
        self.regeneration_t = -1
        # regeneration step flow rate
        self.regeneration_f = -1
        self.regeneration_f_rel = 1  # wash flow rate relative to the load flow rate

        # option to simulate desorption during wash step
        self.wash_desorption = False

        # For PCC
        # option to capture load breakthrough onto next column
        self.load_recycle = False
        # propagation dynamics of unbound material through the column
        self.load_recycle_pdf: _typing.Optional[_core.PDF] = None
        # option to capture breakthrough material during wash step onto third column (2nd is on load step)
        self.wash_recycle = False
        # duration of wash recycling
        self.wash_recycle_duration_cv = -1
        self.wash_recycle_duration_t = -1

    @_core.UnitOperation.log.setter
    def log(self, logger: _core._logger.RtdLogger):
        self._logger = logger
        # propagate logger across other elements with logging
        self._logger.set_data_tree(self._instance_id, self._log_tree)
        if self.load_recycle_pdf is not None:
            self.load_recycle_pdf.set_logger_from_parent(self.uo_id, logger)
        if self.load_recycle_pdf is not None:
            self.elution_peak_shape.set_logger_from_parent(self.uo_id, logger)
        if self.load_recycle_pdf is not None:
            self.load_bt.set_logger_from_parent(self.uo_id, logger)

    def _get_flow_value(self, step_name: str, var_name: str, flow: float, rel_flow: float) -> float:
        if flow > 0:
            self.log.i_data(self._log_tree, var_name, flow)
        elif rel_flow > 0:
            flow = rel_flow * self._load_f
            self.log.i_data(self._log_tree, var_name, flow)
        else:
            self.log.w(step_name + " step flow rate is not defined, using load flow rate instead")
            flow = self._load_f
        return flow

    def _get_time_value(self, step_name: str, var_name: str, t: float, cv: float, flow: float) -> float:
        t_sum = max(t, 0)
        if cv > 0:
            assert flow > 0, step_name + ": Flow rate must be defined ( > 0 ) if the duration was specified in CVs."
            assert self._cv > 0, "CV must be determined (by `calc_cv`) before calculating duration based on CVs"
            t_sum += cv * self._cv / flow  # sum
        if t <= 0 and cv <= 0:  # log
            self.log.w(step_name + " time is not defined")
        else:
            self.log.i_data(self._log_tree, var_name, t_sum)
        return t_sum  # return

    def _assert_non_binding_species(self):
        # make sure binding species list is valid
        if len(self.non_binding_species) > 0:
            assert max(self.non_binding_species) < self._n_species, \
                "Index of non_binding_species too large (indexes start with 0)"
            assert list(set(self.non_binding_species)) \
                == list(self.non_binding_species), \
                "List of non_binding_species should have ascending order"
            assert len(self.non_binding_species) < self._n_species, \
                "All species cannot be non-binding."
        self.log.i_data(self._log_tree,
                        'non_binding_species', self.non_binding_species)

    def _calc_load_f(self):
        assert self._is_flow_box_shaped(), "Inlet flow must be box shaped"
        self._load_f = self._f.max()
        self.log.d_data(self._log_tree, 'load_f', self._load_f)

    def _calc_cv(self):
        self._ensure_single_non_negative_parameter(
            self.log.ERROR, self.log.ERROR,
            cv=self.cv,
            ft_mean_retentate=self.ft_mean_retentate,
        )

        if self.cv > 0:
            self._cv = self.cv
        else:  # self.ft_mean_retentate > 0:
            assert self.column_porosity_retentate > 0, \
                "porosity_retentate must be defined to calc CV from " \
                "ft_mean_retentate"
            assert self._load_f > 0, "load flow rate must be defined to calc CV from ft_mean_retentate"
            self._cv = self.ft_mean_retentate * self._load_f / self.column_porosity_retentate

        self.log.i_data(self._log_tree, 'cv', self._cv)

    def _report_column_dimensions(self):
        # report column dimensions based on desired load linear velocity
        if self.load_target_lin_velocity > 0:
            self._col_h = self._cv * self.load_target_lin_velocity / self._load_f
            self.log.i_data(self._log_tree, "column_h", self._col_h)
            self.log.i_data(self._log_tree, "column_d", (self._cv / self._col_h / _np.pi) ** 0.5 * 2)

    def _calc_equilibration_t(self):
        if self.equilibration_cv > 0:
            # flow rate
            eq_f = self._get_flow_value(
                "Equilibration", "equilibration_f", self.equilibration_f, self.equilibration_f_rel
            )
            # duration
            self._equilibration_t = self._get_time_value(
                "Equilibration", "equilibration_t", self.equilibration_t, self.equilibration_cv, eq_f
            )
        else:
            self._equilibration_t = max(self.equilibration_t, 0)
            self.log.i_data(self._log_tree, 'equilibration_t', self._equilibration_t)

    def _calc_wash_t_and_f(self):
        # flow rate
        self._wash_f = self._get_flow_value("Wash", "wash_f", self.wash_f, self.wash_f_rel)
        # duration
        self._wash_t = self._get_time_value("Wash", "wash_t", self.wash_t, self.wash_cv, self._wash_f)

    def _calc_elution_t_and_f(self):
        # flow rate
        self._elution_f = self._get_flow_value("Elution", "elution_f", self.elution_f, self.elution_f_rel)
        # duration
        self._elution_t = self._get_time_value("Elution", "elution_t", self.elution_t, self.elution_cv, self._elution_f)

    def _calc_elution_peak_t(self):
        # elution peak mean position (1st momentum)
        self._elution_peak_t = self._get_time_value(
            "elution peak position",
            "elution_peak_position_t",
            self.elution_peak_position_t,
            self.elution_peak_position_cv,
            self._elution_f
        )

    def _update_elution_peak_pdf(self):

        assert self._elution_peak_t > 0
        assert self._elution_f > 0

        # calc elution peak shape
        self.elution_peak_shape.update_pdf(
            rt_mean=self._elution_peak_t,
            v_void=self._elution_peak_t * self._elution_f,
            f=self._elution_f
        )
        self._p_elution_peak = self.elution_peak_shape.get_p() \
            * (1 - self.unaccounted_losses_rel)
        self.log.d_data(self._log_tree, "p_elution_peak", self._p_elution_peak)

    def _calc_elution_peak_cut_i_start_and_i_end(self):
        elution_peak_pdf: _np.ndarray = self._p_elution_peak.copy()

        # start peak cut
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            elution_peak_cut_start_peak_area_share=self.elution_peak_cut_start_peak_area_share,
            elution_peak_cut_start_c_rel_to_peak_max=self.elution_peak_cut_start_c_rel_to_peak_max,
            elution_peak_cut_start_cv=self.elution_peak_cut_start_cv,
            elution_peak_cut_start_t=self.elution_peak_cut_start_t
        )
        # calc elution_peak_cut_start_i
        if self.elution_peak_cut_start_peak_area_share >= 0:
            elution_peak_cut_start_i = _utils.vectors.true_start(
                _np.cumsum(elution_peak_pdf * self._dt) >= self.elution_peak_cut_start_peak_area_share
            )
        elif self.elution_peak_cut_start_c_rel_to_peak_max >= 0:
            elution_peak_cut_start_i = _utils.vectors.true_start(
                elution_peak_pdf >= self.elution_peak_cut_start_c_rel_to_peak_max * elution_peak_pdf.max()
            )
        elif self.elution_peak_cut_start_cv >= 0:
            elution_peak_cut_start_i = int(self.elution_peak_cut_start_cv * self._cv / self._elution_f / self._dt)
        elif self.elution_peak_cut_start_t >= 0:
            elution_peak_cut_start_i = int(self.elution_peak_cut_start_t / self._dt)
        else:
            self.log.w("Elution peak cut start is not defined. Now collecting from the beginning of the elution phase")
            elution_peak_cut_start_i = 0
        # log results
        self.log.i_data(self._log_tree, "elution_peak_cut_start_i", elution_peak_cut_start_i)
        self.log.i_data(self._log_tree, "elution_peak_cut_start_t", elution_peak_cut_start_i * self._dt)

        # end peak cut
        self._ensure_single_non_negative_parameter(
            log_level_multiple=self.log.ERROR, log_level_none=self.log.WARNING,
            elution_peak_cut_end_peak_area_share=self.elution_peak_cut_end_peak_area_share,
            elution_peak_cut_end_c_rel_to_peak_max=self.elution_peak_cut_end_c_rel_to_peak_max,
            elution_peak_cut_end_cv=self.elution_peak_cut_end_cv,
            elution_peak_cut_end_t=self.elution_peak_cut_end_t,
        )
        # calc elution_peak_cut_end_i
        if self.elution_peak_cut_end_peak_area_share >= 0:
            elution_peak_cut_end_i = _utils.vectors.true_start(
                _np.cumsum(elution_peak_pdf * self._dt) >= (1 - self.elution_peak_cut_end_peak_area_share)
            )
        elif self.elution_peak_cut_end_c_rel_to_peak_max >= 0:
            elution_peak_cut_end_i = _utils.vectors.true_end(
                elution_peak_pdf >= self.elution_peak_cut_end_c_rel_to_peak_max * elution_peak_pdf.max()
            )
        elif self.elution_peak_cut_end_cv >= 0:
            elution_peak_cut_end_i = int(self.elution_peak_cut_end_cv * self._cv / self._elution_f / self._dt)
        elif self.elution_peak_cut_end_t >= 0:
            elution_peak_cut_end_i = _utils.vectors.true_end(self._t < self.elution_peak_cut_end_t)
        else:
            self.log.w("Elution peak cut end is not defined. Now collecting to the end of the elution phase")
            elution_peak_cut_end_i = elution_peak_pdf.size
        # log results
        self.log.i_data(self._log_tree, "elution_peak_cut_end_i", elution_peak_cut_end_i)
        self.log.i_data(self._log_tree, "elution_peak_cut_end_t", elution_peak_cut_end_i * self._dt)

        self._elution_peak_cut_start_i = elution_peak_cut_start_i
        self._elution_peak_cut_end_i = elution_peak_cut_end_i

        if self._elution_peak_cut_end_i * self._dt < self._elution_peak_t:
            self.log.w("Peak end is cut before its maximum")

        if self._elution_peak_cut_end_i * self._dt > self._elution_t:
            self.log.w("Peak cut end exceeds elution step duration")

        return elution_peak_cut_start_i, elution_peak_cut_end_i

    def _calc_elution_peak_mask(self):
        self._elution_peak_mask = _np.ones(int(round(self._elution_t / self._dt)), dtype=bool)
        self._elution_peak_mask[self._elution_peak_cut_end_i:] = False
        self._elution_peak_mask[:self._elution_peak_cut_start_i] = False
        self.log.d_data(self._log_tree, "elution_peak_interval", self._elution_peak_mask)

    def _update_load_btc(self):

        assert self._cv > 0, "CV must be defined by now"

        self.load_bt.update_btc_parameters(cv=self._cv)

    def _calc_regeneration_t(self):
        if self.regeneration_cv > 0:
            # flow rate
            eq_f = self._get_flow_value(
                "Regeneration", "regeneration_f", self.regeneration_f, self.regeneration_f_rel
            )
            # duration
            self._regeneration_t = self._get_time_value(
                "Regeneration", "regeneration_t", self.regeneration_t, self.regeneration_cv, eq_f
            )
        else:
            self._regeneration_t = max(self.regeneration_t, 0)
            self.log.i_data(self._log_tree, 'regeneration_t', self._regeneration_t)

    def _update_load_recycle_pdf(self, flow):
        """
        Updates pdf that describes propagation dynamics of unbound and desorbed material throughout the column.
        """

        assert self.load_recycle_pdf is not None, "load_recycle_pdf must be defined"
        assert self.column_porosity_retentate > 0, "Retentate porosity must be defined"
        assert self._cv > 0, "CV must be defined by now"

        v_void = self._cv * self.column_porosity_retentate
        self.load_recycle_pdf.update_pdf(v_void=v_void, f=flow, rt_mean=v_void / flow)
        self._p_load_recycle_pdf = self.load_recycle_pdf.get_p()

    def _calc_load_recycle_wash_i(self):
        if self.wash_recycle_duration_t > 0 \
                or self.wash_recycle_duration_cv > 0:
            self._wash_recycle_i_duration = int(self._get_time_value(
                "Wash recycle", "load_wash_recycle_t",
                self.wash_recycle_duration_t,
                self.wash_recycle_duration_cv,
                self._wash_f
            ) / self._dt)
        else:
            # same as wash duration
            assert self._wash_t > 0
            self._wash_recycle_i_duration = int(round(self._wash_t / self._dt))

    def _get_load_bt_cycle_switch_limit(self, load_c_ss: _np.ndarray) -> _np.ndarray:

        assert self.load_c_end_ss is not None or self.load_c_end_relative_ss > 0, \
            "Load step duration should be defined!"

        # calc
        if self.load_c_end_ss is not None:
            load_c_end_ss = self.load_c_end_ss
            if self.load_c_end_relative_ss > 0:
                self.log.w("Cycle time defined by `load_c_end_ss` and `load_c_end_relative_ss`."
                           "Simulation is using `load_c_end_ss`.")
        else:  # self.load_c_end_relative_ss > 0
            load_c_end_ss = self.load_c_end_relative_ss * load_c_ss

        # add to log
        self.log.i_data(self._log_tree, 'load_c_end_ss', load_c_end_ss)

        return load_c_end_ss

    # noinspection DuplicatedCode
    def _calc_cycle_t(self):
        """
        Calculates cycle time.

        Optional delay of first cycle is not part of this function
        """

        assert self._cv > 0
        assert self._load_f > 0

        if self.load_cv > 0:
            t_cycle = self.load_cv * self._cv / self._load_f

            if self.load_c_end_ss is not None or self.load_c_end_relative_ss > 0:
                self.log.w("Cycle time defined in more than one way. Simulation is using `load_cv`.")
        else:
            # ## get bt profile for constant inlet ##
            # inlet c
            binding_species = [i for i in range(self._n_species)
                               if i not in self.non_binding_species]
            load_c_ss = self._estimate_steady_state_mean_c(binding_species)

            #  simulate first cycle at constant load concentration
            f_first_load = self._load_f * _np.ones(self._t.size)
            c_first_load = load_c_ss * _np.ones([len(binding_species), self._t.size])
            bt_first_load: _np.ndarray = load_c_ss - self.load_bt.calc_c_bound(f_first_load, c_first_load)

            # propagate breakthrough
            bt_first_load_out, bt_first_wash_out = self._sim_c_recycle_propagation(f_first_load, bt_first_load, None)

            # calc cycle duration
            load_c_end_ss = self._get_load_bt_cycle_switch_limit(load_c_ss)
            # noinspection PyTypeChecker
            i_t_first_cycle = _utils.vectors.true_start(bt_first_load_out.sum(0) >= load_c_end_ss.sum())
            t_cycle = i_t_first_cycle * self._dt

            # wash desorption
            if self.wash_desorption and self.wash_recycle:
                c_wash_desorbed = self._sim_c_wash_desorption(
                    f_first_load[:i_t_first_cycle],
                    c_first_load[:, :i_t_first_cycle] - bt_first_load[:, :i_t_first_cycle]
                )
            else:
                c_wash_desorbed = None
            bt_first_load_out, bt_first_wash_out = self._sim_c_recycle_propagation(
                f_first_load[:i_t_first_cycle], bt_first_load[:, :i_t_first_cycle], c_wash_desorbed
            )

            if self.load_recycle:
                if not self.load_c_end_estimate_with_iterative_solver:
                    self.log.w("Estimating cycle duration: Assuming sharp breakthrough profile")
                i_load_recycle_start = self._wash_recycle_i_duration if self.wash_recycle else 0
                m_load_recycle = \
                    bt_first_load_out[:, i_load_recycle_start:i_t_first_cycle].sum() * self._load_f * self._dt
                _t_diff = m_load_recycle / self._load_f / load_c_ss.sum()
                t_cycle -= _t_diff
                self._load_recycle_m_ss = m_load_recycle
                self.log.i_data(self._log_tree, 'm_load_recycle_ss', m_load_recycle)
                self.log.i_data(self._log_tree, 'shorten_cycle_t_due_to_bt_recycle', _t_diff)

            if self.wash_recycle:
                if not self.load_c_end_estimate_with_iterative_solver:
                    self.log.w("Estimating cycle duration: Assuming sharp breakthrough profile")
                m_wash_recycle = bt_first_wash_out[:, :self._wash_recycle_i_duration].sum() * self._wash_f * self._dt
                _t_diff = m_wash_recycle / self._load_f / load_c_ss.sum()
                t_cycle -= _t_diff
                self._wash_recycle_m_ss = m_wash_recycle
                self.log.i_data(self._log_tree, 'm_wash_recycle_ss', m_wash_recycle)
                self.log.i_data(self._log_tree, 'shorten_cycle_t_due_to_wash_recycle', _t_diff)

            if self.load_c_end_estimate_with_iterative_solver and (self.wash_recycle or self.load_recycle):
                c_load_fist_cycle = load_c_ss * _np.ones([len(binding_species), i_t_first_cycle * 2])

                def sim_cycle(f_load, c_load, i_prev_cycle) -> (_np.ndarray, _np.ndarray, int):
                    # load
                    bt_load: _np.ndarray = c_load - self.load_bt.calc_c_bound(f_load, c_load)
                    # propagate breakthrough
                    bt_load_out, _ = self._sim_c_recycle_propagation(f_load, bt_load, None)

                    # 'stop' load at specified breakthrough criteria
                    # noinspection PyTypeChecker
                    i_cycle_duration = _utils.vectors.true_start(bt_load_out.sum(0) >= load_c_end_ss.sum())
                    # cut load at specified time
                    bt_load = bt_load[:, :i_cycle_duration]

                    # wash desorption
                    if self.wash_desorption and self.wash_recycle:
                        c_first_wash_desorbed = self._sim_c_wash_desorption(
                            f_load[:i_cycle_duration],
                            c_load[:, :i_cycle_duration] - bt_load[:, :i_cycle_duration]
                        )
                    else:
                        c_first_wash_desorbed = None

                    # propagate load and wash leftovers
                    bt_load_out, bt_wash_out = self._sim_c_recycle_propagation(
                        f_load[:i_cycle_duration], bt_load, c_first_wash_desorbed
                    )

                    # construct load for next cycle
                    # recycle load
                    if self.load_recycle:
                        rec_load = bt_load_out[:, i_prev_cycle:i_cycle_duration]
                    else:
                        rec_load = _np.zeros_like(bt_load_out[:, i_prev_cycle:i_cycle_duration])
                    # next load profiles
                    c_next_load = _np.concatenate((rec_load, c_load_fist_cycle), axis=1)
                    f_next_load = self._load_f * _np.ones(c_next_load.shape[1])

                    wash_recycle_i_duration = self._wash_recycle_i_duration if self.wash_recycle else 0
                    m_load_recycle_ss = \
                        bt_first_load_out[:, wash_recycle_i_duration:i_t_first_cycle].sum() * self._load_f * self._dt
                    self._load_recycle_m_ss = m_load_recycle_ss
                    self.log.i_data(self._log_tree, 'm_load_recycle_ss', m_load_recycle_ss)
                    # recycle wash
                    if self.wash_recycle:
                        c_next_load[:, :self._wash_recycle_i_duration] = bt_wash_out[:, :self._wash_recycle_i_duration]
                        f_next_load[:self._wash_recycle_i_duration] = self._wash_f
                        m_wash_recycle_ss = \
                            bt_wash_out[:, :self._wash_recycle_i_duration].sum() * self._wash_f * self._dt
                        self._wash_recycle_m_ss = m_wash_recycle_ss
                        self.log.i_data(self._log_tree, 'm_wash_recycle_ss', m_wash_recycle_ss)

                    # return next load and cycle duration
                    return f_next_load, c_next_load, i_cycle_duration - i_prev_cycle

                f_load_cycle = self._load_f * _np.ones(c_load_fist_cycle.shape[1])
                c_load_cycle = c_load_fist_cycle
                i_t_cycle_prev = i_t_first_cycle
                i_t_cycle_estimate = 0
                # loop until cycle duration converges
                for i in range(self.load_c_end_estimate_with_iterative_solver_max_iter):
                    if abs(i_t_cycle_prev - i_t_cycle_estimate) <= 1:
                        self.log.i_data(self._log_tree, "t_cycle_optimization_loop_iter", i)
                        break
                    i_t_cycle_prev = i_t_cycle_estimate
                    f_load_cycle, c_load_cycle, i_t_cycle_estimate = \
                        sim_cycle(f_load_cycle, c_load_cycle, i_t_cycle_prev)
                    # print([i, i_t_cycle_prev, i_t_cycle_estimate])
                if abs(i_t_cycle_prev - i_t_cycle_estimate) > 1:
                    self.log.w("Cycle duration estimator did not converge")

                t_cycle = i_t_cycle_estimate * self._dt

            elif self.load_c_end_estimate_with_iterative_solver:
                self.log.w("No need to use iterative solver in case of no recycling of load and/or wash")

        self._cycle_t = t_cycle
        self.log.i_data(self._log_tree, 'cycle_t', t_cycle)

    # noinspection DuplicatedCode
    def _calc_first_cycle_extension_t(self):
        if not self.load_recycle and not self.wash_recycle:
            self.log.w("Estimation of first cycle extension requested on a process without load recycle")
            self._first_cycle_extension_t = 0
            return

        if not self.load_extend_first_cycle:
            self.log.w("Estimation of first cycle extension requested on a process without extended first cycle")
            self._first_cycle_extension_t = 0
            return

        if self.load_extend_first_cycle_t > 0:
            self._first_cycle_extension_t = self.load_extend_first_cycle_t
            return

        if self.load_extend_first_cycle_cv >= 0:
            assert self._cv > 0, "CV should be defined by now"
            assert self._load_f > 0, "Load flow rate should be defined by now"
            self._first_cycle_extension_t = self.load_extend_first_cycle_cv * self._cv / self._load_f
        elif self.load_cv > 0:
            raise NotImplementedError(
                "Estimation of first cycle extension is only supported if the cycle length "
                "is defined by breakthrough cutoff criteria. This is due to the fact that if all the "
                "breakthrough material gets recycles, there is no single steady-state."
            )
        else:
            binding_species = [i for i in range(self._n_species)
                               if i not in self.non_binding_species]
            load_c_ss = self._estimate_steady_state_mean_c(binding_species)

            #  simulate first cycle at constant load concentration
            f_first_load = self._load_f * _np.ones(self._t.size)
            c_first_load = load_c_ss * _np.ones([len(binding_species), self._t.size])
            bt_first_load: _np.ndarray = load_c_ss - self.load_bt.calc_c_bound(f_first_load, c_first_load)

            # propagate breakthrough
            bt_first_load_out, _ = self._sim_c_recycle_propagation(f_first_load, bt_first_load, None)
            load_c_end_ss = self._get_load_bt_cycle_switch_limit(load_c_ss)

            # noinspection PyTypeChecker
            i_t_first_cycle = _utils.vectors.true_start(bt_first_load_out.sum(0) >= load_c_end_ss.sum())
            dm = 0
            if self.load_recycle:
                assert hasattr(self, "_load_recycle_m_ss"), "Function `_calc_cycle_t()` should already be called."
                dm += self._load_recycle_m_ss
            if self.wash_recycle:
                assert hasattr(self, "_wash_recycle_m_ss"), "Function `_calc_cycle_t()` should already be called."
                dm += self._wash_recycle_m_ss

            di = 0
            if dm > 0:
                m_ext_bt = _np.cumsum(bt_first_load_out.sum(0)[i_t_first_cycle:]) * self._load_f * self._dt
                di += _utils.vectors.true_start(m_ext_bt >= dm)
            self._first_cycle_extension_t = di * self._dt

    def _calc_cycle_start_i_list(self):
        assert self._cycle_t > 0, "Cycle length must have been determined (by `_calc_cycle_t`) by now"
        flow_i_start, flow_i_end = _utils.vectors.true_start_and_end(self._f > 0)
        if self.load_extend_first_cycle:
            assert self._first_cycle_extension_t >= 0, \
                "Prolong of first load cycle is set to True, but the length is undefined"
            if self._first_cycle_extension_t == 0:
                self.log.w("Prolong of first load cycle is set to True, but the length of the extension is 0")
            load_extend_first_cycle_t = self._first_cycle_extension_t
            self.log.i_data(self._log_tree, "load_extend_first_cycle_t", load_extend_first_cycle_t)
        else:
            load_extend_first_cycle_t = 0

        cycle_start_t_list = _np.arange(
            self._t[flow_i_start] + load_extend_first_cycle_t,
            self._t[flow_i_end - 1],
            self._cycle_t
        )

        cycle_start_t_list[0] = self._t[flow_i_start]
        self._cycle_start_i_list = _np.rint(cycle_start_t_list / self._dt).astype(_np.int32)
        self.log.i_data(self._log_tree, "cycle_start_t_list", cycle_start_t_list)

    def _prepare_simulation(self):
        """Prepare everything before simulation loop over cycles"""

        self._assert_non_binding_species()

        self._calc_load_f()

        self._calc_cv()  # might depend on load_f
        self._report_column_dimensions()  # optional

        self._calc_equilibration_t()

        self._calc_wash_t_and_f()

        self._calc_elution_t_and_f()
        self._calc_elution_peak_t()
        self._update_elution_peak_pdf()
        self._calc_elution_peak_cut_i_start_and_i_end()
        self._calc_elution_peak_mask()

        self._calc_regeneration_t()

        # prepare for estimation of cycle length
        self._update_load_btc()
        if self.load_recycle:
            self._update_load_recycle_pdf(self._wash_f)
            if self.wash_recycle:
                self._calc_load_recycle_wash_i()
        # calc cycle time
        self._calc_cycle_t()
        if self.load_extend_first_cycle:
            self._calc_first_cycle_extension_t()
        # calc cycle start positions -> column switch time points (at load)
        self._calc_cycle_start_i_list()

        # make sure cycle duration is long enough
        _t_cycle_except_load = self._equilibration_t + self._wash_t + self._elution_t + self._regeneration_t
        if self._cycle_t < _t_cycle_except_load:
            self.log.e(f"Load step ({self._cycle_t}) should not be shorter"
                       f" than eq_t + wash_t + elution_t + regeneration_t"
                       f" ({_t_cycle_except_load: .6})!")

    def _sim_c_load_binding(self, f_load: _np.ndarray, c_load: _np.ndarray) -> (_np.ndarray, _np.ndarray):
        """Evaluates load against breakthrough profile to determine what part of load binds."""

        assert f_load.size == c_load.shape[1], \
            "f_load and c_load must have the same length"
        assert c_load.shape[0] == \
            self._n_species - len(self.non_binding_species), \
            "c_load must contain all binding species"

        c_bound = self.load_bt.calc_c_bound(f_load, c_load)

        return c_bound, c_load - c_bound  # returns bound and unbound parts

    def _sim_c_wash_desorption(self, f_load: _np.ndarray, c_bound: _np.ndarray) -> _np.ndarray:
        """Returns concentration profile of desorbed material during wash step."""

        # Not implemented in core this class, as there is no consensus on typical dynamics or way to describe it.
        raise NotImplementedError("Function not implemented in this class")

    def _sim_c_recycle_propagation(self,
                                   f_load: _np.ndarray,
                                   c_load_unbound: _np.ndarray,
                                   c_wash_desorbed: _typing.Optional[_np.ndarray]) -> (_np.ndarray, _np.ndarray):
        """
        Propagate the unbound (breakthrough during load) and desorbed (during wash) material through the column

        Parameters
        ----------
        f_load
            Flow rate profile of load step (which includes load step + possible wash and load recycle).
        c_load_unbound
            Concentration profile of overloaded material during load step (+ previous wash and load recycle).
        c_wash_desorbed
            Concentration profile of desorbed material during wash step.

        Returns
        -------
        (_np.ndarray, _np.ndarray)
            c_unbound_propagated
                Propagated (through the column) concentration profile of overloaded material during load step.
            c_wash_desorbed_propagated
                Propagated (through the column) concentration profile of desorbed material during wash step.
        """

        assert hasattr(self, "_wash_f") and self._wash_f > 0
        assert hasattr(self, "_wash_t") and self._wash_t > 0
        assert self.load_recycle_pdf is not None
        assert c_load_unbound.shape[0] == \
            self._n_species - len(self.non_binding_species)
        assert c_load_unbound.shape[1] == f_load.size
        if c_wash_desorbed is None or c_wash_desorbed.size == 0:
            c_wash_desorbed = _np.zeros([
                self._n_species - len(self.non_binding_species),
                int(round(self._wash_t / self._dt))])
        else:
            assert c_wash_desorbed.shape[0] == \
                   self._n_species - len(self.non_binding_species)
            assert c_wash_desorbed.shape[1] == \
                int(round(self._wash_t / self._dt))

        # combine on volumetric scale
        v_load = self._dt * f_load.cumsum()
        v_wash = v_load[-1] + self._dt * _np.arange(1, c_wash_desorbed.shape[1] + 1) * self._wash_f
        min_flow = min(f_load.min(), self._wash_f)
        dv = min_flow * self._dt
        v = _np.arange(dv, (v_wash[-1] if v_wash.size > 0 else v_load[-1]) + dv, dv)
        c_v_combined = _interp.interp1d(
            _np.concatenate((v_load, v_wash), axis=0),
            _np.concatenate((c_load_unbound, c_wash_desorbed), axis=1),
            fill_value="extrapolate"
        )(v)
        c_v_combined[c_v_combined < 0] = 0

        # simulate traveling of leftover material through the column
        self._update_load_recycle_pdf(min_flow)
        c_v_combined_propagated = _utils.convolution.time_conv(
            self._dt, c_v_combined, self._p_load_recycle_pdf)

        # split back on time scale
        c_combined_propagated = _interp.interp1d(
            v,
            c_v_combined_propagated,
            fill_value="extrapolate"
        )(_np.concatenate((v_load, v_wash), axis=0))
        c_combined_propagated[c_combined_propagated < 0] = 0

        c_unbound_propagated = c_combined_propagated[:, :v_load.size]
        c_wash_desorbed_propagated = c_combined_propagated[:, v_load.size:]

        return c_unbound_propagated, c_wash_desorbed_propagated

    def _sim_c_elution_desorption(self, m_bound: _np.ndarray) -> (_np.ndarray, _np.ndarray):
        """
        Simulate elution step.

        Parameters
        ----------
        m_bound
            Vector with amount of product being bound to the column. `m_bound.size == n_species`

        Returns
        -------
        (_np.ndarray, _np.ndarray)
            c_elution
                Outlet concentration profile during the elution.
            mask_elution_peak
                Boolean vector. The peak is collected where the value is `True`.
        """

        assert self._elution_f > 0
        assert self._elution_t > 0

        i_elution_duration = int(round(self._elution_t / self._dt))

        c_elution = \
            self._p_elution_peak[_np.newaxis, :i_elution_duration] * \
            m_bound[:, _np.newaxis] / self._elution_f

        if c_elution.shape[1] < i_elution_duration:
            c_elution = _np.pad(c_elution, ((0, 0), (0, i_elution_duration - c_elution.shape[1])), mode="constant")

        mask_elution_peak = self._elution_peak_mask

        return c_elution, mask_elution_peak

    def _sim_c_elution_buffer(self, n_time_steps: int) -> _np.ndarray:
        """Feel free to override this function if you want to simulate linear gradient"""

        # elution buffer composition
        elution_buffer_composition = self.elution_buffer_c.reshape(self.elution_buffer_c.size, 1)

        assert elution_buffer_composition.size == 0 or elution_buffer_composition.size == self._n_species, \
            "Elution buffer composition must be either empty or have a concentration value for each specie"

        assert _np.all(elution_buffer_composition >= 0), "Concentration values in elution buffer must be >= 0"

        if elution_buffer_composition.size == 0:
            elution_buffer_composition = _np.zeros([self._n_species, 1])

        self.log.i_data(self._log_tree, "elution_buffer_composition", elution_buffer_composition)

        return elution_buffer_composition * _np.ones_like(self._t[:n_time_steps])

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _sim_c_regeneration(self, m_bound: _np.ndarray) -> _typing.Optional[_np.ndarray]:
        """
        Simulate regeneration step.

        Parameters
        ----------
        m_bound
            Vector with amount of product being bound to the column. `m_bound.size == n_species`

        Returns
        -------
        _typing.Optional[_np.ndarray]
            Outlet concentration profile during regeneration step.
        """

        c_regeneration = None

        return c_regeneration

    def _sim_c_out_cycle(self, f_load: _np.ndarray, c_load: _np.ndarray) -> (_typing.Optional[_np.ndarray],
                                                                             _typing.Optional[_np.ndarray],
                                                                             _np.ndarray,
                                                                             _np.ndarray,
                                                                             _typing.Optional[_np.ndarray]):
        """
        Simulates load-wash-elution-regeneration steps. Regeneration is optional.

        This function can be replaced in case user wants to use some other evaluation of bind-elution dynamics.

        Parameters
        ----------
        f_load: _np.ndarray
            Inlet load flow rate profile. It might be changing due to wash recycle.
        c_load: _np.ndarray
            Inlet load concentration profile.

        Returns
        -------
        (_typing.Optional[_np.ndarray], _typing.Optional[_np.ndarray],
         _np.ndarray, _np.ndarray,
         _typing.Optional[_np.ndarray])
            Concentration profiles at the outlet of the column for load, wash, elution and regeneration steps.
            Elution peak cut is applied in this function. Elution peak must be defined.
            Profiles that are `None` are considered being zero.
        """

        assert self._load_f > 0
        assert self._wash_f > 0
        assert self._wash_t > 0
        assert self._elution_f > 0
        assert self._elution_t > 0
        assert self._load_f > 0
        assert self._cv > 0

        # evaluate binding
        c_bound, c_unbound = self._sim_c_load_binding(f_load, c_load)
        # log
        m_load = (c_load * f_load[_np.newaxis, :]).sum(1) * self._dt
        m_bound = (c_bound * f_load[_np.newaxis, :]).sum(1) * self._dt
        self.log.i_data(self._cycle_tree, "column_utilization", m_bound / self._cv / self.load_bt.get_total_bc())
        self.log.i_data(self._cycle_tree, "m_load", m_load)
        self.log.i_data(self._cycle_tree, "m_bound", m_bound)
        self.log.i_data(self._cycle_tree, "m_unbound", m_load - m_bound)
        self.log.d_data(self._cycle_tree, "f_load", f_load)
        self.log.d_data(self._cycle_tree, "c_load", c_load)
        self.log.d_data(self._cycle_tree, "c_bound", c_bound)
        self.log.d_data(self._cycle_tree, "c_unbound", c_unbound)

        # evaluate desorption during wash
        c_wash_desorbed = None
        if self.wash_desorption:
            c_wash_desorbed = self._sim_c_wash_desorption(f_load, c_bound)
            if c_wash_desorbed.size > 0:
                m_bound -= c_wash_desorbed.sum(1)  # subtract desorbed material from bound material
            # log
            self.log.i_data(self._cycle_tree, "m_wash_desorbed", c_wash_desorbed.sum(1) * self._wash_f * self._dt)
            self.log.d_data(self._cycle_tree, "c_wash_desorbed", c_wash_desorbed)

        # propagate unbound and desorbed material throughout the column
        c_out_load = c_unbound
        c_out_wash = c_wash_desorbed
        if self.load_recycle or self.wash_recycle:
            c_out_load, c_out_wash = self._sim_c_recycle_propagation(f_load, c_unbound, c_wash_desorbed)

        # get elution peak
        c_out_elution, elution_peak_mask = self._sim_c_elution_desorption(m_bound)
        # log
        m_elution_peak = (c_out_elution * elution_peak_mask[_np.newaxis, :]).sum(1) * self._elution_f * self._dt
        m_elution = c_out_elution.sum(1) * self._elution_f * self._dt
        self.log.i_data(self._cycle_tree, "m_elution_peak", m_elution_peak)
        self.log.i_data(self._cycle_tree, "m_elution", m_elution)
        self.log.i_data(self._cycle_tree, "m_elution_peak_cut_loss", m_elution - m_elution_peak)

        # get regeneration peak
        c_out_regeneration = self._sim_c_regeneration(m_bound - c_out_elution.sum(1) * self._elution_f * self._dt)

        return c_out_load, c_out_wash, c_out_elution, elution_peak_mask, c_out_regeneration

    def _calculate(self):
        # pre calculate parameters and repetitive profiles
        self._prepare_simulation()

        binding_species = [i for i in range(self._n_species)
                           if i not in self.non_binding_species]
        assert len(binding_species) > 0

        # copy inlet vectors
        c_in_load = self._c[binding_species].copy()
        f_in_load = self._f.copy()
        f_in_i_end = min(_utils.vectors.true_end(f_in_load > 0), self._t.size)
        c_in_load[:, f_in_i_end:] = 0

        # clear for results
        self._c[:] = 0
        self._f[:] = 0

        # prepare logger
        log_data_cycles = list()
        self.log.set_branch(self._log_tree, "cycles", log_data_cycles)

        # util variable
        previous_c_bt_wash: _typing.Optional[_np.ndarray] = None

        for i in range(self._cycle_start_i_list.size):
            """
            In one cycle we focus on load-wash-elution-regeneration-equilibration on the column.
            
            The loading step starts at `self._cycle_start_i_list[i]`.
            """

            # prepare logger for this cycle
            self._cycle_tree = dict()
            log_data_cycles.append(self._cycle_tree)

            # Load start and end time as the column sees it
            if i > 0 and self.load_recycle:
                # column sees leftovers from previous load during recycling
                cycle_load_i_start = self._cycle_start_i_list[i - 1]
            else:
                cycle_load_i_start = self._cycle_start_i_list[i]
            # Calc cycle end (either next cycle or end or simulation time)
            if i + 1 < self._cycle_start_i_list.size:
                cycle_load_i_end = self._cycle_start_i_list[i + 1]
            else:
                cycle_load_i_end = f_in_i_end - 1
            # log results
            self.log.i_data(self._cycle_tree, "i_cycle_load_start", cycle_load_i_start)
            self.log.i_data(self._cycle_tree, "i_cycle_load_step_start", self._cycle_start_i_list[i])
            self.log.i_data(self._cycle_tree, "i_cycle_load_end", cycle_load_i_end)

            # calc profiles at column outlet
            c_out_load, c_out_wash, c_out_elution, b_out_elution, c_out_regeneration = self._sim_c_out_cycle(
                f_in_load[cycle_load_i_start:cycle_load_i_end],
                c_in_load[:, cycle_load_i_start:cycle_load_i_end]
            )
            self.log.d_data(self._cycle_tree, "c_out_load", c_out_load)
            self.log.d_data(self._cycle_tree, "c_out_wash", c_out_wash)
            self.log.d_data(self._cycle_tree, "c_out_elution", c_out_elution)
            self.log.d_data(self._cycle_tree, "b_out_elution", b_out_elution)
            self.log.d_data(self._cycle_tree, "c_out_regeneration", c_out_regeneration)

            # load recycle
            if self.load_recycle:
                # recycle load during the load step
                i_load_start_rel = self._cycle_start_i_list[i] - cycle_load_i_start
                c_load_recycle = c_out_load[:, i_load_start_rel:]
                c_in_load[:, self._cycle_start_i_list[i]:cycle_load_i_end] = c_load_recycle
                self.log.i_data(self._cycle_tree, "m_load_recycle", c_load_recycle.sum(1) * self._load_f * self._dt)
                self.log.d_data(self._cycle_tree, "c_load_recycle", c_load_recycle)
                # losses during load == bt through 2nd column == bt before start of load step
                c_loss_bt_2nd_column = c_out_load[:, i_load_start_rel]
                self.log.d_data(self._cycle_tree, "m_loss_bt_2nd_column",
                                c_loss_bt_2nd_column.sum() * self._dt * self._load_f)
                self.log.i_data(self._cycle_tree, "c_loss_bt_2nd_column", c_loss_bt_2nd_column)
            else:
                # report losses during load
                m_loss_load = c_out_load.sum() * self._dt * self._load_f
                self.log.i_data(self._cycle_tree, "m_loss_load", m_loss_load)

            # wash recycle
            if self.wash_recycle:
                if previous_c_bt_wash is not None and previous_c_bt_wash.size > 0:
                    # clip wash recycle duration if needed
                    i_wash_duration = min(self._wash_recycle_i_duration, self._t.size - self._cycle_start_i_list[i])

                    # report losses due to discarding load bt while recycling wash
                    s = c_in_load[:, self._cycle_start_i_list[i]:self._cycle_start_i_list[i] + i_wash_duration]
                    self.log.i_data(self._cycle_tree,
                                    "m_loss_load_bt_during_wash_recycle",
                                    s.sum() * self._dt * self._load_f)
                    self.log.d_data(self._cycle_tree, "c_lost_load_during_wash_recycle", s)
                    self.log.d_data(self._cycle_tree, "c_wash_recycle", previous_c_bt_wash[:, :i_wash_duration])
                    self.log.i_data(self._cycle_tree, "m_wash_recycle",
                                    previous_c_bt_wash[:, :i_wash_duration].sum(1) * self._dt * self._wash_f)

                    # apply previous wash recycle onto the inlet profile
                    s[:] = previous_c_bt_wash[:, :i_wash_duration]
                    f_in_load[self._cycle_start_i_list[i]:self._cycle_start_i_list[i] + i_wash_duration] = self._wash_f
                # save wash from this cycle to be applied during the next cycle
                previous_c_bt_wash = c_out_wash
            else:
                # report losses during wash
                if c_out_wash is None:
                    c_out_wash = _np.zeros([len(binding_species), int(round(self._wash_t / self._dt))])
                m_loss_wash = c_out_wash.sum() * self._dt * self._load_f
                self.log.i_data(self._cycle_tree, "m_loss_wash", m_loss_wash)

            # elution
            [i_el_rel_start, i_el_rel_end] = _utils.vectors.true_start_and_end(b_out_elution)
            i_el_start = min(self._t.size, cycle_load_i_end + c_out_wash.shape[1] + i_el_rel_start)
            i_el_end = min(self._t.size, cycle_load_i_end + c_out_wash.shape[1] + i_el_rel_end)
            i_el_rel_end = i_el_rel_start + i_el_end - i_el_start
            # log
            self.log.i_data(self._cycle_tree, "i_elution_start", i_el_start)
            self.log.i_data(self._cycle_tree, "i_elution_end", i_el_end)

            # write to global outlet
            self._f[i_el_start:i_el_end] = self._elution_f
            self._c[binding_species, i_el_start:i_el_end] = c_out_elution[:, i_el_rel_start:i_el_rel_end]


class ACC(AlternatingChromatography):

    def _sim_c_wash_desorption(self, f_load: _np.ndarray, c_bound: _np.ndarray) -> _np.ndarray:
        raise NotImplementedError("Function not implemented in this class")


class PCC(AlternatingChromatography):

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
        self.wash_recycle = False

        # how fast is the protein traveling through the column during overloaded conditions (in CVs)
        self.column_porosity_retentate = column_porosity_retentate
        # corresponding peak shape
        self.load_recycle_pdf = load_recycle_pdf

    def _sim_c_wash_desorption(self, f_load: _np.ndarray, c_bound: _np.ndarray) -> _np.ndarray:
        raise NotImplementedError("Function not implemented in this class")


class PCCWithWashDesorption(PCC):

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
        self.wash_recycle = True
        self.wash_desorption = True

        # one of those must be defined
        self.wash_desorption_desorbable_material_share = -1
        self.wash_desorption_desorbable_above_dbc = -1

        # this one must be defined
        self.wash_desorption_tail_half_time_cv = -1

    def _sim_c_wash_desorption(self, f_load: _np.ndarray, c_bound: _np.ndarray) -> _np.ndarray:

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

        m_bound = (c_bound * f_load[_np.newaxis, :]).sum(1)[:, _np.newaxis] * self._dt

        k = -1
        if self.wash_desorption_desorbable_material_share > 0:
            k = self.wash_desorption_desorbable_material_share
        if self.wash_desorption_desorbable_above_dbc > 0:
            if k > 0:
                self.log.w("share of desorbable material defined twice!! "
                           "Using load_recycle_wash_desorbable_material_share")
            else:
                k = max(0, 1 - self.wash_desorption_desorbable_above_dbc * self._cv / m_bound.sum())

        assert 1 >= k >= 0

        i_wash_duration = int(round(self._wash_t / self._dt))

        # generate exponential tail
        exp_pdf = _pdf.TanksInSeries(self._t[:i_wash_duration], 1)
        tau = self.wash_desorption_tail_half_time_cv * self._cv / self._wash_f / _np.log(2)
        exp_pdf.update_pdf(rt_mean=tau)
        # scale desorbed material concentration due to differences in flow rate (ensure mass balance)
        c_desorbed = m_bound * k * exp_pdf.get_p()[_np.newaxis, :i_wash_duration] / self._wash_f
        c_desorbed = _np.pad(c_desorbed, ((0, 0), (0, i_wash_duration - c_desorbed.shape[1])), mode="constant")

        # cut
        return c_desorbed
