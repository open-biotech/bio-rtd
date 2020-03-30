"""Template for PCC and PCCWithWashDesorption unit operations.

For more details about process parameters see docstrings of `PCC` and
`PCCWithWashDesorption`.

Guide
-----
1. Define a time step and a simulation time vector.
2. Use `PARAMETERS` and `ATTRIBUTES` as a template.
   Replace variable types with values.
   Remove or comment out the ones that are not needed.
3. Instantiate the unit operation with parameters.
   Updated instance with attributes.

See the example below `PARAMETERS` and `ATTRIBUTES`.

Notes
-----
Example is for `PCCWithWashDesorption`. Procedure is the same for `PCC`.
`PCCWithWashDesorption` has some additional parameters that are marked.

"""

__version__ = '0.3.0'
__author__ = 'Jure Sencar'

import numpy as np
from typing import List

from bio_rtd.uo.sc_uo import PCCWithWashDesorption
from bio_rtd.chromatography import bt_load
from bio_rtd import core, pdf

PARAMETERS = {
    # Required.
    "uo_id": str,
    "load_bt": core.ChromatographyLoadBreakthrough,
    "peak_shape_pdf": core.PDF,
    "load_recycle_pdf": core.PDF,
    # Porosity of the column for protein.
    "column_porosity_retentate": float,

    # Optional.
    "gui_title": str,  # default: PCC, PCCWithWashDesorption
}

ATTRIBUTES = {

    # One of next two.
    "cv": float,
    "ft_mean_retentate": float,

    # Optional. Default = [] (empty).
    "non_binding_species": List[int],  # indexing starts with 0

    # One or both. If both, the duration adds up.
    "equilibration_cv": float,
    "equilibration_t": float,
    # Optional. One of next two. Default: `equilibration_f_rel` = 1
    "equilibration_f": float,
    "equilibration_f_rel": float,  # relative to inlet (load) flow rate

    # One of next three.
    "load_cv": float,
    "load_c_end_ss": np.ndarray,
    "load_c_end_relative_ss": float,
    # Optional. Default = False.
    "load_c_end_estimate_with_iterative_solver": bool,
    # Optional. Default = 1000.
    # Ignored if `load_c_end_estimate_with_iterative_solver == False`.
    "load_c_end_estimate_with_iterative_solver_max_iter": int,

    # Optional.
    "load_extend_first_cycle": bool,  # Default: False
    # Ignored if `load_extend_first_cycle == True`.
    # If both, the duration is added together.
    # If none, the duration is estimated by the model
    "load_extend_first_cycle_cv": float,
    "load_extend_first_cycle_t": float,

    # Optional.
    "load_target_lin_velocity": float,

    # One or both. If both, the duration adds up.
    "wash_cv": float,
    "wash_t": float,
    # Optional. One of next two. Default: `wash_f_rel` = 1
    "wash_f": float,
    "wash_f_rel": float,  # relative to inlet (load) flow rate

    # Optional.
    # Captures breakthrough material during wash step onto third column.
    # while 2nd column is loading.
    "wash_recycle": bool,
    # Optional. If both, the duration is added.
    # If none and `wash_recycle` is True, then the entire wash is recycled.
    "wash_recycle_duration_cv": float,
    "wash_recycle_duration_t": float,

    # Optional. Default = 0.
    # Elution peak is scaled down by (1 - `unaccounted_losses_rel`).
    # Peak cut criteria is applied after the scale down.
    "unaccounted_losses_rel": float,

    # One or both. If both, the duration adds up.
    "elution_cv": float,
    "elution_t": float,
    # Optional. One of next two. Default: `elution_f_rel` = 1
    "elution_f": float,
    "elution_f_rel": float,  # relative to inlet (load) flow rate
    # Optional. Default is empty array (-> all species are 0).
    "elution_buffer_c": np.ndarray,

    # One of next two.
    # Fist momentum relative to the beginning of elution step.
    "elution_peak_position_cv": float,
    "elution_peak_position_t": float,
    # One of next four.
    "elution_peak_cut_start_t": float,
    "elution_peak_cut_start_cv": float,
    "elution_peak_cut_start_c_rel_to_peak_max": float,
    "elution_peak_cut_start_peak_area_share": float,
    # One of next four.
    "elution_peak_cut_end_t": float,
    "elution_peak_cut_end_cv": float,
    "elution_peak_cut_end_c_rel_to_peak_max": float,
    "elution_peak_cut_end_peak_area_share": float,

    # One or both. If both, the duration adds up.
    "regeneration_cv": float,
    "regeneration_t": float,
    # Optional. One of next two. Default: `regeneration_f_rel` = 1
    "regeneration_f": float,
    "regeneration_f_rel": float,  # relative to inlet (load) flow rate

    # ================ For `PCCWithWashDesorption` only ================
    # Required.
    "wash_desorption_tail_half_time_cv": float,
    # One of those two.
    "wash_desorption_desorbable_material_share": float,
    "wash_desorption_desorbable_above_dbc": float,
    # ==================================================================

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""1. Define a time step and a simulation time vector."""
t = np.linspace(0, 1000, 10001)  # it must start with 0
dt = t[1]  # time step


"""2. Use `PARAMETERS` and `ATTRIBUTES` as a template.

Copy/Paste templates.
Replace variable types with values.
Remove or comment out the ones that are not needed.

Notes
-----
Process parameters in the following example were chosen for
demonstrating the model usability (rather than representing a real
chromatographic process).

"""

uo_pars = {
    # Required.
    "uo_id": "pcc_template_implementation",
    "load_bt": bt_load.ConstantPatternSolution(dt, dbc_100=240, k=0.05),
    "peak_shape_pdf": pdf.GaussianFixedDispersion(t, 8**2 / 30),
    "load_recycle_pdf": pdf.GaussianFixedDispersion(t, 2 * 2 / 30),
    # Protein porosity of the column.
    "column_porosity_retentate": 0.64,

    # Optional.
    # "gui_title": str,  # default: PCC, PCCWithWashDesorption
}

uo_attr = {
    # One of next two.
    "cv": 13,
    # "ft_mean_retentate": float,

    # Optional. Default = [].
    # "non_binding_species": List[int],  # indexing starts with 0

    # Optional. One or both. If both, the duration adds up.
    "equilibration_cv": 3,
    # "equilibration_t": float,

    # Optional. One of next two. Default: `equilibration_f_rel` = 1
    # "equilibration_f": float,
    # "equilibration_f_rel": 1,  # relative to inlet (load) flow rate

    # One of next three.
    # "load_cv": float,
    # "load_c_end_ss": np.ndarray,
    "load_c_end_relative_ss": 0.7,  # 70 % of breakthrough

    # Optional. Default = False.
    "load_c_end_estimate_with_iterative_solver": True,

    # Optional. Default = 1000.
    # Ignored if `load_c_end_estimate_with_iterative_solver == False`.
    # "load_c_end_estimate_with_iterative_solver_max_iter": int,

    # Optional.
    "load_extend_first_cycle": True,  # Default: False

    # Ignored if `load_extend_first_cycle == True`.
    # If both, the duration is added together.
    # If none, the duration is estimated by the model
    # "load_extend_first_cycle_cv": float,
    # "load_extend_first_cycle_t": float,

    # Optional.
    # "load_target_lin_velocity": float,

    # One or both. If both, the duration adds up.
    "wash_cv": 5,
    # "wash_t": float,
    # One of next two.
    # "wash_f": float,
    "wash_f_rel": 1,  # relative to inlet (load) flow rate

    # Optional.
    # Captures breakthrough material during wash step onto third column.
    # while 2nd column is loading.
    # "wash_recycle": bool,
    # Optional. If both, the duration is added.
    # If none and `wash_recycle` is True, then the entire wash is recycled.
    # "wash_recycle_duration_cv": float,
    # "wash_recycle_duration_t": float,

    # Optional. Default = 0.
    # Elution peak is scaled down by (1 - `unaccounted_losses_rel`).
    # Peak cut criteria is applied after the scale down.
    "unaccounted_losses_rel": 0.15,

    # One or both. If both, the duration adds up.
    "elution_cv": 3,
    # "elution_t": float,

    # One of next two.
    # "elution_f": float,
    "elution_f_rel": 1 / 4,  # relative to inlet (load) flow rate
    # Optional. Default is empty array (-> all species are 0).
    # "elution_buffer_c": np.ndarray,

    # One of next two.
    # Fist momentum relative to the beginning of elution step.
    "elution_peak_position_cv": 1.6,
    # "elution_peak_position_t": float,

    # One of next four.
    # "elution_peak_cut_start_t": float,
    "elution_peak_cut_start_cv": 1.05,
    # "elution_peak_cut_start_c_rel_to_peak_max": float,
    # "elution_peak_cut_start_peak_area_share": float,

    # One of next four.
    # "elution_peak_cut_end_t": float,
    "elution_peak_cut_end_cv": 2.3,
    # "elution_peak_cut_end_c_rel_to_peak_max": float,
    # "elution_peak_cut_end_peak_area_share": float,

    # One or both. If both, the duration adds up.
    "regeneration_cv": 1,
    # "regeneration_t": float,

    # One of next two.
    # "regeneration_f": float,
    "regeneration_f_rel": 1,  # relative to inlet (load) flow rate

    # ================ For `PCCWithWashDesorption` only ================
    "wash_desorption_tail_half_time_cv": 2,
    # One of those two.
    "wash_desorption_desorbable_material_share": 0.05,
    # "wash_desorption_desorbable_above_dbc": float,
    # ==================================================================

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

pcc = PCCWithWashDesorption(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(pcc, key), f"`{key}` is wrong."
    # Override value.
    setattr(pcc, key, value)

# Voila :)
