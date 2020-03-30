"""Template for FlowThrough and FlowThroughWithSwitching UnitOperation

For more details about process parameters see docstrings of
`FlowThrough` and `FlowThroughWithSwitching`.

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
Example is for `FlowThroughWithSwitching`. Procedure is the same for
`FlowThrough`. `FlowThroughWithSwitching` has some additional
parameters that are marked.

"""

__version__ = '0.3.0'
__author__ = 'Jure Sencar'

import numpy as np
from typing import List

from bio_rtd.uo.fc_uo import FlowThroughWithSwitching
from bio_rtd import core, pdf

"""Direct instance creation."""

t = np.linspace(0, 10, 100)

tmp_uo = FlowThroughWithSwitching(
    t=t,
    pdf=pdf.GaussianFixedDispersion(t, 2 * 2 / 45),
    uo_id="dilution_direct",
    gui_title="Dilution, direct instance"  # Optional.
)
tmp_uo.v_void = 3.4
tmp_uo.t_cycle = 20


"""Using dict of parameters and attributes."""

PARAMETERS = {
    # Required.
    "uo_id": str,
    # Required.
    "pdf": core.PDF,
    # Optional.
    "gui_title": str,  # default: FlowThrough, FlowThroughWithSwitching
}

ATTRIBUTES = {
    # One of next two.
    "v_void": float,
    "rt_target": float,  # target mean residence time

    # One or none. If none, v_init == v_void is assumed.
    "v_init": float,
    "v_init_ratio": float,  # share of void volume (v_void)

    # Optional. Default = [] (empty).
    "c_init": np.ndarray,  # equilibration buffer composition

    # Optional. Default = 0.
    "losses_share": float,
    # List of species to which losses apply. Default = [] (none).
    "losses_species_list": List[int],

    # ============== For `FlowThroughWithSwitching` only ===============
    # One of those three.
    "t_cycle": float,
    "v_cycle": float,
    "v_cycle_relative": float,
    # ==================================================================

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""1. Define a time step and a simulation time vector."""
t = np.linspace(0, 100, 1001)  # it must start with 0
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
    "uo_id": "FlowThrough_template_implementation",
    # Required.
    "pdf": pdf.GaussianFixedDispersion(t, 2 * 2 / 45),
    # Optional.
    # "gui_title": str,  # default: FlowThrough, FlowThroughWithSwitching
}

uo_attr = {
    # One of next two.
    "v_void": 3.4,
    # "rt_target": float,  # target mean residence time

    # One or none. If none, v_init == v_void is assumed.
    # "v_init": float,
    # "v_init_ratio": float,  # share of void volume (v_void)

    # Optional. Default = [] (empty).
    "c_init": np.array([0, 0, 20]),  # equilibration buffer composition

    # Optional. Default = 0.
    "losses_share": 0.02,
    # List of species to which losses apply. Default = [] (none).
    "losses_species_list": [0, 1],

    # ============== For `FlowThroughWithSwitching` only ===============
    # One of those three.
    # "t_cycle": float,
    "v_cycle": 20,
    # "v_cycle_relative": float,
    # ==================================================================

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

flow_through = FlowThroughWithSwitching(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(flow_through, key), f"`{key}` is wrong."
    # Override value.
    setattr(flow_through, key, value)

# Voila :)
