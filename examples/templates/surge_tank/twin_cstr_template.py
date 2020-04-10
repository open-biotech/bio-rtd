"""Template for instantiating TwoAlternatingCSTRs UnitOperation.

For more details see docstrings of `TwoAlternatingCSTRs`.

See Also
--------
:class:`bio_rtd.uo.surge_tank.TwoAlternatingCSTRs`

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as np

from bio_rtd.uo.surge_tank import TwoAlternatingCSTRs

"""Direct instance creation. Minimalistic example."""

tmp_uo = TwoAlternatingCSTRs(
    t=np.linspace(0, 10, 100),
    uo_id="twin_cstr_direct",
    gui_title="TwoAlternatingCSTRs, direct instance"  # Optional.
)


"""Using dict of parameters and attributes.

Guide:

1. Define a time step and a simulation time vector.
2. Use `PARAMETERS` and `ATTRIBUTES` as a template.
   Replace variable types with values.
   Remove or comment out the ones that are not needed.
3. Instantiate the unit operation with parameters.
   Updated instance with attributes.

See the example below `PARAMETERS` and `ATTRIBUTES`.

"""

PARAMETERS = {
    # Required.
    "uo_id": str,
    # Optional.
    "gui_title": str,  # default: TwoAlternatingCSTRs
}

ATTRIBUTES = {
    # Optional. Number of collected periods in periodic inlet.
    "collect_n_periods": int,

    # When CSTRs switch roles within the inlet flow rate off time.
    "relative_role_switch_time": float,  # default: 0.9

    # Required for steady (non-periodic) inlet.
    # One of the following two. Cycle duration for steady inlet.
    "t_cycle": float,
    "v_cycle": float,

    # Optional. One of the following two.
    # Leftover volume after discharge.
    "v_leftover": float,
    "v_leftover_rel": float,  # relative to collected volume in a cycle

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

"""

uo_pars = {
    # Required.
    "uo_id": "twin_cstr_1",
    # Optional.
    # "gui_title": str,  # default: TwoAlternatingCSTRs
}

uo_attr = {
    # Optional. Number of collected periods in periodic inlet.
    "collect_n_periods": 2,

    # When CSTRs switch roles within the inlet flow rate off time.
    "relative_role_switch_time": 0.8,  # default: 0.9

    # Required for steady (non-periodic) inlet.
    # One of the following two. Cycle duration for steady inlet.
    # "t_cycle": float,
    # "v_cycle": float,

    # Optional. One of the following two.
    # Leftover volume after discharge.
    # "v_leftover": float,
    "v_leftover_rel": 0.05,  # 95 % discharge

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

cstr = TwoAlternatingCSTRs(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(cstr, key), f"`{key}` is wrong."
    # Override value.
    setattr(cstr, key, value)

# Voila :)
