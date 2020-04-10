"""Template for instantiating CSTR UnitOperation.

For more details see docstrings of `CSTR`.

See Also
--------
:class:`bio_rtd.uo.surge_tank.CSTR`

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as np

from bio_rtd.uo.surge_tank import CSTR

"""Direct instance creation. Minimalistic example."""

tmp_uo = CSTR(
    t=np.linspace(0, 10, 100),
    uo_id="cstr_direct",
    gui_title="CSTR, direct instance"  # Optional.
)
# Size of the surge tank.
tmp_uo.v_void = 140


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
    "gui_title": str,  # default: CSTR
}

ATTRIBUTES = {
    # Required. One of the following four.
    # It determines the size of the CSTR.
    "rt_target": float,  # target mean residence time (first momentum)
    "v_void": float,
    # Next two require periodic inlet flow rate profile.
    "v_min": float,
    "v_min_ratio": float,  # 10 % safety margin

    # Optional. One of the following two. It determines initial fill volume.
    # If none are defined, the initial fill volume is the same as void volume.
    "v_init": float,
    "v_ratio": float,  # 20 % of the total (== void) volume

    # Optional. Default = False.
    # If True, then `v_init` and `v_init_ratio` are ignored.
    "starts_empty": bool,

    # Optional. Initial buffer composition.
    # Default is empty array (equivalent to 0).
    "c_add_buffer": np.ndarray,

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
    "uo_id": "cstr_1",
    # Optional.
    # "gui_title": str,  # default: Cstr
}

uo_attr = {
    # Required. One of the following four.
    # It determines the size of the CSTR.
    # "rt_target": float,  # target mean residence time (first momentum)
    # "v_void": float,
    # Next two require periodic inlet flow rate profile.
    # "v_min": float,
    "v_min_ratio": 0.1,  # 10 % safety margin

    # Optional. One of the following two. It determines initial fill volume.
    # If none are defined, the initial fill volume is the same as void volume.
    # "v_init": float,
    # "v_ratio": float,  # 20 % of the total (== void) volume

    # Optional. Default = False.
    # If True, then `v_init` and `v_init_ratio` are ignored.
    "starts_empty": True,

    # Optional. Initial buffer composition.
    # Default is empty array (equivalent to 0).
    # "c_add_buffer": np.ndarray,

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

cstr = CSTR(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(cstr, key), f"`{key}` is wrong."
    # Override value.
    setattr(cstr, key, value)

# Voila :)
