"""Template for instantiating Dilution UnitOperation.

For more details see docstrings of `Dilution`.

"""

__version__ = '0.3.0'
__author__ = 'Jure Sencar'

import numpy as np

from bio_rtd.uo.fc_uo import Dilution

"""Direct instance creation."""

tmp_uo = Dilution(
    t=np.linspace(0, 10, 100),
    dilution_ratio=1.6,  # 60 % addition of dilution buffer
    uo_id="dilution_direct",
    gui_title="Dilution, direct instance"  # Optional.
)
# Optional. Dilution buffer composition.
tmp_uo.c_add_buffer = np.array([0, 0, 200])


"""Using dict of parameters and attributes.

-----
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
    # Required.
    # 1 = no dilution, 1.6 = 60 % addition of dilution buffer.
    "dilution_ratio": float,

    # Optional.
    "gui_title": str,  # default: Dilution
}

ATTRIBUTES = {
    # Optional. Composition of dilution buffer.
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
    "uo_id": "dilution_1",
    # Required.
    # 1 = no dilution, 1.6 = 20 % addition of dilution buffer.
    "dilution_ratio": 1.6,

    # Optional.
    # "gui_title": str,  # default: Dilution
}

uo_attr = {
    # Optional. Composition of dilution buffer.
    # Default is empty array (equivalent to 0).
    "c_add_buffer": np.array([0, 0, 200]),

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

dilution = Dilution(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(dilution, key), f"`{key}` is wrong."
    # Override value.
    setattr(dilution, key, value)

# Voila :)
