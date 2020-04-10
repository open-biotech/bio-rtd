"""Template for instantiating `Concentration` `UnitOperation`.

See Also
--------
:class:`bio_rtd.uo.fc_uo.Concentration`

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

from typing import List

import numpy as np

from bio_rtd.uo.fc_uo import Concentration

"""Direct instance creation."""

tmp_uo = Concentration(
    t=np.linspace(0, 10, 100),
    flow_reduction=8,  # f_out = f_in / flow_reduction
    uo_id="concentration",
    gui_title="Concentration, direct instance"  # Optional.
)
# Optional. Which species are non-retained (e.g. salt).
tmp_uo.non_retained_species = [2]
# Optional. Relative losses. Does not apply to `non_retained_species`.
tmp_uo.relative_losses = 0.05  # 5 % losses


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
    # Required. Example: `flow_reduction` = 4 -> f_out = f_in / 4.
    "flow_reduction": float,
    # Optional.
    "gui_title": str,  # default: Concentration
}

ATTRIBUTES = {
    # Optional. Which species are non-retained (e.g. salt).
    "non_retained_species": List,

    # Optional. Relative losses. Does not apply to `non_retained_species`.
    "relative_losses": float,

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
    "uo_id": "concentration",
    # Required. Example: `flow_reduction` = 4 -> f_out = f_in / 4.
    "flow_reduction": 8,
    # Optional.
    # "gui_title": str,  # default: Concentration
}

uo_attr = {
    # Optional. Which species are non-retained (e.g. salt).
    "non_retained_species": [2],

    # Optional. Relative losses. Does not apply to `non_retained_species`.
    "relative_losses": 0.05,

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

concentration = Concentration(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(concentration, key), f"`{key}` is wrong."
    # Override value.
    setattr(concentration, key, value)

# Voila :)
