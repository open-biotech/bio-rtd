"""Template for instantiating ComboUO, joining multiple UnitOperations

`ComboUO` accepts list of individual unit operations and represents
them as a single unit operation.

Logger set in (or to) `ComboUO` is propagated to children.

See Also
--------
:class:`bio_rtd.uo.special_uo.ComboUO`

Notes
-----
`ComboUO` and nested unit operations should have the same time vector.

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

from typing import List

import numpy as np

from bio_rtd.core import UnitOperation
from examples.templates.fc_uo import concentration_template
from examples.templates.fc_uo import buffer_exchange_template
from examples.templates.fc_uo import flow_through_template

from bio_rtd.uo.special_uo import ComboUO

"""Direct instance creation."""

tmp_uo = ComboUO(
    # All unit operations should have the same time vector.
    t=np.linspace(0, 100, 1001),
    sub_uo_list=[concentration_template.concentration,
                 buffer_exchange_template.buffer_exchange,
                 flow_through_template.flow_through],
    uo_id="combo_uo",
    gui_title="Combo UO, direct instance"  # Optional.
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
    # Required.
    "sub_uo_list": List[UnitOperation],
    # Optional.
    "gui_title": str,  # default: ComboUO
}

ATTRIBUTES = {
    # `ComboUO` has no specific attributes,
    # apart from the ones in `PARAMETERS`.

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
    "uo_id": "uf_df",
    # Required.
    "sub_uo_list": [concentration_template.concentration,
                    buffer_exchange_template.buffer_exchange,
                    flow_through_template.flow_through],
    # Optional.
    # "gui_title": str,  # default: ComboUO
}

uo_attr = {
    # `ComboUO` has no specific attributes,
    # apart from the ones in `PARAMETERS`.

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}

"""3. Instantiate unit operation and populate attributes."""

cstr = ComboUO(t, **uo_pars)

# Can be omitted.
for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(cstr, key), f"`{key}` is wrong."
    # Override value.
    setattr(cstr, key, value)

# Voila :)
