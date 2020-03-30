"""Template for instantiating BufferExchange UnitOperation.

For more details see docstrings of `BufferExchange`.

"""

__version__ = '0.3.0'
__author__ = 'Jure Sencar'

from typing import List

import numpy as np

from bio_rtd.uo.fc_uo import BufferExchange

"""Direct instance creation."""

tmp_uo = BufferExchange(
    t=np.linspace(0, 10, 100),
    exchange_ratio=0.95,
    uo_id="buffer_exchange",
    gui_title="BufferExchange, direct instance"  # Optional.
)
# Optional. Which species are non-retained (e.g. salt).
tmp_uo.non_retained_species = [2]
# Optional. Relative losses. Does not apply to `non_retained_species`.
tmp_uo.relative_losses = 0.05  # 5 % losses
# Optional. Exchange buffer composition. Default = empty array (= 0).
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
    "exchange_ratio": float,
    # Optional.
    "gui_title": str,  # default: BufferExchange
}

ATTRIBUTES = {
    # Optional. Which species are non-retained (e.g. salt).
    "non_retained_species": List,

    # Optional. Relative losses. Does not apply to `non_retained_species`.
    "relative_losses": float,

    # Optional. Composition of exchange buffer.
    # Default is empty array (equivalent to 0).
    "c_exchange_buffer": np.ndarray,

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
    "uo_id": "buffer_exchange",
    # Required.
    "exchange_ratio": 0.95,
    # Optional.
    # "gui_title": str,  # default: BufferExchange
}

uo_attr = {
    # Optional. Which species are non-retained (e.g. salt).
    "non_retained_species": [1],

    # Optional. Relative losses. Does not apply to `non_retained_species`.
    "relative_losses": 0.05,

    # Optional. Composition of exchange buffer.
    # Default is empty array (equivalent to 0).
    # "c_exchange_buffer": np.ndarray,

    # Additional attributes are inherited from `UnitOperation`.
    # See `examples/templates/add_on_attributes.py`.
    # Add them to the list if needed.
}


"""3. Instantiate unit operation and populate attributes."""

buffer_exchange = BufferExchange(t, **uo_pars)

for key, value in uo_attr.items():
    # Make sure attribute exist.
    assert hasattr(buffer_exchange, key), f"`{key}` is wrong."
    # Override value.
    setattr(buffer_exchange, key, value)

# Voila :)
