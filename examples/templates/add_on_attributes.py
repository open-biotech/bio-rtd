"""Additional attribute list for unit operations.

Templates for individual unit operations can be found in
`examples/templates/fc_uo/`,
`examples/templates/sc_uo/`,
`examples/templates/surge_tank/` and
`examples/templates/special_uo/`.

This file only contains additional attributes that can be added to any
unit operation.

"""

__version__ = '0.3.0'
__author__ = 'Jure Sencar'

import numpy as np
from typing import List

from bio_rtd.logger import RtdLogger
from bio_rtd.adj_par import AdjustableParameter

ADD_ON_ATTRIBUTES = {
    # List of adjustable parameters exposed to the GUI.
    "adj_par_list": List[AdjustableParameter],

    # Hide plot of the unit operation (default False).
    "gui_hidden": bool,

    # Optional. One of next four.
    # Discard inlet until given time.
    "discard_inlet_until_t": float,
    # Discard inlet until given concentration is reached for each component.
    "discard_inlet_until_min_c": np.ndarray,
    # Discard inlet until specified concentration between inlet concentration
    # and steady-state inlet concentration is reached for each component.
    "discard_inlet_until_min_c_rel": np.ndarray,
    # Discard first n cycles of the periodic inlet flow rate profile.
    "discard_inlet_n_cycles": float,

    # Optional. One of next four.
    # Discard outlet until given time.
    "discard_outlet_until_t": float,
    # Discard outlet until given concentration is reached for each component.
    "discard_outlet_until_min_c": np.ndarray,
    # Discard outlet until specified concentration between outlet concentration
    # and steady-state outlet concentration is reached for each component.
    "discard_outlet_until_min_c_rel": np.ndarray,
    # Discard first n cycles of the periodic outlet flow rate profile.
    "discard_outlet_n_cycles": float,

    # Optional. Default = logger.DefaultLogger().
    # If the unit operation is a part of `RtdModel`,  then the logger
    # is inherited from `RtdModel`.
    "log": RtdLogger,
}
