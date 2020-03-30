"""
Load breakthrough profiles

Classes in this file extend `_core.ChromatographyLoadBreakthrough` abstract classes.
If the class depends on specific curve shape, the curve shape must be defined in `./bt_curve` package
"""

__all__ = ['ConstantPatternSolution']
__version__ = '0.2'
__author__ = 'Jure Sencar'


import numpy as _np

import bio_rtd.core as _core

from . import bt_curve as _bt_curve


class ConstantPatternSolution(_core.ChromatographyLoadBreakthrough):
    _possible_key_groups = [['cv']]
    _optional_keys = []

    def __init__(self, dt, dbc_100: float, k: float, bt_profile_id: str = ""):
        super().__init__(dt, bt_profile_id)
        self.dbc_100 = dbc_100
        self.k = k
        # placeholder for column volume
        self.cv = -1

    def _update_btc_parameters(self, kw_pars: dict) -> None:

        self.cv = kw_pars["cv"]

    def _calc_unbound_to_load_ratio(self, m_cumulative_load: _np.ndarray) -> _np.ndarray:

        assert self.cv > 0, "CV must be defined by now. Make sure `update_btc_parameters` function was called before."

        ratio = _bt_curve.btc_constant_pattern_solution(m_cumulative_load, self.dbc_100, self.k, self.cv, self.log)

        return ratio

    def get_total_bc(self) -> float:
        assert self.cv > 0
        return self.dbc_100 * self.cv
