"""Load breakthrough profiles.

Load breakthrough profiles determine what parts of load bind to the
column.

See Also
--------
:class:`bio_rtd.core.ChromatographyLoadBreakthrough`

"""

__all__ = ['ConstantPatternSolution']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'


import numpy as _np

from bio_rtd import core as _core
from bio_rtd.chromatography import bt_curve as _bt_curve


class ConstantPatternSolution(_core.ChromatographyLoadBreakthrough):
    """Breakthrough profile - Constant Pattern Solution.

    r = 1 / (1 + exp(`k` * (`dbc_100` - `m_load` / `cv`)))

    Parameters
    ----------
    dt
        Time step duration.
    dbc_100
        Dynamic binding capacity if the load would last indefinitely.
    k
        Steepness of the breakthrough profile.
    bt_profile_id
         Unique identifier of the PDF instance. Used for logs.

    Examples
    --------
    >>> t = _np.linspace(0, 120, 1001)  # min
    >>> dt = t[1]
    >>> btc = ConstantPatternSolution(dt, dbc_100=240, k=0.05)
    >>> btc.update_btc_parameters(cv=8.0)  # column volume [mL]
    >>> btc.get_total_bc()  # 8.0 [mL] * 240 [mg/mL] = 1920.0 [mg]
    1920.0
    >>> # Load.
    >>> f_in = _np.ones_like(t)  # mL / min
    >>> c_in = _np.ones([1, t.size]) * 15  # mg / mL
    >>> c_captured = btc.calc_c_bound(f_in, c_in)
    >>> c_captured.shape == c_in.shape
    True
    >>> c_captured
    array([[14.99990679, 14.99990574, 14.99990467, ..., 10.22437591,
            10.18768049, 10.15083683]])
    >>> round((c_captured * f_in * dt).sum(), 1)  # < 1920.0
    1739.0


    """
    POSSIBLE_KEY_GROUPS = [['cv']]
    OPTIONAL_KEYS = []

    def __init__(self, dt, dbc_100: float, k: float,
                 bt_profile_id: str = "ConstantPatternSolution"):
        super().__init__(dt, bt_profile_id)
        self.dbc_100 = dbc_100
        """Dynamic binding capacity at 100 % load breakthrough."""
        self.k = k
        """Steepness of the breakthrough profile."""
        # placeholder for column volume
        self._cv = -1

    def _update_btc_parameters(self, kw_pars: dict) -> None:

        self._cv = kw_pars["cv"]

    def _calc_unbound_to_load_ratio(
            self,
            m_cumulative_load: _np.ndarray) -> _np.ndarray:

        assert self._cv > 0, f"CV must be defined by now." \
                             f" Make sure `update_btc_parameters`" \
                             f" function was called before."

        ratio = _bt_curve.btc_constant_pattern_solution(m_cumulative_load,
                                                        self.dbc_100,
                                                        self.k,
                                                        self._cv,
                                                        self.log)

        return ratio

    def get_total_bc(self) -> float:
        assert self._cv > 0
        return self.dbc_100 * self._cv
