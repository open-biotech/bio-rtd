"""Module with inlet profiles.

Inlet profiles defined in this module are subclasses of
`bio_rtd.core.Inlet`.
They define the starting flow rate and concentration profiles for the
DSP unit operation train.

"""

__version__ = '0.7'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

from bio_rtd.core import Inlet as _Inlet


class ConstantInlet(_Inlet):
    """Constant flow rate and constant process fluid composition

    Parameters
    ----------
    t
        Simulation time vector
    f
        Constant flow rate.
    c
        Constant concentration for each specie.
        For single specie use `np.array([c_value])`.

    """

    def __init__(self, t: _np.array, f: float, c: _np.ndarray,
                 species_list: _typing.Sequence[str], inlet_id: str,
                 gui_title: str = "ConstantInlet"):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f = f
        self.c = c.copy()
        self._refresh()

    def _refresh(self):
        """Recalculate profiles based on attributes."""
        self._f_out = _np.ones_like(self._t) * self.f
        assert len(self.c.shape) == 1, \
            "c_init should be 1D vector with concentration for each specie"
        assert self.c.shape[0] == self._n_species
        self._c_out = \
            _np.ones([self._n_species, self._t.size]) * self.c[:, _np.newaxis]


class IntervalInlet(_Inlet):
    """Constant flow rate profile and box shaped concentration profile

    Attributes
    ----------
    t_start
        Start position of box, inclusive ( >= ).
        Initial == t[0] == 0
    t_end
        End position of box, excluding ( < ).
        Initial == t[-1] + t[1]
    c_inner : np.ndarray
        Concentrations inside the box.
    c_outer : np.ndarray
        Concentrations outside the box.

    """

    def __init__(self, t: _np.array, f: float,
                 c_inner: _np.ndarray, c_outer: _np.ndarray,
                 species_list: _typing.Sequence[str],
                 inlet_id: str, gui_title: str):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f = f
        self.c_inner = c_inner.copy()
        self.c_outer = c_outer.copy()
        # Box limits.
        self.t_start = t[0]
        self.t_end = t[-1] + t[1]  # extra step as t_end is not inclusive

        self._refresh()

    def _refresh(self):
        """Recalculate profiles based on attributes."""
        self._f_out = _np.ones_like(self._t) * self.f
        self._c_out = _np.zeros_like(self._c_out)
        # Ensure proper dimensions.
        assert self.c_inner.size == self._n_species
        assert self.c_outer.size == self._n_species
        # Apply box shape to concentration profiles.
        self._c_out[:, (self._t >= self.t_start) * (self._t < self.t_end)] \
            = self.c_inner[:, _np.newaxis]
        self._c_out[:, (self._t < self.t_start) + (self._t >= self.t_end)] \
            = self.c_outer[:, _np.newaxis]


class CustomInlet(_Inlet):
    """Custom flow rate profile and concentration profiles

    Attributes
    ----------
    t
        Simulation time vector.
    f
        Custom flow rate profile. Should have the same size az `t`.
    c
        Custom concentration profiles.
        `c.shape == (len(species_list), t.size)`

    """
    def __init__(self, t: _np.ndarray, f: _np.ndarray, c: _np.ndarray,
                 species_list: _typing.Sequence[str],
                 inlet_id: str, gui_title: str):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f = f
        self.c = c
        self._refresh()

    def _refresh(self):
        """Recalculate profiles based on attributes."""
        assert self.f.shape == self._t.shape
        assert self.c.shape == (self._n_species, self._t.size)
        self._f_out = self.f.copy()
        self._c_out = self.c.copy()
