"""Module with inlet profiles.

Inlet profiles defined in this module are subclasses of
:class:`bio_rtd.core.Inlet`.

They define starting flow rate and concentration profiles for a
unit operation train.

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

from bio_rtd.core import Inlet as _Inlet


class ConstantInlet(_Inlet):
    """Constant flow rate and constant process fluid composition

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    f
        Constant flow rate.
    c
        Constant concentration for each specie.
        For single specie use `np.array([c_value])`.
    species_list
        List with names of simulating process fluid species.
    inlet_id
        Unique identifier of an instance. It is stored in :attr:`uo_id`.
    gui_title
        Readable title of an instance. Default = "ConstantInlet".

    """

    def __init__(self, t: _np.array, f: float, c: _np.ndarray,
                 species_list: _typing.Sequence[str], inlet_id: str,
                 gui_title: str = "ConstantInlet"):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f: float = f
        """Constant flow rate."""
        self.c: _np.ndarray = c.copy()
        """Constant concentration value for each specie."""
        self.refresh()

    def refresh(self):
        """Recalculate profiles based on attributes."""
        self._f_out = _np.ones_like(self._t) * self.f
        assert len(self.c.shape) == 1, \
            f"`c_init` should be 1D vector with concentration for each specie."
        assert self.c.shape[0] == self._n_species
        self._c_out = \
            _np.ones([self._n_species, self._t.size]) * self.c[:, _np.newaxis]


class IntervalInlet(_Inlet):
    """Constant flow rate and box shaped concentration profile.

    Parameters
    ----------
    t
        Simulation time vector.
        Starts with 0 and has a constant time step.
    f
        Constant flow rate.
    c_inner
        Constant concentration for each specie inside the interval.
        For single specie use `np.array([c_value])`.
    c_outer
        Constant concentration for each specie outside the interval.
        For single specie use `np.array([c_value])`.
    species_list
        List with names of simulating process fluid species.
    inlet_id
        Unique identifier of an instance. It is stored in :attr:`uo_id`.
    gui_title
        Readable title of an instance. Default = "ConstantInlet".

    Attributes
    ----------
    t_start
        Start of interval, inclusive ( >= ).
        Default = 0.
    t_end
        End of interval, excluding ( < ).
        Default = t[-1] + `time step`.
    c_inner
        Concentrations inside the interval.
    c_outer
        Concentrations outside the interval.

    """

    def __init__(self, t: _np.array, f: float,
                 c_inner: _np.ndarray, c_outer: _np.ndarray,
                 species_list: _typing.Sequence[str],
                 inlet_id: str, gui_title: str):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f = f
        """Constant flow rate."""
        self.c_inner: _np.ndarray = c_inner.copy()
        """Concentration value for each specie inside the interval."""
        self.c_outer: _np.ndarray = c_outer.copy()
        """Concentration value for each specie outside the interval."""
        self.t_start: float = t[0]
        """Start of interval, inclusive ( >= ). Default = 0."""
        self.t_end: float = t[-1] + self._dt
        """End of interval, excluding ( < ).
        
        Default = t[-1] + `time step`.
        
        """
        self.refresh()

    def refresh(self):
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
    """Custom flow rate profile and custom concentration profile.

    Parameters
    ----------
    t
        Simulation time vector.
    f
        Custom flow rate profile. Should have the same size as `t`.
    c
        Custom concentration profiles.
        `c`.shape == (len(:attr:`species_list`), `t`.size)
    species_list
        List with names of simulating process fluid species.
    inlet_id
        Unique identifier of an instance. It is stored in :attr:`uo_id`.
    gui_title
        Readable title of an instance. Default = "ConstantInlet".

    """
    def __init__(self, t: _np.ndarray, f: _np.ndarray, c: _np.ndarray,
                 species_list: _typing.Sequence[str],
                 inlet_id: str, gui_title: str):
        super().__init__(t, species_list, inlet_id, gui_title)
        self.f: _np.ndarray = f
        """Custom flow rate profile. Should be the same size as `t`."""
        self.c: _np.ndarray = c
        """Custom concentration profiles.
        
        `c`.shape == (len(:attr:`species_list`), `t`.size)
        
        """
        self.refresh()

    def refresh(self):
        """Recalculate profiles based on attributes."""
        assert self.f.shape == self._t.shape
        assert self.c.shape == (self._n_species, self._t.size)
        self._f_out = self.f.copy()
        self._c_out = self.c.copy()
