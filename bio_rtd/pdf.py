"""Wrapping classes for probability distribution functions

This module contains subclasses of `bio_rtd.core.PDF`.

The integral of a probability distribution function (PDF) over time has
a value of 1.

Examples
--------
>>> t = _np.linspace(0, 100, 1001)
>>> dt = t[1]
>>> pdf = GaussianFixedDispersion(t, 0.2)
>>> pdf.update_pdf(rt_mean=40)
>>> p = pdf.get_p()
>>> print(round(p.sum() * dt, 8))
1.0
>>> t[p.argmax()]
40.0

"""

__version__ = '0.7'
__author__ = 'Jure Sencar'

import numpy as _np

import bio_rtd.peak_shapes as _peak_shapes

from bio_rtd.core import PDF as _PDF


class GaussianFixedDispersion(_PDF):
    """Gaussian PDF with fixed dispersion

    Parameters
    ----------
    dispersion_index : float
        Dispersion index, defined as `sigma * sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
    cutoff : float
        Cutoff limit for trimming front and end tailing.
        Cutoff limit is relative to the peak max value.

    See Also
    --------
    The documentation of superclass (`rtd_lib.core.PDF`).

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> pdf = GaussianFixedDispersion(t, dispersion_index=0.2)
    >>> pdf.update_pdf(rt_mean=40)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 8))
    1.0
    >>> t[p.argmax()]
    40.0

    """

    # info about what key-value argument combinations could be passed into `update_pdf()`
    _possible_key_groups = [['f', 'v_void'], ['rt_mean']]
    _optional_keys = []

    def __init__(self, t: _np.ndarray,
                 dispersion_index: float,
                 cutoff=0.0001,
                 pdf_id: str = "GaussianFixedDispersion"):
        super().__init__(t, pdf_id)
        self.dispersion_index = dispersion_index
        self.cutoff_relative_to_max = cutoff

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:
        # Get rt_mean from parameters.
        rt_mean = kw_pars['rt_mean'] \
            if 'rt_mean' in kw_pars.keys() \
            else kw_pars['v_void'] / kw_pars['f']
        sigma = (rt_mean * self.dispersion_index) ** 0.5
        # Calc probability distribution `p`.
        if self.trim_and_normalize:
            t_max = rt_mean + sigma * \
                    _np.sqrt(-2 * _np.log(self.cutoff_relative_to_max))
            i_max = min(_np.ceil(t_max / self._dt) + 1, self._t_steps_max)
        else:
            i_max = self._t_steps_max * self._dt
        t = _np.arange(i_max) * self._dt
        p = _peak_shapes.gaussian(t, rt_mean, sigma)

        return p


class GaussianFixedRelativeWidth(_PDF):
    """
    Gaussian PDF with fixed relative peak width

    relative_sigma = sigma / rt_mean
    """

    # info about what key-value argument combinations could be passed into `update_pdf()`
    _possible_key_groups = [['f', 'v_void'], ['rt_mean']]
    _optional_keys = []
    # dispersion index

    def __init__(self, t: _np.array, relative_sigma: float, cutoff=0.0001, pdf_id: str = ""):
        super().__init__(t, pdf_id)
        self.relative_sigma = relative_sigma
        self.cutoff_relative_to_max = cutoff

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:

        # get rt_mean from parameters
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() else kw_pars['v_void'] / kw_pars['f']

        # calc sigma
        sigma = rt_mean * self.relative_sigma

        # calc pdf
        t_max = sigma * _np.sqrt(-2 * _np.log(self.cutoff_relative_to_max)) + rt_mean
        t_max = min(_np.ceil(t_max / self._dt), self._t_steps_max) * self._dt
        p = _peak_shapes.gaussian(_np.arange(0, t_max, self._dt), rt_mean, sigma)

        return p


class ExpModGaussianFixedDispersion(_PDF):
    """
    Exponentially Modified Gaussian PDF with fixed dispersion

    `dispersion_index = sigma**2 / rt_mean`
    `pdf(rt_mean) = calc_emg(t, rt_mean, sigma, skew)`

    Methods
    -------
    set_pdf_pars(dispersion_index: float, skew: float)
        Sets the dispersion index and skew factor

    Abstract Methods
    ----------------
    update_pdf(**kwargs)
        Calculate new pdf for a given set of parameters.
        `pdf_update_keys` contains groups of possible parameter combinations.
    _possible_key_groups : Sequence[Sequence[str]]
        List of keys for key-value argument combinations that could be specified to `update_pdf(**kwarg)` function
    """

    # info about what key-value argument combinations could be passed into `update_pdf()`
    _possible_key_groups = [['f', 'v_void'], ['rt_mean']]
    _optional_keys = ['skew']

    def __init__(self, t: _np.array, dispersion_index: float, skew: float, pdf_id: str = ""):
        super().__init__(t, pdf_id)
        self.dispersion_index = dispersion_index
        self.skew = skew

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:

        # get rt_mean from parameters
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() else kw_pars['v_void'] / kw_pars['f']

        # get skew from parameters if provided
        skew = kw_pars['skew'] if 'skew' in kw_pars.keys() else self.skew

        # calc sigma
        sigma = (rt_mean * self.dispersion_index) ** 0.5

        # calc pdf
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.emg(t, rt_mean, sigma, skew)

        return p


class ExpModGaussianFixedRelativeWidth(_PDF):
    """
    Exponentially Modified Gaussian PDF with fixed relative peak width
    """

    # info about what key-value argument combinations could be passed into `update_pdf()`
    _possible_key_groups = [['rt_mean'], ['f', 'v_void']]
    _optional_keys = ['skew']

    def __init__(self, t: _np.array, sigma_relative: float, skew: float, pdf_id: str = ""):
        super().__init__(t, pdf_id)
        self.sigma_relative = sigma_relative
        self.skew = skew

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:

        # get rt_mean from parameters
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() else kw_pars['v_void'] / kw_pars['f']

        # get skew from parameters if provided
        skew = kw_pars['skew'] if 'skew' in kw_pars.keys() else self.skew

        # calc sigma
        sigma = rt_mean * self.sigma_relative

        # calc pdf
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.emg(t, rt_mean, sigma, skew)

        return p


class TanksInSeries(_PDF):
    """
    Tanks in series PDF

    `rt_mean` means flow-through time through entire unit operation (all tanks)

    Methods
    -------
    set_pdf_pars(n_tanks: float)
        Sets number of tanks

    Abstract Methods
    ----------------
    update_pdf(**kwargs)
        Calculate new pdf for a given set of parameters.
        `pdf_update_keys` contains groups of possible parameter combinations.
    _possible_key_groups : Sequence[Sequence[str]]
        List of keys for key-value argument combinations that could be specified to `update_pdf(**kwarg)` function
    """

    _possible_key_groups = [['rt_mean'], ['f', 'v_void']]
    _optional_keys = ['n_tanks']

    def __init__(self, t: _np.array, n_tanks: float, pdf_id: str = ""):
        super().__init__(t, pdf_id)
        self.n_tanks = n_tanks

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:

        # get rt_mean from parameters
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() else kw_pars['v_void'] / kw_pars['f']

        # get skew n_tanks parameters if provided
        n_tanks = kw_pars['n_tanks'] if 'n_tanks' in kw_pars.keys() else self.n_tanks

        # calc pdf
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.tanks_in_series(t, rt_mean, n_tanks)

        return p
