"""Wrappers for probability distribution functions.

This module contains subclasses of :class:`bio_rtd.core.PDF`.

The integral of a probability distribution function (PDF) over time has
a value of 1.

Examples
--------
>>> t = _np.linspace(0, 100, 1001)
>>> dt = t[1]
>>> pdf = GaussianFixedDispersion(t, 0.2)
>>> pdf.trim_and_normalize = False
>>> pdf.update_pdf(rt_mean=40)
>>> p = pdf.get_p()
>>> print(round(p.sum() * dt, 8))
1.0
>>> t[p.argmax()]
40.0

"""

__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import numpy as _np

from bio_rtd.core import PDF as _PDF
from bio_rtd import peak_shapes as _peak_shapes


class GaussianFixedDispersion(_PDF):
    """Gaussian PDF with fixed dispersion.

    Parameters
    ----------
    dispersion_index
        Dispersion index.

        Dispersion index is defined as `sigma * sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation on time scale (not volume).
    cutoff
        Cutoff limit for trimming front and end tailing.

        Cutoff limit is relative to the peak max value.
    pdf_id
        Unique identifier. Default = "GaussianFixedDispersion".

    See Also
    --------
    :class:`rtd_lib.core.PDF`

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> pdf = GaussianFixedDispersion(t, dispersion_index=0.2)
    >>> pdf.trim_and_normalize = False
    >>> pdf.update_pdf(rt_mean=40)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 4))
    1.0
    >>> t[p.argmax()]
    40.0

    """

    POSSIBLE_KEY_GROUPS = [['f', 'v_void'], ['rt_mean']]
    OPTIONAL_KEYS = []

    def __init__(self, t: _np.ndarray,
                 dispersion_index: float,
                 cutoff=0.0001,
                 pdf_id: str = "GaussianFixedDispersion"):
        super().__init__(t, pdf_id)
        self.dispersion_index = dispersion_index
        """Dispersion index.

        Dispersion index is defined as `sigma * sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
        
        """
        self.cutoff_relative_to_max = cutoff
        """Cutoff limit for trimming front and end tailing.

        Cutoff limit is relative to the peak max value.
        
        """

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
            i_max = self._t_steps_max
        t = _np.arange(i_max) * self._dt
        p = _peak_shapes.gaussian(t, rt_mean, sigma, self.log)

        return p


class GaussianFixedRelativeWidth(_PDF):
    """Gaussian PDF with fixed relative peak width.

    Parameters
    ----------
    relative_sigma
        Relative sigma.

        Relative sigma is defined as `sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
    cutoff
        Cutoff limit for trimming front and end tailing.

        Cutoff limit is relative to the peak max value.
    pdf_id
        Unique identifier. Default = "GaussianFixedRelativeWidth".

    See Also
    --------
    :class:`rtd_lib.core.PDF`

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> pdf = GaussianFixedRelativeWidth(t, relative_sigma=0.15)
    >>> pdf.trim_and_normalize = False
    >>> pdf.update_pdf(rt_mean=40)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 4))
    1.0
    >>> t[p.argmax()]
    40.0

    """

    POSSIBLE_KEY_GROUPS = [['f', 'v_void'], ['rt_mean']]
    OPTIONAL_KEYS = []

    def __init__(self, t: _np.array,
                 relative_sigma: float,
                 cutoff=0.0001,
                 pdf_id: str = "GaussianFixedRelativeWidth"):
        super().__init__(t, pdf_id)
        self.relative_sigma = relative_sigma
        """Relative sigma.

        Relative sigma is defined as `sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
        
        """
        self.cutoff_relative_to_max = cutoff
        """Cutoff limit for trimming front and end tailing.

        Cutoff limit is relative to the peak max value.
        
        """

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:
        # Get `rt_mean` from parameters.
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() \
            else kw_pars['v_void'] / kw_pars['f']
        # Calc `sigma`.
        sigma = rt_mean * self.relative_sigma
        # Calc probability distribution (`p`).
        t_max = sigma * _np.sqrt(-2 * _np.log(self.cutoff_relative_to_max)) \
            + rt_mean
        t_max = min(_np.ceil(t_max / self._dt), self._t_steps_max) * self._dt
        p = _peak_shapes.gaussian(
            _np.arange(0, t_max, self._dt), rt_mean, sigma, self.log)
        return p


class ExpModGaussianFixedDispersion(_PDF):
    """Exponentially Modified Gaussian PDF with fixed dispersion.

    Parameters
    ----------
    dispersion_index
        Dispersion index of Gaussian part.

        Dispersion index is defined as `sigma * sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation on time scale (not volume).
    skew
        Rate of exponential part.
    pdf_id
        Unique identifier. Default = "ExpModGaussianFixedDispersion".

    See Also
    --------
    :class:`rtd_lib.core.PDF`

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> dispersion_index = 0.2
    >>> skew = 0.5
    >>> pdf = ExpModGaussianFixedDispersion(t, dispersion_index, skew)
    >>> pdf.trim_and_normalize = False
    >>> pdf.update_pdf(rt_mean=40)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 8))
    1.0
    >>> t[p.argmax()]  # position of peak max
    39.6
    >>> print(round((p * t[:p.size]).sum() * dt, 3))  # 1st momentum
    40.0

    """

    POSSIBLE_KEY_GROUPS = [['f', 'v_void'], ['rt_mean']]
    OPTIONAL_KEYS = ['skew']

    def __init__(self, t: _np.array,
                 dispersion_index: float,
                 skew: float,
                 pdf_id: str = "ExpModGaussianFixedDispersion"):
        super().__init__(t, pdf_id)
        self.dispersion_index = dispersion_index
        """Dispersion index for Gaussian part.

        Dispersion index is defined as `sigma * sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
        
        """
        self.skew = skew
        """Rate of exponential part."""

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:
        # Get `rt_mean` from parameters.
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() \
            else kw_pars['v_void'] / kw_pars['f']
        # Get `skew` from parameters if provided.
        skew = kw_pars['skew'] if 'skew' in kw_pars.keys() else self.skew
        # Calc `sigma`.
        sigma = (rt_mean * self.dispersion_index) ** 0.5
        # Calc probability distribution (`p`).
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.emg(t, rt_mean, sigma, skew, self.log)
        return p


class ExpModGaussianFixedRelativeWidth(_PDF):
    """Exponentially Modified Gaussian PDF with fixed relative sigma.

    Parameters
    ----------
    sigma_relative
        Relative sigma for Gaussian part.

        Relative sigma is defined as `sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
    tau_relative
        Relative characteristic time of exponential part.

        It is defined as 1 / (`skew` * `rt_mean`).
    pdf_id
        Unique identifier. Default = "ExpModGaussianFixedRelativeWidth".

    See Also
    --------
    :class:`rtd_lib.core.PDF`

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> sigma_relative = 0.15
    >>> skew = 0.5
    >>> pdf = ExpModGaussianFixedDispersion(t, sigma_relative, skew)
    >>> pdf.trim_and_normalize = False
    >>> pdf.update_pdf(rt_mean=40)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 8))
    1.0
    >>> t[p.argmax()]  # position of peak max
    39.5
    >>> print(round((p * t[:p.size]).sum() * dt, 2))  # 1st momentum
    40.0

    """

    POSSIBLE_KEY_GROUPS = [['rt_mean'], ['f', 'v_void']]
    OPTIONAL_KEYS = ['skew']

    def __init__(self, t: _np.array,
                 sigma_relative: float,
                 tau_relative: float,
                 pdf_id: str = "ExpModGaussianFixedRelativeWidth"):
        super().__init__(t, pdf_id)
        self.sigma_relative = sigma_relative
        """Relative sigma for Gaussian part.

        Relative sigma is defined as `sigma / rt_mean`. Where
        `rt_mean` is a mean residence time and `sigma` is a
        standard deviation.
        
        """
        self.tau_relative = tau_relative
        """Relative characteristic time of exponential part.

        Relative characteristic time is defined as 
        1 / (`skew` * `rt_mean`). Where `rt_mean` is a mean residence
        time and `skew` is the rate of the exponential part.
        
        """

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:
        # Get `rt_mean` from parameters.
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() \
            else kw_pars['v_void'] / kw_pars['f']
        # Get `skew` from parameters if provided.
        skew = kw_pars['skew'] if 'skew' in kw_pars.keys() \
            else 1 / self.tau_relative / rt_mean
        # Calc `sigma`.
        sigma = rt_mean * self.sigma_relative
        # Calc probability distribution (`p`).
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.emg(t, rt_mean, sigma, skew, self.log)
        return p


class TanksInSeries(_PDF):
    """Tanks in series PDF.

    `rt_mean` means flow-through time through entire unit operation
    (all tanks).

    For `n_tanks` == 1, the distribution becomes exponential drop.

    Parameters
    ----------
    n_tanks
        Number of tanks.
    pdf_id
        Unique identifier. Default = "TanksInSeries".

    See Also
    --------
    :class:`rtd_lib.core.PDF`

    Examples
    -------
    >>> t = _np.linspace(0, 100, 1001)
    >>> dt = t[1]
    >>> sigma_relative = 0.15
    >>> skew = 0.5
    >>> pdf = TanksInSeries(t, n_tanks=5)
    >>> pdf.update_pdf(rt_mean=10)
    >>> pdf.trim_and_normalize = False
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 8))
    1.0
    >>> t[p.argmax()]  # position of peak max
    8.0
    >>> print(round((p * t[:p.size]).sum() * dt, 2))  # 1st momentum
    10.0
    >>> pdf = TanksInSeries(t, n_tanks=1)
    >>> pdf.trim_and_normalize = False
    >>> pdf.update_pdf(rt_mean=10)
    >>> p = pdf.get_p()
    >>> print(round(p.sum() * dt, 2))
    1.0
    >>> t[p.argmax()]  # position of peak max
    0.0
    >>> print(round((p * t[:p.size]).sum() * dt, 1))  # 1st momentum
    10.0

    """

    POSSIBLE_KEY_GROUPS = [['rt_mean'], ['f', 'v_void']]
    OPTIONAL_KEYS = ['n_tanks']

    def __init__(self, t: _np.array,
                 n_tanks: float,
                 pdf_id: str = "TanksInSeries"):
        super().__init__(t, pdf_id)
        self.n_tanks = n_tanks
        """Number of tanks."""
        self.allow_open_end: bool = False
        """Prevent warnings and errors if the pdf does not fit on `t`.
        
        Default: `False`
        
        If `True`, no warnings or errors are reported in case the
        distribution does not fit on provided  time vector.
        
        """

    def _calc_pdf(self, kw_pars: dict) -> _np.ndarray:
        # Get `rt_mean` from parameters.
        rt_mean = kw_pars['rt_mean'] if 'rt_mean' in kw_pars.keys() \
            else kw_pars['v_void'] / kw_pars['f']
        # Get`n_tanks` parameter if provided.
        n_tanks = kw_pars['n_tanks'] if 'n_tanks' in kw_pars.keys() \
            else self.n_tanks
        # Calc probability distribution (`p`).
        t = _np.arange(0, self._t_steps_max * self._dt, self._dt)
        p = _peak_shapes.tanks_in_series(t, rt_mean, n_tanks, self.log,
                                         self.allow_open_end)
        return p
