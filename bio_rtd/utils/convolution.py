__all__ = ['time_conv', 'piece_wise_conv_with_init_state']
__version__ = '0.2'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.logger as _logger
import bio_rtd.utils.vectors as _vectors


def time_conv(dt: float,
              c_in: _np.ndarray,
              rtd: _np.ndarray,
              c_equilibration: _typing.Optional[_np.ndarray] = None,
              logger: _typing.Optional[_logger.RtdLogger] = None) -> _np.ndarray:
    """
    Perform time convolution

    First time-point of `c_in` and 'c_rtd' is at t == 0 (and not at t == dt).
    Convolution is applied to all columns (= species) of the `c_in`

    Parameters
    ----------
    dt: float
        Time step.
    c_in: np.ndarray
        Starting concentration profile for each specie
    rtd
        Residence time distribution (= unit impulse response)
    c_equilibration
        Initial concentrations of species inside the void volume of the unit operation.
        E. g. the composition of equilibration buffer for flow-through column.
    logger
        Logger can be passed from unit operations

    Returns
    -------
    np.ndarray
        Final concentration profile for each specie
    """

    # it can happen that array is empty, then just return empty one
    if c_in.size == 0:
        if logger:
            logger.i("Convolution: Got empty c_in")
        return c_in.copy()
    if rtd.size == 0:
        if logger:
            logger.w("Convolution: Got empty bio_rtd")
        return c_in.copy()

    if c_equilibration is not None and _np.all(c_equilibration == 0):
        c_equilibration = None

    c_out = _np.zeros_like(c_in)

    # simulate pre-flushing and washout
    c_ext = c_in

    n_prepend = rtd.size if c_equilibration is not None else 0
    if c_equilibration is not None:
        c_ext = _np.pad(c_ext, ((0, 0), (n_prepend, 0)), mode="constant")
        c_ext[:, :n_prepend] = c_equilibration

    # convolution
    for j in range(c_out.shape[0]):
        c_out[j] = _np.convolve(c_ext[j], rtd)[n_prepend:n_prepend + c_in.shape[1]] * dt

    return c_out


def piece_wise_conv_with_init_state(dt: float,
                                    f_in: _np.ndarray,
                                    c_in: _np.ndarray,
                                    t_cycle: float,
                                    rt_mean: float,
                                    rtd: _np.ndarray,
                                    c_equilibration: _typing.Optional[_np.ndarray] = None,
                                    c_wash: _typing.Optional[_np.ndarray] = None,
                                    logger: _typing.Optional[_logger.RtdLogger] = None) -> _np.ndarray:

    assert c_in.shape[1] == f_in.size
    assert t_cycle > 0
    assert rt_mean >= 0

    # it can happen that the input array is empty, then just return empty one
    if c_in.size == 0:
        if logger:
            logger.i("Convolution: Got empty c_in")
        return c_in.copy()

    if rtd.size == 0:
        if logger:
            logger.w("Convolution: Got empty bio_rtd")
        return c_in.copy()

    if f_in.sum() == 0:
        if logger:
            logger.i("Convolution: Got empty f_in")
        return _np.zeros_like(c_in)

    i_cycle = int(round(t_cycle / dt))
    i_rt_mean = int(round(rt_mean / dt))
    i_start, i_end = _vectors.true_start_and_end(f_in > 0)
    i_switch_inlet = _np.rint(_np.arange(i_start, i_end, t_cycle / dt)).astype(int)
    i_switch_inlet_off = _np.append(i_switch_inlet[1:], i_end)
    i_switch_outlet = (i_switch_inlet + i_rt_mean).clip(max=f_in.size)
    i_switch_outlet_off = _np.append(i_switch_outlet[1:], min(i_switch_outlet[-1] + i_cycle, f_in.size))

    c_out = _np.zeros_like(c_in)

    for i in range(i_switch_inlet.size):
        # inlet concentration profile for the cycle; prolonged by wash buffer
        c_conv_inlet = c_in[:, i_switch_inlet[i]:i_switch_outlet_off[i]].copy()
        c_conv_inlet[:, i_switch_inlet_off[i] - i_switch_inlet[i]:] = c_wash if c_wash is not None else 0

        # calculate outlet concentration profile
        c_conv_outlet = time_conv(dt, c_conv_inlet, rtd, c_equilibration, logger)

        # insert the result in the outlet vector
        c_out[:, i_switch_outlet[i]:i_switch_outlet_off[i]] = \
            c_conv_outlet[:, i_switch_outlet[i] - i_switch_inlet[i]:i_switch_outlet_off[i] - i_switch_inlet[i]]
    return c_out
