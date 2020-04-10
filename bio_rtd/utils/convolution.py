"""Helper functions for performing convolution on time axis."""

__all__ = ['time_conv', 'piece_wise_time_conv']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.logger as _logger
import bio_rtd.utils.vectors as _vectors


def time_conv(dt: float,
              c_in: _np.ndarray,
              rtd: _np.ndarray,
              c_equilibration: _typing.Optional[_np.ndarray] = None,
              logger: _typing.Optional[_logger.RtdLogger] = None
              ) -> _np.ndarray:
    """Perform convolution on time axis.

    First time-point of `c_in` and `c_rtd` is at t == 0 (and not `dt`).

    Convolution is applied to all species of `c_in`.

    Parameters
    ----------
    dt
        Time step.
    c_in
        Starting concentration profile for each specie.

        `c_in`.shape == [n_species, n_time_steps]
    rtd
        Residence time distribution (= unit impulse response).
    c_equilibration
        Initial concentrations inside the unit operation.

        E.g.: Composition of equilibration buffer for flow-through
        chromatography.
    logger
        Logger for messaging events.

    Returns
    -------
    c_out: ndarray
        Final concentration profile for each specie.

        `c_out`.shape == `c_in`.shape

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
        c_out[j] = _np.convolve(c_ext[j], rtd)[
                   n_prepend:n_prepend + c_in.shape[1]] * dt

    return c_out


def piece_wise_time_conv(
        dt: float,
        f_in: _np.ndarray,
        c_in: _np.ndarray,
        t_cycle: float,
        rt_mean: float,
        rtd: _np.ndarray,
        c_equilibration: _typing.Optional[_np.ndarray] = None,
        c_wash: _typing.Optional[_np.ndarray] = None,
        logger: _typing.Optional[_logger.RtdLogger] = None) -> _np.ndarray:
    """Perform convolution on time axis with periodic switching.

    First time-point of `c_in` and `c_rtd` is at t == 0 (and not `dt`).

    Convolution is applied to all species of `c_in`.

    Parameters
    ----------
    dt
        Time step.
    f_in
        Flow rate profile. It has to be either constant or box-shaped.
    c_in
        Starting concentration profile for each specie.

        `c_in`.shape == [n_species, n_time_steps]
    t_cycle
        Switch cycle duration.
    rt_mean
        Delay between inlet and outlet switch times.
    rtd
        Residence time distribution (= unit impulse response).
    c_equilibration
        Composition of equilibration buffer.
    c_wash
        Composition of wash buffer.
    logger
        Logger for messaging events.

    Returns
    -------
    c_out: ndarray
        Final concentration profile for each specie.

        `c_out`.shape == `c_in`.shape

    """

    assert c_in.shape[1] == f_in.size
    assert t_cycle > 0
    assert rt_mean >= 0

    # If input array is empty, then return empty.
    if c_in.size == 0:
        if logger:
            logger.i("Convolution: Got empty c_in")
        return c_in.copy()
    elif rtd.size == 0:
        if logger:
            logger.w("Convolution: Got empty bio_rtd")
        return c_in.copy()
    elif f_in.sum() == 0:
        if logger:
            logger.i("Convolution: Got empty f_in")
        return _np.zeros_like(c_in)

    i_cycle = int(round(t_cycle / dt))
    i_rt_mean = int(round(rt_mean / dt))
    i_start, i_end = _vectors.true_start_and_end(f_in > 0)
    assert _np.all(f_in[i_start:i_end] == f_in.max()), \
        "Flow rate profile must be boxed shaped"
    i_switch_inlet = _np.rint(
        _np.arange(i_start, i_end, t_cycle / dt)
    ).astype(int)
    i_switch_inlet_off = _np.append(i_switch_inlet[1:], i_end)
    i_switch_outlet = (i_switch_inlet + i_rt_mean).clip(max=f_in.size)
    i_switch_outlet_off = _np.append(
        i_switch_outlet[1:],
        min(i_switch_outlet[-1] + i_cycle, f_in.size)
    )

    c_out = _np.zeros_like(c_in)

    for i in range(i_switch_inlet.size):
        # Inlet concentration profile for the cycle.
        # Profile is prolonged by wash buffer.
        c_conv_inlet = c_in[:, i_switch_inlet[i]:i_switch_outlet_off[i]].copy()
        c_conv_inlet[:, i_switch_inlet_off[i] - i_switch_inlet[i]:] = \
            c_wash if c_wash is not None else 0

        # Calculate outlet concentration profile.
        c_conv_outlet = time_conv(dt, c_conv_inlet, rtd,
                                  c_equilibration, logger)

        # Insert the result into outlet vector.
        c_out[:, i_switch_outlet[i]:i_switch_outlet_off[i]] = c_conv_outlet[
            :,
            i_switch_outlet[i] - i_switch_inlet[i]:
            i_switch_outlet_off[i] - i_switch_inlet[i]]
    return c_out
