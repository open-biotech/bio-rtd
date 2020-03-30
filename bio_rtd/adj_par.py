"""Module for defining adjustable attributes of the RtdModel.

Adjustable attributes are mainly used by GUI in order to offer
responsive parameter setting.

"""

__all__ = ['AdjParSlider', 'AdjParRange', 'AdjParBoolean',
           'AdjParBooleanMultiple', 'AdjustableParameter']
__version__ = '0.7'
__author__ = 'Jure Sencar'

import typing as _typing
from abc import ABC as _ABC


class AdjustableParameter(_ABC):
    """Adjustable parameter for a UnitOperation (or Inlet).

    The class does not provide any logic. It just provides a
    unified way to specify an adjustable attribute.

    Parameters
    ----------
    var_list
        A list of attributes of a `UnitOperation` affected by the
        `AdjustableParameter`.
    v_range
        Range of values.
        For boolean set None.
        For range specify [start, end, step].
        For list set list. If the list has two columns,
        the first one is the title and the second is the value.
    v_init
        Initial value. If None (default), the value is inherited from
        the initial attribute value (at the start of the runtime).
    scale_factor
        The scale factor to compensate for differences in units.
        E.g.: Setting time in [h] for attribute set in [min] requires
        a `scale_factor` of 60.
    par_name
        Attribute title (e.g. 'Initial delay [min]') or list of titles
        in case of multiple options (e.g. radio button group).
    gui_type
        Preferred method for setting parameter in gui parameters.
        If None (default) or the gui does not support the type,
        then the gui should decide the method based on `v_init` and/or
        `var_list`.
        Example values:
        'checkbox', 'range', 'slider', 'select', 'multi-select'

    Notes
    -----
    Technically the class relates to instance attributes
    rather than parameters. However, the term 'process parameters'
    is well established in biotech, hence the class name.

    """

    def __init__(self,
                 var_list: _typing.List[str],
                 v_range: _typing.Optional[_typing.Union[
                     _typing.Tuple[float, float, float],
                     _typing.Sequence[str],
                     _typing.Sequence[float],
                     _typing.Sequence[_typing.Tuple[str, str]]]] = None,
                 v_init: _typing.Optional[_typing.Union[
                     float,
                     bool,
                     _typing.Sequence[float]]] = None,
                 scale_factor: float = 1,
                 par_name: _typing.Optional[_typing.Union[
                     str,
                     _typing.Sequence[str]]] = None,
                 gui_type: _typing.Optional[str] = None):
        self.var_list = var_list
        self.v_range = v_range
        self.v_init = v_init
        self.scale_factor = scale_factor
        self.par_name = par_name
        self.gui_type = gui_type


class AdjParBoolean(AdjustableParameter):
    """A boolean variation of :class:`AdjustableParameter`

    Parameters
    ----------
    var
        Variable of a UnitOperation affected by the parameter.
    v_init or list of bool
        Initial value or list of initial values for each item in
        `var_list`.
    par_name or list of str
        List of parameter names. If defined,
        the shape of the should match `var_list`.

    """

    def __init__(self,
                 var: str,
                 v_init: _typing.Optional[bool] = None,
                 par_name: _typing.Optional[str] = None,
                 scale_factor: float = 1,
                 gui_type: _typing.Optional[str] = 'checkbox'):
        super().__init__([var],
                         v_init=v_init,
                         par_name=par_name,
                         scale_factor=scale_factor,
                         gui_type=gui_type)


class AdjParBooleanMultiple(AdjustableParameter):
    """A radio group variation of `AdjustableParameter`

    Parameters
    ----------
    var_list : list of str
        A list of attributes of a `UnitOperation` affected by the
        `AdjustableParameter`.
    v_init
        Initial value for each entry in `var_list`.
        If None (default), the value is inherited from the initial
        attribute values at the start of the runtime. If specified,
        the shape should match the shape of `var_list`.
    par_name : list of str
        Attribute title (e.g. 'Initial delay [min]') for each option.
        The shape should batch the shape of `var_list`.

    """

    def __init__(self,
                 var_list: _typing.List[str],
                 v_init: _typing.Optional[_typing.Sequence[bool]] = None,
                 par_name: _typing.Optional[_typing.Sequence[str]] = None,
                 scale_factor: float = 1,
                 gui_type: _typing.Optional[str] = 'radio_group'):
        if v_init is not None:
            assert len(v_init) == len(var_list)
        if par_name is not None:
            assert len(par_name) == len(var_list)
        super().__init__(var_list=var_list,
                         v_init=v_init,
                         par_name=par_name,
                         scale_factor=scale_factor,
                         gui_type=gui_type)


class AdjParSlider(AdjustableParameter):
    """A value slider version of `AdjustableParameter`

    Parameters
    ----------
    var
        Variable of a UnitOperation affected by the parameter.
    v_init or Sequence[bool]
    v_range or (float, float) or (float, float, float)
        Defines range end or [start, end] or [start, end, step].
        If step is not defined, the `step = (start - end) / 10`.
        If start is not defined the `start = 0`.

    Notes
    -----
    For more info see docstring of `AdjustableParameter` superclass.

    """

    def __init__(self,
                 var: str,
                 v_range: _typing.Union[float,
                                        _typing.Tuple[float, float],
                                        _typing.Tuple[float, float, float]],
                 v_init: _typing.Optional[float] = None,
                 scale_factor: float = 1,
                 par_name: _typing.Optional[str] = None,
                 gui_type: _typing.Optional[str] = 'slider'):
        if type(v_range) is int:
            v_range = float(v_range)
        if type(v_range) is float:
            v_range = (0, v_range)
        if len(v_range) == 2:
            v_range = (v_range[0], v_range[1], (v_range[1] - v_range[0]) / 10)
        super().__init__([var],
                         v_range,
                         v_init=v_init,
                         scale_factor=scale_factor,
                         par_name=par_name,
                         gui_type=gui_type)


class AdjParRange(AdjustableParameter):
    """
    Defines a range slider version of `AdjustableParameter`

    Attributes
    ----------
    var_list : Tuple[str, str]
        Defines attributed affected by the `AdjustableParameter`.
        First has value of the start of the interval and second of the
        end of the interval.
    v_range or Tuple[float, float] or Tuple[float, float, float]
        Defines range end or [start, end] or [start, end, step].
        If step is not defined, the `step = (start - end) / 10`.
        If start is not defined the `start = 0`.
    v_init : Tuple[float, float]
        Initial value [interval start, interval end]. If None (default),
        then the initial values are assigned from the initial conditions
        at the start of the simulation.

    """

    def __init__(self,
                 var_list: _typing.Tuple[str, str],
                 v_range: _typing.Union[float,
                                        _typing.Tuple[float, float],
                                        _typing.Tuple[float, float, float]],
                 v_init: _typing.Optional[_typing.Tuple[float, float]] = None,
                 scale_factor: float = 1,
                 par_name: _typing.Optional[str] = None,
                 gui_type: _typing.Optional[str] = 'range'):
        if type(v_range) is int:
            v_range = float(v_range)
        if type(v_range) is float:
            v_range = (0, v_range)
        if len(v_range) == 2:
            v_range = (v_range[0], v_range[1], (v_range[1] - v_range[0]) / 10)
        super().__init__(list(var_list),
                         v_range,
                         v_init=v_init,
                         scale_factor=scale_factor,
                         par_name=par_name,
                         gui_type=gui_type)
