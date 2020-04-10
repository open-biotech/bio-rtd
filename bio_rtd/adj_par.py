"""Module for defining adjustable attributes of the RtdModel.

Adjustable attributes are mainly used by GUI in order to offer
responsive parameter setting.

Notes
-----
Technically this module relates to instance attributes
rather than parameters. However, the term 'process parameters'
is well established in biotech, hence the word 'parameter' appears
in class names and docstring instead of 'instance attribute`.

"""

__all__ = ['AdjParSlider', 'AdjParRange', 'AdjParBoolean',
           'AdjParBooleanMultiple', 'AdjustableParameter']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
from abc import ABC as _ABC


class AdjustableParameter(_ABC):
    """Adjustable parameter for `UnitOperation` (or `Inlet`).

    The class does not provide any logic. It just provides a
    unified way to specify an adjustable attribute.

    This class primarily serves as an abstract class for specific
    extensions.

    Parameters
    ----------
    var_list
        A list of attributes affected by the `AdjustableParameter`.
    v_range
        Range of values.
    v_init
        Initial value. If None, the value is inherited from attribute.
    scale_factor
        Scale factor.
    par_name
        Attribute title.
    gui_type
        Method for setting parameter in GUI.

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
        self.var_list: _typing.List[str] = var_list
        """A list of attributes affected by the `AdjustableParameter`.
        
        Attributes typically belong to 
        :class:`bio_rtd.core.UnitOperation` or
        :class:`bio_rtd.core.Inlet` instance.
        
        Examples
        --------
        var_list = ['c_start[1]', 'c_end[1]']
        
        var_list = ['starts_empty']
        
        """
        self.v_range: _typing.Optional[_typing.Union[
                     _typing.Tuple[float, float, float],
                     _typing.Sequence[str],
                     _typing.Sequence[float],
                     _typing.Sequence[_typing.Tuple[str, str]]]] = v_range
        """Range of values.
        
        For boolean set None.
        
        For range specify [start, end, step].
        
        For list set list. If the list has two columns,
        the first one is the title and the second is the value.
        
        """
        self.v_init: _typing.Optional[_typing.Union[
                     float,
                     bool,
                     _typing.Sequence[float]]] = v_init
        """Initial value.
        
        If None (default), the value is inherited from initial attribute
        value at the start of runtime.
        
        """
        self.scale_factor: float = scale_factor
        """Scale factor for compensation for differences in units.
        
        Examples
        --------
        Setting time (:attr:`v_init`) in [h] for attribute 
        (in :attr:`var_list`) in [min] requires
        a `scale_factor` of 60.
        
        """
        self.par_name: _typing.Optional[_typing.Union[
                     str,
                     _typing.Sequence[str]]] = par_name
        """Attribute title.
        
        Examples
        --------
        par_name = 'Initial delay [min]'  # slider, range, boolean
        
        par_name = ['Option 1', 'Option 2', 'Option 3']  # radio buttons
        
        """
        self.gui_type: _typing.Optional[str] = gui_type
        """Method for setting parameter in GUI.
        
        If None (default) or the gui does not support the type,
        then the gui should decide the method based on `v_init` and/or
        `var_list`.
        
        Example values:
        'checkbox', 'range', 'slider', 'select', 'multi-select'
        
        """


class AdjParBoolean(AdjustableParameter):
    """A boolean variation of :class:`AdjustableParameter`

    Parameters
    ----------
    var
        Attribute affected by the `AdjParBoolean`.
    v_init
        Initial value. If None, the value is inherited from attribute.
    par_name
        Attribute title.
    gui_type
        Method for setting parameter in GUI. Default = 'checkbox'.

    See Also
    --------
    :class:`bio_rtd.adj_par.AdjustableParameter`

    """

    def __init__(self,
                 var: str,
                 v_init: _typing.Optional[bool] = None,
                 par_name: _typing.Optional[str] = None,
                 gui_type: _typing.Optional[str] = 'checkbox'):
        super().__init__([var],
                         v_init=v_init,
                         par_name=par_name,
                         gui_type=gui_type)


class AdjParBooleanMultiple(AdjustableParameter):
    """A radio group variation of `AdjustableParameter`

    Parameters
    ----------
    var_list
        List of attributes affected by the `AdjParBooleanMultiple`.
    v_init
        Initial value. If None, the value is inherited from attribute.
    par_name
        Attribute title.
    gui_type
        Method for setting parameter in GUI. Default = 'radio_group'.

    See Also
    --------
    :class:`bio_rtd.adj_par.AdjustableParameter`

    """

    def __init__(self,
                 var_list: _typing.List[str],
                 v_init: _typing.Optional[_typing.Sequence[bool]] = None,
                 par_name: _typing.Optional[_typing.Sequence[str]] = None,
                 gui_type: _typing.Optional[str] = 'radio_group'):
        if v_init is not None:
            assert len(v_init) == len(var_list)
        if par_name is not None:
            assert len(par_name) == len(var_list)
        super().__init__(var_list=var_list,
                         v_init=v_init,
                         par_name=par_name,
                         gui_type=gui_type)


class AdjParSlider(AdjustableParameter):
    """A value slider version of `AdjustableParameter`.

    Parameters
    ----------
    var
        Attribute affected by the `AdjParSlider`.
    v_range
        Range of values.

        Defines range end or [start, end] or [start, end, step].

        If step is not defined, the `step = (start - end) / 10`.

        If start is not defined the `start = 0`.
    v_init
        Initial value. If None, the value is inherited from attribute.
    scale_factor
        Scale factor for compensation for differences in units.

        Examples
        --------
        Setting time (:attr:`v_init`) in [h] for attribute
        (in :attr:`var_list`) in [min] requires
        a `scale_factor` of 60.
    par_name
        Attribute title.
    gui_type
        Method for setting parameter in GUI. Default = 'slider'.

    See Also
    --------
    :class:`bio_rtd.adj_par.AdjustableParameter`

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

    Parameters
    ----------
    var_list
        Defines attributed affected by the `AdjParRange`.

        First has value of the start of the interval and second of the
        end of the interval.
    v_range
        Range of values.

        Defines range end or [start, end] or [start, end, step].

        If step is not defined, the `step = (start - end) / 10`.

        If start is not defined the `start = 0`.
    v_init
        Initial value [interval start, interval end]. If None (default),
        then the initial values are assigned from the initial conditions
        at the start of the simulation.
    scale_factor
        Scale factor for compensation for differences in units.

        Examples
        --------
        Setting time (:attr:`v_init`) in [h] for attribute
        (in :attr:`var_list`) in [min] requires
        a `scale_factor` of 60.
    par_name
        Attribute title.
    gui_type
        Method for setting parameter in GUI. Default = 'range'.

    See Also
    --------
    :class:`bio_rtd.adj_par.AdjustableParameter`

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
