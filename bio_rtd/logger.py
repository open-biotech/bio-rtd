"""Module with different loggers for rtd models and its components.

`RtdLogger` is main abstract class that all other loggers implement.
Loggers handle non-fatal events (info, debug, warning and non-fatal
error messages) as well as internally stored intermediate process data.

General guidelines:

- **Skip logging and raise Exception directly** in case of a serious
  error.
- ERROR messages for highly probable sources of inaccuracies within the
  model (e.g. in case a probability distributions contains only a few
  data points), but we don't necessarily want to raise an exception.
- WARNING messages for warnings about potential sources of inaccuracy
  in the model.
- INFO for storing short intermediate data (everything apart from time
  series).
- DEBUG for storing long intermediate data (time series).

See Also
--------
:class:`bio_rtd.logger.RtdLogger`

Examples
--------
>>> log = DefaultLogger()
>>> log.e("Error Description")  # log error
Traceback (most recent call last):
RuntimeError: Error Description
>>> log.w("Warning Description")  # log waring
Warning Description
>>> log.i("Info")  # log info
>>> log.log_data = True
>>> log.log_level_data = RtdLogger.DEBUG
>>> data_tree = dict()
>>> log.set_data_tree("test_data_tree", data_tree)
>>> log.i_data(data_tree, "a", 3)  # store value in logger
>>> log.d_data(data_tree, "b", "lots_of_data")  # store at DEBUG level
>>> log.get_data_tree("test_data_tree",)["b"]
'lots_of_data'
>>> log = StrictLogger()  # raises RuntimeError already on WARNING level
>>> log.w("Warning Info")
Traceback (most recent call last):
RuntimeError: Warning Info

"""

__all__ = ['RtdLogger', 'DefaultLogger', 'DataStoringLogger', 'StrictLogger']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'


import inspect as _inspect
import typing as _typing

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from collections import OrderedDict as _OrderedDict


class RtdLogger(_ABC):
    """Abstract class for log and log data in rtd models.

    Logger has different levels: DEBUG, INFO, WARNING, ERROR.
    Logger has option to hold copies of intermediate data.

    Attributes
    ----------
    log_level
        Verbosity level for messages. Default = `WARNING`.
    log_data
        If True, logger collects copies of intermediate data.
        Default = False.
    log_level_data
        'Verbosity level' for data.
        Default = `DEBUG`. Ignored if `log_data` is False.


    """

    ERROR = 40
    """Log level ERROR.
    
    ERROR level is meant for signaling events (with messages) that very
    likely impact the accuracy of the model.
    
    """
    WARNING = 30
    """Log level WARNING.
    
    WARNING level is meant for signaling events (with messages) that
    might be the source of inaccuracies in the model.
    
    """
    INFO = 20
    """Log level INFO.
    
    INFO level is meant for events that might indicate potential mishaps
    in the model, but can also occur normally (e.g. if surge tank runs 
    dry). This info might help explain the spectra, but it might also
    be a normal occurrence during the shut-down phase.
    
    INFO level is also meant for keeping small intermediate results
    (no time series) in logger.
    
    """
    DEBUG = 10
    """Log level DEBUG.
    
    DEBUG level is meant for keeping also large intermediate data
    (time series) in logger.
    
    """

    def __init__(self,
                 log_level: int = WARNING,
                 log_data: bool = False,
                 log_level_data: int = DEBUG):
        self.log_level = log_level
        self.log_data = log_data
        self.log_level_data = log_level_data
        self._data_tree_root = _OrderedDict()

    def i(self, msg: str):
        """Log message at INFO level

        INFO level is meant for events that might indicate potential
        mishaps the model, but can also occur normally.

        Examples
        --------
        Reporting that surge tank ran dry.
            This info might help explain the spectra, but it might just
            as well be a normal occurrence during the shut-down phase.

        """
        self.log(level=self.INFO, msg=msg)

    def w(self, msg: str):
        """Log message at WARNING level

        WARNING level is meant for signaling events that might be the
        source of inaccuracies in the model.

        Examples
        --------
        Suspicious probability distributions.
        Empty concentration of flow rate profiles.
        Assumptions that might impact the results, such as assumptions
        during steady-state estimation.

        """
        self.log(level=self.WARNING, msg=msg)

    def e(self, msg: str):
        """Log message at ERROR level

        ERROR level is meant for signaling events that are very likely
        the source of or the sign of inaccuracies in the model, but not
        severe enough to raise an exception in all situations (e.g. in
        an interactive session via GUI).

        If the issue is big enough to warrant an exception, then raise
        the exception instead of using this log.

        Examples
        --------
        Probability distribution with only few data points.
        Probability distribution with a high cut-off at time 0.
        Load phase of PCC being shorter than the rest of the process.

        """
        self.log(level=self.ERROR, msg=msg)

    def set_data_tree(self, data_tree_id: str, data_tree: dict):
        """Seeds data tree in root dictionary.

        Typically `tree_id = uo.id` and `tree_root = dict()`.

        Unit operation needs to save a reference to the `tree_root`
        in order to store data in this logger.

        Parameters
        ----------
        data_tree_id: str
            Id of root tree (typically id of unit operation).
        data_tree: dict
            Root for data logging (for that unit operation).

        Examples
        --------
        >>> log = DataStoringLogger()
        >>> data_tree = dict()
        >>> log.set_data_tree("my_uo", data_tree)
        >>> log.i_data(data_tree, "par_a", 10.4)
        >>> data_tree["par_a"]
        10.4

        """
        self._data_tree_root[data_tree_id] = data_tree

    # noinspection PyMethodMayBeStatic
    def set_branch(self,
                   data_tree: dict,
                   branch_name: str,
                   branch: _typing.Union[dict, list]):
        """Seeds branch into data tree.

        Typically used to store data from nested unit operations or
        parts of unit operations (evaluation data of probability
        distributions, breakthrough profiles, etc.).

        Parameters
        ----------
        data_tree: dict
            Parent data tree.
        branch_name: dict
            Branch (child tree) name.
        branch: dict
            Branch (child tree).

        Examples
        --------
        >>> log = DataStoringLogger()
        >>> data_tree = dict()
        >>> branch_tree = dict()
        >>> log.set_data_tree("my_uo", data_tree)
        >>> log.set_branch(data_tree, "pdf", branch_tree)
        >>> log.i_data(branch_tree, "par_b", 8.4)
        >>> data_tree["pdf"]["par_b"]
        8.4
        >>> branch_tree["par_b"]
        8.4

        """
        data_tree[branch_name] = branch

    def i_data(self, tree: dict, key: str, value: any):
        """Log smaller data (no time series).

        Populates (key, value.copy()) pair into `tree`
        if `log_level_data` >= `INFO` and `log_data` is True.

        If the copy of the value is stored, then the
        `self._on_data_stored` function is called.

        """
        self._store_data(level=self.INFO, tree=tree, key=key, value=value)

    def d_data(self, tree: dict, key: str, value: any):
        """Log larger intermediate data (time series).

        Populates (key, value.copy()) pair into `tree`
        if `log_level_data` >= `DEBUG` and `log_data` is True.

        If the copy of the value is stored, then the
        `self._on_data_stored` function is called.

        """
        self._store_data(level=self.DEBUG, tree=tree, key=key, value=value)

    def _store_data(self, level: int, tree: dict, key: str, value: any):
        if self.log_data and level >= self.log_level_data:
            tree[key] = value.copy() if hasattr(value, 'copy') else value
            self._on_data_stored(level, tree, key, value)

    @_abstractmethod
    def _on_data_stored(self,
                        level: int, tree: dict,
                        key: str, value: any):  # pragma: no cover
        """Functions that is called after the data has been logged."""
        raise NotImplementedError

    @staticmethod
    def _get_parent_log_id() -> str:
        """Find closest 'log_entity_id' of the notifying instance.

        It is convenience method to provide unit operation to be used
        e.g. for adding information to the printed log messages.

        """
        for frame in _inspect.stack():
            # noinspection PyProtectedMember
            if "self" in frame[0].f_locals.keys() \
                    and hasattr(frame[0].f_locals["self"], '_log_entity_id')\
                    and len(frame[0].f_locals['self']._log_entity_id) > 0:
                # noinspection PyProtectedMember
                return f"{frame[0].f_locals['self']._log_entity_id}: "
        return ""

    def get_data_tree(self, data_tree_id: str) -> dict:
        """Returns reference to the registered data tree."""
        return self._data_tree_root[data_tree_id]

    def get_entire_data_tree(self):
        """Returns reference to the dict with all logged data."""
        return self._data_tree_root

    @_abstractmethod
    def log(self, level: int, msg: str):  # pragma: no cover
        """Log messages at specific log level.

        See documentation of `self.e()`, `self.w()` and `self.i()` on
        info about what belongs under which log level.

        """
        raise NotImplementedError


class DefaultLogger(RtdLogger):
    """Prints warnings to terminal and raises errors.

    Does not store data.

    Parameters
    ----------
    log_level
        Default: WARNING.
    log_data
        Default: False.
    log_level_data
        Default: DEBUG.

    Examples
    --------
    >>> log = DefaultLogger()
    >>> data_tree = dict()
    >>> log.set_data_tree("my_uo", data_tree)
    >>> log.i_data(data_tree, "par_i", 8.4)  # ignored
    >>> log.d_data(data_tree, "par_d", 5.4)  # ignored
    >>> len(data_tree.keys())  # no data
    0
    >>> log.i(msg="Info msg")  # ignored
    >>> log.w(msg="Warning msg")
    Warning msg
    >>> log.e(msg="Error msg")
    Traceback (most recent call last):
    RuntimeError: Error msg

    See Also
    --------
    :class:`bio_rtd.logger.RtdLogger`

    """

    def __init__(self,
                 log_level: int = RtdLogger.WARNING,
                 log_data: bool = False,
                 log_level_data: int = RtdLogger.DEBUG):
        super().__init__(log_level, log_data, log_level_data)

    def log(self, level: int, msg: str):
        """Log messages.

        Prints warnings (and infos)
        to terminal and raises errors as `RuntimeError`.

        """
        if level >= self.log_level:
            if level >= self.ERROR:
                raise RuntimeError(msg)
            elif level == self.WARNING:
                print(f"WARNING: {self._get_parent_log_id()}{msg}")
            else:
                print(f"{self._get_parent_log_id()}{msg}")

    def _on_data_stored(self, level: int, tree: dict, key: str, value: any):
        pass


class DataStoringLogger(RtdLogger):
    """Prints messages to terminal. Stores all data.

    Parameters
    ----------
    log_level
        Default: WARNING.
    log_data
        Default: True.
    log_level_data
        Default: DEBUG.

    Examples
    --------
    >>> log = DataStoringLogger()
    >>> data_tree = dict()
    >>> log.set_data_tree("my_uo", data_tree)
    >>> log.i_data(data_tree, "par_i", 8.4)  # stored
    >>> log.d_data(data_tree, "par_d", 5.4)  # stored
    >>> len(data_tree.keys())  # no data
    2
    >>> data_tree["par_i"]
    8.4
    >>> data_tree["par_d"]
    5.4
    >>> log.i(msg="Info msg")  # ignored
    >>> log.w(msg="Warning msg")
    Warning msg
    >>> log.e(msg="Error msg")
    Error msg
    >>> log.log_level = log.INFO
    >>> log.i_data(data_tree, "par_i_2", 18.4)  # stored and printed
    Value set: par_i_2: 18.4
    >>> data_tree["par_i_2"]
    18.4

    See Also
    --------
    :class:`bio_rtd.logger.RtdLogger`

    """

    def __init__(self,
                 log_level: int = RtdLogger.WARNING,
                 log_data: bool = True,
                 log_level_data: int = RtdLogger.DEBUG):
        super().__init__(log_level, log_data, log_level_data)

    def log(self, level: int, msg: str):
        """Prints to terminal everything above `log_level`."""
        if level >= self.log_level:
            if level == self.ERROR:
                print(f"ERROR: {self._get_parent_log_id()}{msg}")
            elif level == self.WARNING:
                print(f"WARNING: {self._get_parent_log_id()}{msg}")
            elif level >= self.INFO:
                print(f"INFO: {self._get_parent_log_id()}{msg}")
            else:
                print(f"DEBUG: {self._get_parent_log_id()}{msg}")

    def _on_data_stored(self, level: int, tree: dict, key: str, value: any):
        # send to log function if INFO or higher
        if level >= self.INFO:
            self.log(level, "Value set: " + key + ": " + str(value))


class StrictLogger(RtdLogger):
    """Raises RuntimeError on warning and error messages.

    log_level = WARNING.

    log_data = False.

    Examples
    --------
    >>> log = StrictLogger()
    >>> log.i(msg="Info msg")  # ignored
    >>> log.w(msg="Warning msg")
    Traceback (most recent call last):
    RuntimeError: Warning msg
    >>> log.e(msg="Error msg")
    Traceback (most recent call last):
    RuntimeError: Error msg

    See Also
    --------
    :class:`bio_rtd.logger.RtdLogger`

    """

    def __init__(self):
        super().__init__(log_level=RtdLogger.WARNING, log_data=False)

    def log(self, level: int, msg: str):
        """Raises RuntimeError if `level` >= WARNING."""
        if level >= RtdLogger.WARNING:
            raise RuntimeError(self._get_parent_log_id() + msg)

    def _on_data_stored(self, level: int, tree: dict, key: str, value: any):
        pass
