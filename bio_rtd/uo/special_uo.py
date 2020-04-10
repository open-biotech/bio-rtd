"""Special unit operations.

Special unit operations are the ones that cannot be assigned to one of
the following groups:

- Fully continuous (accept constant and provide constant flow rate)
- Semi continuous (accept constant and provide periodic flow rate)
- Surge tanks (accept any and provide constant flow rate)

"""

__all__ = ['ComboUO']
__version__ = '0.7.1'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.core as _core
import bio_rtd.logger as _logger


class ComboUO(_core.UnitOperation):
    """Combines multiple unit operations into one.

    Parameters
    ----------
    t
        Simulation time vector.

        All unit operations need to have the same `t`.
    sub_uo_list
        List of unit operations to be merged into `ComboUO`.
    uo_id
        Unique identified of combined unit operation.
    gui_title
        Human readable title for GUI or plots. Default = "ComboUO".

    """

    def __init__(self, t: _np.ndarray,
                 sub_uo_list: _typing.Sequence[_core.UnitOperation],
                 uo_id: str, gui_title: str = "ComboUO"):
        super().__init__(t, uo_id, gui_title)
        # Do not accept empty list.
        assert len(sub_uo_list) > 0
        # Make sure all uo-s have the same time vector.
        for uo in sub_uo_list[1:]:
            _np.testing.assert_array_almost_equal(t, uo._t)
        # Ensure unique `uo_id`s.
        ids = [uo.uo_id for uo in sub_uo_list]
        ids.append(uo_id)
        assert len(ids) == len(set(ids)), \
            "Each unit operation must have a unique id (`uo_id`)"

        self.sub_uo_list = sub_uo_list
        """List of unit operations that are merged into `ComboUO`."""

    def _calculate(self):  # pragma: no cover
        """This method has no 'flow-processing' logic."""
        pass

    def evaluate(self,
                 f_in: _np.array,
                 c_in: _np.ndarray
                 ) -> _typing.Tuple[_np.ndarray, _np.ndarray]:
        """Evaluates all child unit operations.

        Parameters
        ----------
        f_in
            Inlet flow rate profile.
        c_in
            Inlet concentration profile.

        Returns
        -------
        f_out
            Outlet flow rate profile.
        c_out
            Outlet concentration profile.

        """
        self._f = f_in.copy()
        self._c = c_in.copy()

        for uo in self.sub_uo_list:
            self._f, self._c = uo.evaluate(self._f, self._c)

        return self._f, self._c

    @_core.UnitOperation.log.setter
    def log(self, logger: _logger.RtdLogger):
        """Logger is passed to all child unit operations."""
        self._logger = logger
        self._logger.set_data_tree(self.uo_id, self._log_tree)
        for uo in self.sub_uo_list:
            uo.log = self._logger
