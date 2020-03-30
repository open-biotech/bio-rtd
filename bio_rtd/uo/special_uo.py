__all__ = ['ComboUO']
__version__ = '0.2'
__author__ = 'Jure Sencar'

import typing as _typing
import numpy as _np

import bio_rtd.core as _core
import bio_rtd.logger as _logger


class ComboUO(_core.UnitOperation):
    """ A class that can hold a list of unit subsequent operation and present them as one """

    def __init__(self, t: _np.ndarray,
                 sub_uo_list: _typing.Sequence[_core.UnitOperation],
                 uo_id: str, gui_title: str = "ComboUO"):

        # do not accept empty list
        assert len(sub_uo_list) > 0

        # make sure all uo-s have the same time vector
        for uo in sub_uo_list[1:]:
            _np.testing.assert_array_almost_equal(t, uo._t)

        # ensure unique ids
        ids = [uo.uo_id for uo in sub_uo_list]
        ids.append(uo_id)
        assert len(ids) == len(set(ids)), "Each unit operation must have a unique id (`uo_id`)"

        # save reference to list of child unit operations
        self.sub_uo_list = sub_uo_list

        # call super method
        super().__init__(t, uo_id, gui_title)

    def _calculate(self):  # pragma: no cover
        """ This method has no 'flow-processing' logic. """
        pass

    def evaluate(self, f_in: _np.array, c_in: _np.ndarray):
        """ Evaluates all child unit operations. """
        self._f = f_in.copy()
        self._c = c_in.copy()

        for uo in self.sub_uo_list:
            self._f, self._c = uo.evaluate(self._f, self._c)

        return self._f, self._c

    @_core.UnitOperation.log.setter
    def log(self, logger: _logger.RtdLogger):
        """ Extension of default logger setter. Logger is passed to child unit operations. """
        self._logger = logger
        self._logger.set_data_tree(self.uo_id, self._log_tree)
        for uo in self.sub_uo_list:
            uo.log = self._logger
