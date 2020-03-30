"""Helper file for reading process data from Excel spreadsheet."""

__version__ = '0.2'
__author__ = 'Jure Sencar'

import numpy as _np
import pandas as _pd


def read_bio_process_xlsx_data(file_path) -> (dict, dict, dict):
    """Reads process data from Excel spreadsheet.

    Parameters
    ----------
    file_path
        Path to Excel spreadsheet file.

    Returns
    -------
    (dict, dict, dict)
        uo_lib
            Dict of all unit operations.
            uo[uo_id]['data'] : dict
            uo[uo_id]['title']
            uo[uo_id]['data id']
        dsp
            Dict of all DSP scenarios.
            dsp[dsp_scenario] : list of uo_id
        usp
            Dict of all USP scenarios.
            Entry contains key-value pairs as specified in spreadsheets.
            usp[usp_scenario] : dict

    """

    def _is_blank(d) -> bool:
        """Checks for empty data field.

        Parameters
        ----------
        d
            Value of a field of a pandas data-frame.

        Returns
        -------
        bool
            True if `d` represents empty field in Excel spreadsheet.

        """
        return d.__class__ == float and _np.isnan(d)

    def _next_item_location(df: _pd.DataFrame, target,
                            start_row=0, start_column=0) -> [int, int]:
        """Returns position of first match.

        Function scans row-by-row.

        Parameters
        ----------
        df : _pd.DataFrame
            DataFrame which is supposed to contain the `target` value
        target : any
            Value of a filed that the function is locating.
        start_row : int
            Start searching at `start_row`.
        start_column : int
            Start searching at `start_column` of `start_row`.
            All columns are scanned for following rows.

        Returns
        -------
        (int, int)
            Item position (row, column).
            If the item is not found (-1, -1) is returned.

        """
        for i in range(start_row, df.shape[0]):
            init_column_position = start_column if i == start_row else 0
            for j in range(init_column_position, df.shape[1]):
                if df.iat[i, j] == target:
                    # print([i, j, start_row, start_column])
                    return i, j
        return -1, -1

    def parse_process_parameters(pp_sheet) -> dict:
        process_parameters = dict()
        r = 0
        c = -1
        while r >= 0:
            r, c = _next_item_location(pp_sheet, 'data id', r, c + 1)
            if c == -1:
                return process_parameters

            pp_group_id = pp_sheet.iat[r, c + 1]
            pp_group_data = dict()

            for pp_row in range(r + 1, pp_sheet.shape[0]):
                key = pp_sheet.iat[pp_row, c]
                if _is_blank(key):
                    break
                pp_group_data[key] = pp_sheet.iat[pp_row, c + 1]

            process_parameters[pp_group_id] = pp_group_data
        return process_parameters

    def parse_uo(uo_sheet, pp_sheet) -> dict:
        # Get process parameters.
        proc_par = parse_process_parameters(pp_sheet)
        # Combine with unit operations.
        uo_lib = dict()
        r, c = _next_item_location(uo_sheet, 'id')
        for i in range(r + 1, uo_sheet.shape[0]):
            uo = dict()
            uo_id = uo_sheet.iat[i, c]
            if _is_blank(uo_id):
                return uo_lib
            for j in range(c, uo_sheet.shape[1]):
                key = uo_sheet.iat[r, j]
                if _is_blank(key):
                    continue
                value = uo_sheet.iat[i, j]
                if _is_blank(value):
                    raise ValueError(f"Value at {[i, j]}"
                                     f" ({uo_id}, {key})"
                                     f" is missing.")
                uo[key] = value
                if key == 'data id':
                    assert value in proc_par.keys(), \
                        f"Process parameters `{value}` missing for `{uo_id}`."
                    uo['data id'] = value
                    uo['data'] = proc_par[value]
            uo_lib[uo_id] = uo
        return uo_lib

    def parse_dsp(dsp_sheet) -> dict:
        dsp = dict()
        r = 0
        c = -1
        while r >= 0:
            r, c = _next_item_location(dsp_sheet, 'Scenario', r, c + 1)
            if c == -1:
                return dsp

            dsp_scenario = list()
            scenario_name = dsp_sheet.iat[r, c + 1]
            uo_add_data_row = r + 1

            while uo_add_data_row < dsp_sheet.shape[0]:
                value = dsp_sheet.iat[uo_add_data_row, c + 1]
                if type(value) is not str:
                    break
                dsp_scenario.append(value)
                uo_add_data_row += 1

            dsp[scenario_name] = dsp_scenario
        return dsp

    def parse_usp(usp_sheet) -> dict:
        usp = dict()
        # find scenario index column
        r, c = _next_item_location(usp_sheet, 'Scenario')
        # parse fields
        for i in range(r + 1, usp_sheet.shape[0]):
            usp_scenario = dict()
            for j in range(c + 1, usp_sheet.shape[1]):
                usp_scenario[usp_sheet.iat[r, j]] = usp_sheet.iat[i, j]
            # save scenario
            usp[usp_sheet.iat[i, c]] = usp_scenario
        return usp

    return (
        parse_uo(_pd.read_excel(file_path, 'Unit Operations'),
                 _pd.read_excel(file_path, 'Process Parameters')),
        parse_dsp(_pd.read_excel(file_path, 'Scenario by DSP')),
        parse_usp(_pd.read_excel(file_path, 'Scenario by USP'))
    )
