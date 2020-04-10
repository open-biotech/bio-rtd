"""Example with graphical user interface build on top of RtdModel

`RtdModel` instance is taken from `integrated_mab.py`.

`BokehServerGui` is implementation of `UserInterface` abstract class.
For more information see docstrings of both classes.

"""

import numpy as np

from bio_rtd import adj_par
from examples.models.integrated_mab import rtd_model

from examples.models.util.gui_bokeh import BokehServerGui

"""Exposing process parameter for manipulation via GUI."""

# Inlet
rtd_model.inlet.adj_par_list = [
    adj_par.AdjParSlider(
        var="c[0]",
        v_range=(0.2, 10, 0.2),
        par_name='mAB concentration [mg/mL]'),
    adj_par.AdjParSlider(
        var="f",
        v_range=(0.1, 2, 0.1),
        scale_factor=1000 / 60,
        par_name='flow rate [L/h]'),
]
# PCC column size
rtd_model.dsp_uo_chain[1].adj_par_list = [
    adj_par.AdjParSlider(
        var="cv",
        v_range=(50, 500, 50),
        par_name='column size [mL]'),
]
# CSTR size
rtd_model.dsp_uo_chain[2].adj_par_list = [
    adj_par.AdjParBoolean(
        var="starts_empty",
        par_name='starts empty'),
    adj_par.AdjParSlider(
        var="v_void",
        v_range=(10, 200, 10),
        par_name='void volume [mL]'),
]
# Set CSTR value definition to absolute.
rtd_model.dsp_uo_chain[2].v_void = 50
rtd_model.dsp_uo_chain[2].v_min_ratio = -1
# AEX column size
rtd_model.dsp_uo_chain[4].adj_par_list = [
    adj_par.AdjParSlider(
        var="v_void",
        v_range=(1, 50, 1),
        par_name='void volume [mL]'),
]

"""Set up `UserInterface` instance."""
gui = BokehServerGui(rtd_model=rtd_model)

# Customize `BokehServerGui`.
gui.plot_height = 300
gui.x_scale_factor = 1 / 60  # from min -> hours
gui.x_label = 't [h]'
gui.y_label_f = 'f [mL/min]'
gui.y_label_c = 'c [mg/mL]'
gui.species_label = ['mAB concentration']
gui.custom_x_ticks = np.arange(25)
gui.legend_only_on_first = True
gui.dti_plot = 1  # do not reduce data points (increase for more responsive UI)
gui.line_colors = ['black', 'grey', 'black']
gui.line_dashes = ['dashed']
gui.font_size_pt = 14

# Build GUI.
gui.build_ui()

# Run simulation.
gui.recalculate(True)

"""Run GUI.

To enable interactive session, the script must be run through
`python bokeh serve` or `bokeh serve` command with bio_rtd folder
added to the PYTHON_PATH.
In terminal go to the repo (bio-unit_test) folder and run:
 export PYTHONPATH="$PWD"
 python `which bokeh` serve --show examples/models/integrated_mab_gui.py 

"""
