"""Example use of TwoAlternatingCSTRs unit operation.

In this example we simulate propagation of periodic inlet flow rate
and concentration profiles throughout the unit operation with two
alternating CSTRs. Afterwards we create a bunch of plots.

"""

import numpy as np
from bokeh.layouts import column

from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure, show

from bio_rtd.uo.surge_tank import TwoAlternatingCSTRs


"""Simulation time vector."""
t = np.linspace(0, 100, 1001)
dt = t[1]

"""Define unit operation."""
a2cstr = TwoAlternatingCSTRs(t, "a2cstr")
a2cstr.v_leftover_rel = 0.05  # 95 % discharge

"""Inlet profile."""
# Periodic inlet profile.
f_in = np.zeros_like(t)
c_in = np.zeros([5, t.size])
i_f_on_duration = int(round(0.025 * t.size))
for i, period_start in enumerate(np.arange(0.2, 0.9, 0.1) * t.size):
    i_f_start = int(round(period_start))
    f_in[i_f_start:i_f_start + i_f_on_duration] = 3.5
    i_mod = i % c_in.shape[0]
    c_in[i_mod, i_f_start:i_f_start + i_f_on_duration] = 1 + 0.1 * i_mod

"""Simulation."""
f_out, c_out = a2cstr.evaluate(f_in, c_in)

"""Plot inlet and outlet profiles."""


def make_plot(f, c, title_extra):
    p1 = figure(plot_width=695, plot_height=350,
                title=f"TwoAlternatingCSTRs - {title_extra}",
                x_axis_label="t [min]", y_axis_label="c [mg/mL]")
    # Add flow rate axis.
    p1.extra_y_ranges = {'f': Range1d(0, f.max() * 1.1)}
    p1.y_range = Range1d(0, c.max() * 1.5)
    p1.add_layout(LinearAxis(y_range_name='f'), 'right')
    p1.yaxis[1].axis_label = "f [mL/min]"
    # Flow.
    p1.line(t, f, y_range_name='f',
            line_width=1.5, color='black', legend_label='flow rate')
    # Conc.
    line_colors = ['#3288bd', '#11d594', 'red', '#fc8d59', 'navy']
    for i_specie in range(c.shape[0]):
        p1.line(t, c[i_specie],
                line_width=2,
                color=line_colors[i_specie],
                legend_label=f"conc {i_specie}")
    p1.legend.location = "top_left"
    return p1


p_in = make_plot(f_in, c_in, "inlet")
p_out = make_plot(f_out, c_out, "outlet")
show(column(p_in, p_out))
