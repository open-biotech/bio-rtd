"""Modeling UFDF

We will describe uf-df with a combination of three unit operations:
 1. `Concentration`
 2. `BufferExchange`
 3. `FlowThrough`
    (for describing residence time distribution of the product)

All three unit operations will be joined into single `ComboUO`.

"""

import numpy as np
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Range1d, LinearAxis
from bokeh.plotting import figure

from bio_rtd import pdf
from bio_rtd.uo import fc_uo, special_uo

# Time step (`dt`) simulation time (`t`).
t = np.linspace(0, 100, 1000)
dt = t[1]

# ## CREATING INSTANCE ##

# Concentration.
conc = fc_uo.Concentration(
    t,
    flow_reduction=10,
    uo_id="concentration_sub_step"
)
conc.relative_losses = 0.10  # 10 % losses
# BufferExchange
buffer_exchange = fc_uo.BufferExchange(
    t,
    exchange_ratio=0.95,
    uo_id="buffer_exchange_sub_step"
)
# FlowThrough
flow_through = fc_uo.FlowThrough(
    t,
    # Peak shape description (result of a pulse injection experiment)
    pdf=pdf.GaussianFixedRelativeWidth(t, relative_sigma=0.2),
    uo_id="rtd_sub_step"
)
# Peak position ( = first momentum of a peak at pulse injection experiment)
flow_through.rt_target = 5  # min

# UFDF
uf_df = special_uo.ComboUO(t,
                           sub_uo_list=[conc, buffer_exchange, flow_through],
                           uo_id="uf_df",
                           gui_title="UfDf step")

# ## SIMULATION ##

# Inlet flow rate and concentration profile with 3 species.
f = np.ones_like(t) * 200.0  # mL/min
c = np.zeros([3, t.size])
# We choose protein as a 1st specie and set inlet concentration to 14 mg/ml.
c[0] = 14  # mg/ml
# We 'label' a part of the product and treat is as separate specie.
c[1, 300:600] = 14
# Last component represent salt that we try to remove from the process fluid.
c[2] = 1000  # mM

# Update `ud_df` so it does not retain salt.
conc.non_retained_species = [2]
buffer_exchange.non_retained_species = [2]

# Evaluate ( = run simulation).
f_out, c_out = uf_df.evaluate(f, c)

# Plot results.
p1 = figure(plot_width=690, plot_height=350, title="UfDf",
            x_axis_label="t [min]", y_axis_label="c [mg/mL]")
# Add new axis for flow rate to the right.
p1.extra_y_ranges = {'f': Range1d(0, max(f.max(), f_out.max()) * 1.1)}
p1.add_layout(LinearAxis(y_range_name='f'), 'right')
p1.yaxis[1].axis_label = "f [mL/min]"
# Protein conc and flow rate.
p1.line(t, c[0],
        line_width=2, color='green',
        legend_label='c_in, protein [mg/mL]')
p1.line(t, c[1],
        line_width=2, color='red',
        legend_label='c_in, protein, labeled [mg/mL]')
p1.line(t, f,
        y_range_name='f', line_width=1, color='black',
        legend_label='f_in')
# Flow.
p1.line(t, f_out,
        y_range_name='f', line_width=1, color='black',
        line_dash='dashed', legend_label='f_out')
p1.line(t, c_out[0],
        line_width=2, color='green', line_dash='dashed',
        legend_label='c_out, protein [mg/mL]')
p1.line(t, c_out[1],
        line_width=2, color='red', line_dash='dashed',
        legend_label='c_out, protein, labeled [mg/mL]')
p1.y_range.start = 0
p1.y_range.end = max(c[0:1].max(), c_out[0:1].max()) * 1.25
p1.legend.location = "center_right"

# Plot results.
p2 = figure(plot_width=690, plot_height=350, title="UfDf",
            x_axis_label="t [min]", y_axis_label="c [mM]")
# Add new axis for flow rate to the right.
p2.extra_y_ranges = {'f': Range1d(0, max(f.max(), f_out.max()) * 1.1)}
p2.add_layout(LinearAxis(y_range_name='f'), 'right')
p2.yaxis[1].axis_label = "f [mL/min]"
# Salt conc and flow rate.
p2.line(t, c[2], line_width=2, color='navy',
        legend_label='c_in, salt [mM]')
p2.line(t, f,
        y_range_name='f', line_width=1, color='black',
        legend_label='f_in')
p2.line(t, c_out[2], line_width=2, color='navy', line_dash='dashed',
        legend_label='c_out, salt [mM]')
p2.line(t, f_out,
        y_range_name='f', line_width=1, color='black',
        line_dash='dashed', legend_label='f_out')
p2.legend.location = "center_right"

# Plot both plots.
show(column(p1, p2))
