import numpy as np
from bokeh.plotting import figure, show
from bio_rtd import pdf, uo

# Define inlet profiles.
t = np.linspace(0, 10, 201)  # time
c_in = np.ones([1, t.size])  # concentration (constant)
f = np.ones_like(t) * 3.5  # flow rate

# Define unit operation.
ft_uo = uo.fc_uo.FlowThrough(
    t=t, uo_id="ft_example",
    pdf=pdf.ExpModGaussianFixedDispersion(t, 0.3 ** 2 / 2, 1.0))
ft_uo.v_void = 2 * f[0]  # set void volume (rt * flow rate)

# Simulation.
f_out, c_out = ft_uo.evaluate(f, c_in)

# Plot.
p = figure(plot_width=690, plot_height=350,
           title="Unit Operation - Breakthrough",
           x_axis_label="t [min]", y_axis_label="c [mg/mL]")
p.line(t, c_out[0], line_width=2, color='black',
       legend_label='c [mg/mL]')
show(p)
