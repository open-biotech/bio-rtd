import numpy as np
from bokeh.plotting import figure, show
from bio_rtd import pdf, uo, peak_shapes, peak_fitting

t = np.linspace(0, 10, 201)

# Concentration with spike at t = 0
c_in = np.zeros([1, t.size])
c_in[0][0] = 1 / t[1]

# Flow rate
f = np.ones_like(t) * 3.5

# Calc rt_mean, sigma and skew from peak points
rt, sigma, skew = peak_fitting.calc_emg_parameters_from_peak_shape(
    t_peak_max=1.4, t_peak_start=0.6, t_peak_end=3.9,
    relative_threshold=0.1
)

# Define unit operation
ft_uo = uo.fc_uo.FlowThrough(
    t=t, uo_id="ft_example",
    pdf=pdf.ExpModGaussianFixedDispersion(t, sigma ** 2 / rt, skew),
)
ft_uo.v_void = rt * f[0]  # set void volume as rt * flow rate

# Simulate.
f_out, c_out = ft_uo.evaluate(f, c_in)

# Define noisy data.
y = peak_shapes.emg(t, 2, 0.3, 1.0)  # clean signal
y_noisy = y + (np.random.random(y.shape) - 0.5) * y.max() / 10

# Plot.
p = figure(plot_width=690, plot_height=350,
           title="Unit Operation - Pulse Response",
           x_axis_label="t [min]", y_axis_label="c [mg/mL]")
p.line(t, y_noisy, line_width=2, color='green', alpha=0.6,
       legend_label='c [mg/mL] (data)')
p.line(t, c_out[0], line_width=2, color='black', alpha=1,
       legend_label='c [mg/mL] (model)')
show(p)
