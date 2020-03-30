import numpy as np
from bokeh.plotting import figure, show
from bio_rtd import peak_shapes, utils

# Time vector.
t = np.linspace(0, 10, 201)

# Generate noisy data.
y = peak_shapes.emg(t, 2, 0.3, 1.0)  # clean signal
y_noisy = y + (np.random.random(y.shape) - 0.5) * y.max() / 10

# Determine peak start, end and max position.
i_start, i_end = utils.vectors.true_start_and_end(y > 0.1 * y.max())
i_max = y.argmax()

# Plot.
p = figure(plot_width=690, plot_height=350, title="Measurements",
           x_axis_label="t [min]", y_axis_label="c [mg/mL]")
p.line(t, y_noisy, line_width=2, color='green', alpha=0.6,
       legend_label='c [mg/mL]')
for i in [i_max, i_start, i_end]:  # plot circles
    p.circle(t[i], y[i], size=15, fill_alpha=0,
             line_color="blue", line_width=2)
show(p)
