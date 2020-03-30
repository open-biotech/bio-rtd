import numpy as np
from bokeh.plotting import figure, show
from bio_rtd import pdf, uo
from bio_rtd.core import RtdModel
from bio_rtd.inlet import ConstantInlet

t = np.linspace(0, 10, 201)  # time

# Define inlet
inlet = ConstantInlet(t, inlet_id="sample_inlet",
                      f=3.5, c=np.array([1.0]),
                      species_list=['protein [mg/mL]'])

# Define unit operation.
ft_uo_1 = uo.fc_uo.FlowThrough(
    t=t, uo_id="ft_example",
    pdf=pdf.ExpModGaussianFixedDispersion(t, 0.3 ** 2 / 2, 1.0))
ft_uo_1.rt_target = 2.0

# Define another unit operation.
ft_uo_2 = uo.fc_uo.FlowThrough(t=t, uo_id="ft_example_2",
                               pdf=pdf.TanksInSeries(t, 3))
ft_uo_2.rt_target = 1.2

# Create an RtdModel.
rtd_model = RtdModel(inlet, dsp_uo_chain=[ft_uo_1, ft_uo_2])

# Run simulation.
rtd_model.recalculate()

# Plot.
p = figure(plot_width=690, plot_height=350,
           title="Model with 2 unit operations - Breakthrough",
           x_axis_label="t [min]", y_axis_label="c [mg/mL]")
p.line(t, inlet.get_result()[1][0], line_width=2, color='black',
       legend_label='inlet')
p.line(t, ft_uo_1.get_result()[1][0], line_width=2, color='green',
       legend_label='outlet of uo_1')
p.line(t, ft_uo_2.get_result()[1][0], line_width=2, color='blue',
       legend_label='outlet of uo_2')
p.legend.location = "bottom_right"
show(p)
