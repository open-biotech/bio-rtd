"""Example `RtdModel` for mAB downstream process

In this example we define unit operations for a hypothetical mAB
process. We use only single specie in the model in order to narrow
the scope of the sample.

Examples
--------
Additional species can be added to the model by following the steps:
>>> # Add new species to the inlet concentration vector.
>>> rtd_model.inlet.species_list = ['mAB', 'new_sp_1', 'new_sp_2']
>>> rtd_model.inlet.c = np.array([2.4, 20, 30])
>>> # Update unit operations.
>>> # E.g. PCC should not bind new species.
>>> pcc_pro_a.non_binding_species = [1, 2]
>>> # E.g. UFDF should not retain not bind new species.
>>> conc.non_retained_species = [1, 2]
>>> buffer_exchange.non_retained_species = [1, 2]
>>> # Update parameters, like elution buffer composition, if needed.

Notes
-----
Process parameters
    Process parameters were chosen in a way that the workflow and the
    results can be easily interpreted by the user. This means that the
    process parameters might not always represent an actual mAB
    production process.

    See templates of individual unit operations for more information
    about which process parameters can be defined.

    See docstrings of individual unit operations for more details about
    individual process parameters.

Units
    RtdModel does not depend on any specific set of units.
    They just need to be consistent across the model. E.g.:
    Unit `base`:
      * time [min]
      * volume [mL]
      * weight [mg]
    Independent species can have different units, e.g.:
      * protein [IU]
      * acetate [mmol]
      * salt [mg]
    All derived units need to be a combination of 'basic' units.
      * flow rate [mL/min]
      * column height [cm]
      * column binding capacity (protein) [IU/mL]
      * etc.
    In presented model, [mL], [mg] and [min] were used as a 'base'.

"""

import numpy as np
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Range1d, LinearAxis
from bokeh.plotting import figure

from bio_rtd import pdf, peak_fitting, inlet
from bio_rtd.chromatography import bt_load
from bio_rtd.uo import sc_uo, fc_uo, surge_tank, special_uo
from bio_rtd.core import RtdModel
from bio_rtd.logger import DataStoringLogger


"""Simulation time."""

dt = 0.5  # min
t = np.arange(0, 24.1 * 60, dt)


"""DSP train."""

# Cell removal.
ft_cell_removal = fc_uo.FlowThroughWithSwitching(
    t, pdf=pdf.GaussianFixedRelativeWidth(t, relative_sigma=0.2),
    uo_id="cell_removal_ft",
    gui_title="Cell removal"
)
ft_cell_removal.v_void = 200  # mL
ft_cell_removal.v_cycle = 20000  # mL; switch filter unit after 20 L

# ProteinA PCC.
# Describe binding dynamics during load.
load_bt = bt_load.ConstantPatternSolution(dt, dbc_100=50, k=0.12)
# Describe elution peak with EMG pdf.
ep_rt_mean, sigma, skew = \
    peak_fitting.calc_emg_parameters_from_peak_shape(t_peak_start=9,
                                                     t_peak_max=16,
                                                     t_peak_end=27.5,
                                                     relative_threshold=0.05)
# Assuming the above experiments were done with 10 mL column.
ep_rt_mean_cv = ep_rt_mean / 10
# Elution peak shape pdf.
peak_shape_pdf = pdf.ExpModGaussianFixedRelativeWidth(
    t,
    sigma_relative=sigma / ep_rt_mean,
    tau_relative=1 / skew / ep_rt_mean,
    pdf_id="emg_rw_el_peak"
)
# Load recycle pdf. We try to describe the propagation of unbound and/or
# desorbed material throughout the column during load (and wash) step.
load_recycle_pdf = pdf.GaussianFixedDispersion(t,
                                               dispersion_index=2 * 2 / 30,
                                               pdf_id="g_fd_load_recycle_pdf")
pcc_pro_a = sc_uo.PCC(
    t,
    load_bt=load_bt,
    peak_shape_pdf=peak_shape_pdf,
    load_recycle_pdf=load_recycle_pdf,
    # Porosity of the column for protein.
    column_porosity_retentate=0.64,
    uo_id="pro_a_pcc",
    gui_title="ProteinA PCC",
)
pcc_pro_a.cv = 100  # mL
# Equilibration step.
pcc_pro_a.equilibration_cv = 1.5
# Equilibration flow rate is same as load flow rate.
pcc_pro_a.equilibration_f_rel = 1
# Load until 70 % breakthrough.
pcc_pro_a.load_c_end_relative_ss = 0.7
# Automatically prolong first cycle in order to achieve steady-state faster.
pcc_pro_a.load_extend_first_cycle = True
# Define wash step. There is no desorption during wash step in this example.
pcc_pro_a.wash_cv = 5
pcc_pro_a.wash_recycle = True
pcc_pro_a.wash_recycle_duration_cv = 2
# Elution step.
pcc_pro_a.elution_cv = 3
# 1st momentum of elution peak from data from above.
pcc_pro_a.elution_peak_position_cv = ep_rt_mean_cv
pcc_pro_a.elution_peak_cut_start_c_rel_to_peak_max = 0.05
pcc_pro_a.elution_peak_cut_end_c_rel_to_peak_max = 0.05
# Regeneration step.
pcc_pro_a.regeneration_cv = 1.5

# Surge tank - CSTR.
st_cstr = surge_tank.CSTR(t, uo_id="st_cstr", gui_title="Surge Tank - CSTR")
st_cstr.v_min_ratio = 0.1  # 10 % fill level remains after discharge
st_cstr.starts_empty = True

# Virus inactivation - FlowThrough, no switching
ft_vi = fc_uo.FlowThrough(
    t,
    pdf=pdf.GaussianFixedDispersion(t, dispersion_index=20 ** 2 / 100),
    uo_id="vi_ft", gui_title="Virus Inactivation - flow-through column")
ft_vi.rt_target = 68  # min

# AEX polishing step, FlowThroughWithSwitching.
ft_aex = fc_uo.FlowThroughWithSwitching(
    t,
    pdf=pdf.GaussianFixedDispersion(t, dispersion_index=2 ** 2 / 8),
    uo_id="aex_ft", gui_title="Polishing, AEX, flow-through")
ft_aex.v_void = 10  # mL
ft_aex.v_cycle = 50 * 10  # mL; switch column unit after 20 L

# UFDF
# Concentration.
conc = fc_uo.Concentration(t, flow_reduction=40, uo_id="uf_df_conc")
# BufferExchange
buffer_exchange = fc_uo.BufferExchange(t, exchange_ratio=0.995,
                                       uo_id="uf_df_buffer_exchange")
# FlowThrough
flow_through = fc_uo.FlowThrough(t, pdf=pdf.TanksInSeries(t, n_tanks=3),
                                 uo_id="uf_df_rtd")
flow_through.v_void = 0.5 * 3  # mL
uf_df = special_uo.ComboUO(
    t, sub_uo_list=[conc, buffer_exchange, flow_through],
    uo_id="uf_df", gui_title="UFDF")

# DSP train
dsp_train = [ft_cell_removal, pcc_pro_a, st_cstr, ft_vi, ft_aex, uf_df]


"""Inlet."""

const_inlet = inlet.ConstantInlet(t,
                                  f=1000 / 60,  # mL/min (= 1 L / h)
                                  c=np.array([2.4]),  # mg/mL
                                  species_list=["mAB [mg/mL]"],
                                  inlet_id="constant_inlet",
                                  gui_title="Constant inlet")


"""`RtdModel`."""

rtd_model = RtdModel(inlet=const_inlet, dsp_uo_chain=dsp_train,
                     logger=DataStoringLogger(),
                     title="Sample model for mAB production process")

if __name__ != "examples.models.integrated_mab":
    # Ignore warnings when not running script directly.
    rtd_model.log.log_level = rtd_model.log.ERROR

"""Running simulation."""

rtd_model.recalculate()


"""Display info about the losses during PCC step from log.

Notes
-----
Various additional process data and profiles are stored within the log.
`examples/models/single_pcc.py` contains an example of plotting
intermediate data, such as concentration profiles during the wash step.

"""


def print_pcc_info():
    print(f"\nDisplaying data about load material"
          f" distribution during the pcc:")
    print(f"--" * 32)
    pcc_data = rtd_model.log.get_data_tree(pcc_pro_a.uo_id)

    for i, cycle_data in enumerate(pcc_data["cycles"]):
        print(f"Cycle {i + 1}:")
        m_load = cycle_data['m_load'][0]
        print(f"Loaded amount:"
              f" {m_load:.0f} mg")
        print(f"Captured protein:"
              f" {cycle_data['m_bound'][0] / m_load * 100:.0f} %")
        print(f"Load recycled during load step:"
              f" {cycle_data['m_load_recycle'][0] / m_load * 100:.0f} %")
        # Wash gets recycled on 2nd column, thus we pull data from next cycle.
        m_wash = pcc_data["cycles"][i + 1]['m_wash_recycle'][0] \
            if i + 1 < len(pcc_data["cycles"]) else 0
        print(f"Load recycled during wash step:"
              f" {m_wash / m_load * 100:.0f} %")
        m_elution = cycle_data['m_elution'][0]
        m_elution_peak = cycle_data['m_elution_peak_cut_loss'][0]
        print(f"Losses due to peak cut:"
              f" {m_elution_peak / m_elution * 100:.0f} %")
        if i == len(pcc_data["cycles"]) - 1:
            print(f"--" * 20)
        else:
            print(f"")


"""Plot flow rate and concentration profile."""


# noinspection DuplicatedCode
def plot_profiles():
    plt_group = []
    for i, uo in enumerate([const_inlet, *dsp_train]):
        f, c = uo.get_result()  # get profiles
        # Prepare figure.
        plt = figure(plot_width=690, plot_height=300, title=uo.gui_title,
                     tools="crosshair,reset,save,wheel_zoom,box_zoom,hover",
                     x_axis_label="t [h]", y_axis_label="c [mg/mL]")
        plt.extra_y_ranges = {'f': Range1d(0, f.max() * 1.1)}
        plt.add_layout(LinearAxis(y_range_name='f'), 'right')
        plt.yaxis[1].axis_label = "f [mL/min]"
        plt.y_range.start = 0
        plt.y_range.end = c.max() * 1.25
        # Plot profiles.
        plt.line(t / 60, c[0], line_width=2, color='navy',
                 legend_label='mAB conc')
        plt.line(t / 60, f, y_range_name='f', color='green',
                 legend_label='flow rate')
        # Add plot to plot list.
        plt.legend.location = "bottom_right"
        plt_group.append(plt)
    # Show plots.
    show(column(*plt_group))


if __name__ == "__main__":
    plot_profiles()
    print_pcc_info()

if __name__ != "examples.models.integrated_mab":
    # Just plot, unless imported as module.
    plot_profiles()
