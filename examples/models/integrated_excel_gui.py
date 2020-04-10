"""RtdModel with process data read from Excel document

In this example we define a model of another hypothetical mAB process
defined in Excel spreadsheet: `examples/models/integrated_excel.xlsx`.

Step-by-step:
#. Reading parameters from Excel.
#. Binding read parameters to process `PARAMETERS` and `ATTRIBUTES`
    based on templates (`examples/templates/`) for each unit operation.
#. Instantiate `UnitOperation`s and `Inlet`.
#. Creating `RtdModel` from `Inlet` and list of unit operations.

In this example we build a GUI on top of the model. GUI is run on
`bokeh` server. Once the GUI is running, one can simply modify a value
of a parameter in spreadsheet file and refresh web page. Refreshing
web page will re-run the model and thus 'update' is based on new data.

Notes
-----
Excel spreadsheet file does not depend (of include information) on the
RTD modeling approach. It simply serves as a data holder, thus if user
adds any new parameter in spreadsheet file, then the data binding part
of the script needs to be updated accordingly.

Interactive Bokeh GUI can easily be removed from this example and
replaced by plots and/or reports as in `examples/integrated_mab.py`.

"""
import inspect
import pathlib
import numpy as np
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure

from bio_rtd import pdf, peak_fitting
from bio_rtd.uo import sc_uo, fc_uo, surge_tank, special_uo
from bio_rtd.core import RtdModel
from bio_rtd.inlet import IntervalInlet
from bio_rtd.logger import DataStoringLogger
from bio_rtd.adj_par import AdjParRange, AdjParSlider, AdjParBoolean
from bio_rtd.chromatography.bt_load import ConstantPatternSolution

from examples.models.util.gui_bokeh import BokehServerGui
from examples.models.util.xlsx_data import read_bio_process_xlsx_data


def create_uo_pars_and_attrs(t: np.ndarray, _uo_lib: dict):
    """Adds methods, parameters and attributes to `uo_list`."""

    # Assert proper simulation time vector.
    assert t[0] == 0
    dt = t[-1] / (t.size - 1)  # simulation time step

    def atd(uo: dict):
        """Function adds parameters and attributes to the uo dict.

        Copy dictionary content template for each unit operation from
        templates (found in `examples/templates/`).

        """
        d = uo['data']
        v_void = d['void volume [mL]']
        v_sigma = d['RTD sigma [mL]']

        uo['uo_class'] = fc_uo.FlowThrough
        uo['parameters'] = {
            "uo_id": uo['id'],
            "pdf": pdf.GaussianFixedRelativeWidth(t, v_sigma / v_void),
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "v_void": v_void,
        }

    def df(uo: dict):
        d = uo['data']
        v_void = d['void volume [mL]']
        v_sigma = d['RTD sigma [mL]']
        v_switch = d['switch volume [L]'] * 1000  # L -> mL

        uo['uo_class'] = fc_uo.FlowThroughWithSwitching
        uo['parameters'] = {
            "uo_id": uo['id'],
            "pdf": pdf.GaussianFixedRelativeWidth(t, v_sigma / v_void),
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "v_void": v_void,
            "v_cycle": v_switch,
        }

    def cvi_column(uo: dict):
        d = uo['data']
        rt_target = d['mean RT [min]']
        t_sigma = d['peak sigma [min]']

        uo['uo_class'] = fc_uo.FlowThrough
        uo['parameters'] = {
            "uo_id": uo['id'],
            "pdf": pdf.GaussianFixedDispersion(t, t_sigma ** 2 / rt_target),
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "rt_target": rt_target,
        }

    def ftc_aex(uo: dict):
        d = uo['data']
        cv = d['CV [mL]']
        v_void = cv * d['porosity protein []']
        v_peak = d['peak position [mL]']
        v_peak_sigma = d['peak sigma [mL]']
        v_cycle = cv * d['life cycle [CV]']

        uo['uo_class'] = fc_uo.FlowThroughWithSwitching
        uo['parameters'] = {
            "uo_id": uo['id'],
            "pdf": pdf.GaussianFixedRelativeWidth(t,
                                                  v_peak_sigma / v_peak),
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "v_void": v_void,
            "v_cycle": v_cycle,
        }

    def st_single(uo: dict):
        d = uo['data']
        v_min_rel = d['min fill level rel []']
        starts_empty = d['starts empty []']

        uo['uo_class'] = surge_tank.CSTR
        uo['parameters'] = {
            "uo_id": uo['id'],
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "v_min_ratio": v_min_rel,
            "starts_empty": starts_empty,
        }

    def uf_df(uo: dict):
        d = uo['data']
        n_tanks = d['n tanks []']
        rt = d['residence time [min]']
        volume_reduction = d['volume reduction []']
        t_switch = d['switch time [min]']
        efficiency = d['efficiency []']

        concentration = fc_uo.Concentration(
            t, flow_reduction=volume_reduction,
            uo_id=f"{uo['id']}_concentration")
        buffer_exchange = fc_uo.BufferExchange(
            t, exchange_ratio=efficiency,
            uo_id=f"{uo['id']}_buffer_exchange")
        flow_through = fc_uo.FlowThroughWithSwitching(
            t, pdf=pdf.TanksInSeries(t, n_tanks),
            uo_id=f"{uo['id']}_rtd")
        flow_through.rt_target = rt
        flow_through.t_cycle = t_switch

        uo['uo_class'] = special_uo.ComboUO
        uo['parameters'] = {
            "uo_id": uo['id'],
            "sub_uo_list": [concentration, buffer_exchange, flow_through],
            "gui_title": uo['title'],
        }
        uo['attributes'] = {}

    # noinspection DuplicatedCode
    def acc_cex(uo: dict):
        d = uo['data']
        # Column volume.
        cv = d['CV [mL]']
        # Operating linear flow rate during load.
        v_lin_load = d['load flowrate [cm/h]'] / 60  # cm/min
        # Other steps info (durations and flow rates).
        f_eq_rel = d['equilibration / load flowrate []']
        f_wash_rel = d['wash / load flowrate []']
        f_elution_rel = d['elution / load flowrate []']
        f_reg_rel = d['regeneration / load flowrate []']
        v_eq_cv = d['equilibration [CV]']
        v_wash_cv = d['wash [CV]']
        v_elution_cv = d['elution [CV]']
        v_reg_cv = d['regeneration [CV]']
        # Switch columns at x % protein breakthrough during load.
        load_switch_c_rel = d['load outlet conc ratio []']
        # Breakthrough profile.
        dbc_10 = d['DBC_10% [mg/mL]']
        dbc_100 = d['DBC_100% [mg/mL]']
        k = np.log(9) / (dbc_100 - dbc_10)
        # Elution peak (experimental data with column of different size).
        ep_rt, ep_sigma, ep_skew = \
            peak_fitting.calc_emg_parameters_from_peak_shape(
                t_peak_start=d['peak_a [mL]'],
                t_peak_max=d['peak_max [mL]'],
                t_peak_end=d['peak_b [mL]'],
                relative_threshold=d['peak_ab / peak_max []'])
        v_elution_peak_cv = ep_rt / d['peak_eval_CV [mL]']
        # Elution peak cut.
        el_peak_cut_start_rel = d['peak_start / peak_max []']
        el_peak_cut_end_rel = d['peak_end / peak_max []']
        unaccounted_losses = d['unaccounted_losses []']

        uo['uo_class'] = sc_uo.ACC
        uo['parameters'] = {
            "uo_id": uo['id'],
            "load_bt": ConstantPatternSolution(dt, dbc_100=dbc_100, k=k),
            "peak_shape_pdf": pdf.ExpModGaussianFixedRelativeWidth(
                t, sigma_relative=ep_sigma / ep_rt,
                tau_relative=1 / ep_skew / ep_rt),
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "cv": cv,
            "equilibration_cv": v_eq_cv,
            "equilibration_f_rel": f_eq_rel,
            "load_c_end_relative_ss": load_switch_c_rel,
            "load_c_end_estimate_with_iterative_solver": True,
            "load_target_lin_velocity": v_lin_load,
            "wash_cv": v_wash_cv,
            "wash_f_rel": f_wash_rel,
            "unaccounted_losses_rel": unaccounted_losses,
            "elution_cv": v_elution_cv,
            "elution_f_rel": f_elution_rel,
            "elution_peak_position_cv": v_elution_peak_cv,
            "elution_peak_cut_start_c_rel_to_peak_max": el_peak_cut_start_rel,
            "elution_peak_cut_end_c_rel_to_peak_max": el_peak_cut_end_rel,
            "regeneration_cv": v_reg_cv,
            "regeneration_f_rel": f_reg_rel,
        }

    # noinspection DuplicatedCode
    def pcc_pro_a(uo: dict):
        d = uo['data']
        # Column volume.
        cv = d['CV [mL]']
        # Operating linear flow rate during load.
        v_lin_load = d['load flowrate [cm/h]'] / 60  # cm/min
        # Other steps info (durations and flow rates).
        f_eq_rel = d['equilibration / load flowrate []']
        f_wash_rel = d['wash / load flowrate []']
        f_elution_rel = d['elution / load flowrate []']
        f_reg_rel = d['regeneration / load flowrate []']
        v_eq_cv = d['equilibration [CV]']
        v_wash_cv = d['wash [CV]']
        v_elution_cv = d['elution [CV]']
        v_reg_cv = d['regeneration [CV]']
        # Switch columns at x % protein breakthrough during load.
        load_switch_c_rel = d['load col1 outlet conc ratio []']
        # Breakthrough profile.
        dbc_10 = d['DBC_10% [mg/mL]']
        dbc_100 = d['DBC_100% [mg/mL]']
        k = np.log(9) / (dbc_100 - dbc_10)
        # Elution peak (experimental data with column of different size).
        ep_rt, ep_sigma, ep_skew = \
            peak_fitting.calc_emg_parameters_from_peak_shape(
                t_peak_start=d['peak_a [mL]'],
                t_peak_max=d['peak_max [mL]'],
                t_peak_end=d['peak_b [mL]'],
                relative_threshold=d['peak_ab / peak_max []'])
        v_elution_peak_cv = ep_rt / d['peak_eval_CV [mL]']
        # Elution peak cut.
        el_peak_cut_start_cv = d['peak collection start [CV]']
        el_peak_cut_end_cv = d['peak collection end [CV]']
        protein_porosity = d['porosity protein []']
        hetp = d['HETP [cm]']
        unaccounted_losses = d['unaccounted_losses []']
        wash_recycle = d['recycle_wash []']
        extend_first_cycle = d['extend first load []']

        uo['uo_class'] = sc_uo.PCC
        uo['parameters'] = {
            "uo_id": uo['id'],
            "load_bt": ConstantPatternSolution(dt, dbc_100=dbc_100, k=k),
            "peak_shape_pdf": pdf.ExpModGaussianFixedRelativeWidth(
                t,
                sigma_relative=ep_sigma / ep_rt,
                tau_relative=1 / ep_skew / ep_rt),
            "load_recycle_pdf": pdf.GaussianFixedDispersion(
                t, dispersion_index=hetp / v_lin_load),
            # Porosity of the column for protein.
            "column_porosity_retentate": protein_porosity,
            "gui_title": uo['title'],
        }
        uo['attributes'] = {
            "cv": cv,
            "equilibration_cv": v_eq_cv,
            "equilibration_f_rel": f_eq_rel,
            "load_c_end_relative_ss": load_switch_c_rel,
            "load_c_end_estimate_with_iterative_solver": True,
            "load_extend_first_cycle": extend_first_cycle,
            "load_target_lin_velocity": v_lin_load,
            "wash_cv": v_wash_cv,
            "wash_f_rel": f_wash_rel,
            "wash_recycle": wash_recycle,
            "unaccounted_losses_rel": unaccounted_losses,
            "elution_cv": v_elution_cv,
            "elution_f_rel": f_elution_rel,
            "elution_peak_position_cv": v_elution_peak_cv,
            "elution_peak_cut_start_cv": el_peak_cut_start_cv,
            "elution_peak_cut_end_cv": el_peak_cut_end_cv,
            "regeneration_cv": v_reg_cv,
            "regeneration_f_rel": f_reg_rel,
        }

    # Map method names to functions.
    method_2_func = {
        'ATD': atd,
        'DF': df,
        'PCC_ProA': pcc_pro_a,
        'ST_Single': st_single,
        'CVI_Column': cvi_column,
        'ACC_Cex': acc_cex,
        'FTC_Aex': ftc_aex,
        'UFDF': uf_df,
    }

    try:
        for _uo in _uo_lib.values():
            method_2_func.get(_uo['method'])(_uo)
    except TypeError:
        # handle errors
        # find missing methods
        mm = set([_uo['method'] for _uo in _uo_lib.values()
                  if _uo['method'] not in method_2_func.keys()])
        if mm.__len__() > 0:
            raise NameError(f"No parsing functions for methods:"
                            f" `{'`, `'.join(mm)}`")
        else:
            raise
    except KeyError as ke:
        # noinspection PyUnboundLocalVariable
        raise KeyError(f"Unit operation: `{_uo['id']}"
                       f"` with title: `{_uo['title']}"
                       f"` is missing filed: `{ke.args[0]}`")


def generate_dsp_uo_train(t, _uo_list, _uo_lib):
    _dsp_train = []
    for uo_id in _uo_list:
        uo_pars = _uo_lib[uo_id]['parameters']
        uo_attr = _uo_lib[uo_id]['attributes']
        uo_class = _uo_lib[uo_id]['uo_class']
        uo_instance = uo_class(t, **uo_pars)
        for key, value in uo_attr.items():
            # Make sure attribute exist.
            assert hasattr(uo_instance, key), f"`{key}` is wrong."
            # Override value.
            setattr(uo_instance, key, value)
        _dsp_train.append(uo_instance)
    return _dsp_train


def generate_inlet(t, inlet_data, species):
    inlet_c = inlet_data['Titer [g/L]']  # [mg/mL]
    inlet_f = inlet_data['Flow [RV/day]'] * inlet_data['V [L]']
    inlet_f *= 1000 / 24 / 60  # [L/day] -> [mL/min]
    inlet = IntervalInlet(
        t=t, f=inlet_f,
        c_inner=np.array([0.0, inlet_c]),  # modification not in excel
        c_outer=np.array([inlet_c, 0.0]),  # modification not in excel
        species_list=species,
        inlet_id="usp",
        # gui_title=inlet_data['Title'],
        gui_title="Inlet",
    )
    # Modifications not in Excel.
    inlet.t_start = 60 * 5
    inlet.t_end = 30 + inlet.t_start
    return inlet


# noinspection DuplicatedCode
def plot_profiles(_rtd_model):
    plt_group = []
    for i, uo in enumerate([_rtd_model.inlet, *_rtd_model.dsp_uo_chain]):
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
        plt.line(_rtd_model.inlet.get_t() / 60, c[0],
                 line_width=2, color='green', legend_label='product')
        plt.line(_rtd_model.inlet.get_t() / 60, c[1], line_width=2,
                 color='red', legend_label='section of product')
        plt.line(_rtd_model.inlet.get_t() / 60, f, y_range_name='f',
                 color='blue', legend_label='flow rate')
        # Add plot to plot list.
        plt.legend.location = "bottom_right"
        plt_group.append(plt)
    # Show plots.
    show(column(*plt_group))


def add_adj_pars(_rtd_model):
    """Add adjustable parameters, exposed to gui."""
    # Inlet.
    uo = _rtd_model.inlet
    uo.adj_par_list = [
        AdjParRange(var_list=('t_start', 't_end'),
                    v_range=(0, _rtd_model.inlet.get_t()[-1], 30),
                    par_name='Inlet interval [min]'),
        AdjParSlider(var='c_outer[0]',
                     v_range=(0, 10, 0.5),
                     par_name='Titer outside interval'),
        AdjParSlider(var='c_inner[1]',
                     v_range=(0, 10, 0.5),
                     par_name='Titer in interval'),
    ]
    # PCC.
    _rtd_model.get_dsp_uo('pro_a_pcc').adj_par_list = [
        AdjParSlider(
            var='cv',
            v_range=(50, 500, 50),
            par_name='Column volume [mL]')
    ]
    # Surge tank 1.
    uo = _rtd_model.get_dsp_uo('surge_tank_1')
    uo.adj_par_list = [
        AdjParBoolean(
            var='starts_empty',
            par_name='Starts empty',
            v_init=uo.starts_empty),
        AdjParSlider(
            var='v_min_ratio',
            v_range=(0, 100, 5),
            par_name='Minimum fill level [%]',
            scale_factor=0.01)
    ]
    # FTC_AEX.
    try:
        uo = _rtd_model.get_dsp_uo('aex_ftc')
        uo.adj_par_list = [
            AdjParSlider(
                var='v_void',
                v_range=(1, 30, 1),
                scale_factor=0.8,  # compensation for porosity
                par_name='Column volume [mL]')
        ]
    except KeyError:  # uo might not be present in current scenario
        pass
    # Surge tank 2.
    try:
        uo = _rtd_model.get_dsp_uo('surge_tank_2')
        uo.adj_par_list = [
            AdjParBoolean(
                var='starts_empty',
                par_name='Starts empty',
                v_init=uo.starts_empty),
            AdjParSlider(
                var='v_min_ratio',
                v_range=(0, 100, 5),
                par_name='Minimum fill level [%]',
                scale_factor=0.01)
        ]
    except KeyError:  # uo might not be present in current scenario
        pass


def customize_gui(_gui):
    # customize GUI
    _gui.plot_height = 280
    _gui.line_colors = ['#3288bd', 'green', 'red']
    _gui.x_scale_factor = 1 / 60
    _gui.x_label = 't [h]'
    _gui.y_label_f = 'f [mL/min]'
    _gui.y_label_c = 'c [mg/mL]'
    _gui.custom_x_ticks = np.arange(25)
    _gui.legend_only_on_first = True
    _gui.dti_plot = 1
    _gui.plot_first_component_as_sum = False
    _gui.line_dashes = ['dashed']
    _gui.font_size_pt = 14


def generate_up_rtd_model():
    # Define species names and simulation time vector.
    dt = 0.5  # min
    t = np.arange(0, 24.1 * 60, dt)
    species = ['product', 'section of product']
    # Choose scenario for USP and DSP.
    usp_scenario = 1
    dsp_scenario = 'A'

    # Read data from Excel.
    uo_lib, dsp, usp = read_bio_process_xlsx_data(
        pathlib.Path(__file__).parent / 'integrated_excel.xlsx'
    )
    # Determine parameters and attributes for unit operations.
    create_uo_pars_and_attrs(t, uo_lib)
    # Generate inlet.
    inlet = generate_inlet(t, usp[usp_scenario], species)
    # Generate DSP train.
    dsp_train = generate_dsp_uo_train(t, dsp[dsp_scenario], uo_lib)
    # Create `RtdModel`.
    _rtd_model = RtdModel(inlet, dsp_train, DataStoringLogger(),
                          'Sample integrated process',
                          'Data was sourced from Excel file')
    # Add Adjustable parameters to the model.
    add_adj_pars(_rtd_model)
    return _rtd_model


rtd_model = generate_up_rtd_model()

if __name__[:9] == "bokeh_app" \
        and "sphinx" not in inspect.stack()[-1].filename:
    # Create GUI.
    gui = BokehServerGui(rtd_model)
    # Customize GUI.
    customize_gui(gui)
    # Build GUI.
    gui.build_ui()
    # Run simulation.
    gui.recalculate(True)
else:
    rtd_model.log.log_level = rtd_model.log.ERROR
    rtd_model.recalculate(-1)
    plot_profiles(rtd_model)
