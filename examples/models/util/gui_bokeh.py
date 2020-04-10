import numpy as np
from scipy import ndimage
from typing import Union
from functools import partial

from bokeh.io import curdoc, save
from warnings import warn
from bokeh.models.widgets import Div, Button, Slider, RangeSlider
from bokeh.models.widgets import CheckboxGroup, RadioGroup
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, HoverTool
from bokeh.layouts import row, column

from bio_rtd.core import RtdModel, UserInterface, Inlet, UnitOperation


class BokehServerGui(UserInterface):
    # automatically run simulation upon changes to exposed parameters
    auto_recalculate = True

    # recalculate only from i-th unit operation onwards
    start_at = -1

    # style

    line_width = 2
    line_alpha = 0.9

    x_label = 't [min]'
    x_scale_factor = 1
    y_label_c = 'c'
    y_label_f = 'f [mL/min]'
    font_size_pt = 10
    c_y_max = 11

    species_label = []
    flow_label = 'flow rate'

    plot_version = 1
    include_header = True
    include_footer = True

    custom_x_ticks = None
    plot_first_component_as_sum = False
    legend_only_on_first = False

    session = None

    line_colors = ['#3288bd', '#11d594', 'red',
                   '#fee08b', '#fc8d59', '#d53e4f']
    line_dashes = []

    _plot_list = list()
    _plot_root_dict = dict()
    _data_source_list = list()
    _ui_elements_list = list()

    def __init__(self, rtd_model: RtdModel, page_width=1200, plot_points=1000):
        super().__init__(rtd_model)
        self.width = page_width
        self.plot_width = int(page_width * 3 / 4)
        self.controls_width = page_width - self.plot_width
        self.plot_height = int(page_width / 4)
        self._t = self.rtd_model.inlet.get_t()
        self.plot_points = plot_points
        self.dti_plot = max(int(self._t.__len__() / plot_points), 1)

    def build_ui(self):
        curdoc().clear()
        self._plot_list = list()
        self._plot_root_dict = dict()
        self._data_source_list = list()
        self._ui_elements_list = list()
        if self.include_header:
            self._build_header()
        self._build_inlet_gui()
        self._build_uo_list_gui()
        if self.include_footer:
            self._build_footer()

    def _build_header(self):
        header_box = Div(text='<p><h1>' + self.rtd_model.title + '</h1>'
                              + self.rtd_model.desc + '</p>',
                         style={'font-size': str(self.font_size_pt) + 'pt'})
        header_box.width = self.width
        curdoc().add_root(header_box)

    def _generate_is_hidden_text(self, uo_title):
        text_box = Div(text=f'<p style="color:#464646"'
                            f' style="margin-left:50px"><b>'
                            f'{uo_title} < <it>plot is hidden</it> ></b> '
                            f'</p>',
                       style={'font-size': str(self.font_size_pt) + 'pt'})
        text_box.width = self.width
        return text_box

    def _build_inlet_gui(self):
        controls = self._generate_controls(self.rtd_model.inlet)
        source_dict, plot = self._generate_plot(self.rtd_model.inlet) \
            if self.plot_version == 1 \
            else self._generate_plot_v2(self.rtd_model.inlet)
        self._plot_list.append(plot)
        self._data_source_list.append(source_dict)
        if controls.__len__() == 0:
            control_box = Div(width=self.controls_width)
        else:
            control_box = column(*controls, width=self.controls_width)
        element = row(control_box, plot)
        self._plot_root_dict[plot.id] = element.id
        self._ui_elements_list.append(element)
        curdoc().add_root(element)

    def _build_uo_list_gui(self):
        for ui in self.rtd_model.dsp_uo_chain:
            controls = self._generate_controls(ui)
            source_dict, plot = self._generate_plot(ui) \
                if self.plot_version == 1 \
                else self._generate_plot_v2(ui)
            self._plot_list.append(plot)
            self._data_source_list.append(source_dict)
            if controls.__len__() == 0:
                control_box = Div(width=self.controls_width)
            else:
                control_box = column(*controls, width=self.controls_width)
            if ui.gui_hidden:
                element = control_box
            else:
                element = row(control_box, plot, width=self.width)
            self._plot_root_dict[plot.id] = element.id
            self._ui_elements_list.append(element)
            curdoc().add_root(element)

    def _build_footer(self):
        recalculate_btn = Button(label='Recalculate')
        recalculate_btn.on_click(partial(self.recalculate, True))
        auto_re_calc_cb = CheckboxGroup(labels=['Auto-recalculate'],
                                        active=[0])
        auto_re_calc_cb.on_click(self.toggle_auto_recalculate)
        auto_re_calc_cb.align = "center"
        rebuild_ui_btn = Button(label='Rebuild ui')
        rebuild_ui_btn.on_click(self.build_ui)
        curdoc().add_root(row(rebuild_ui_btn,
                              recalculate_btn,
                              auto_re_calc_cb,
                              width=self.controls_width))

    def toggle_auto_recalculate(self, value):
        self.auto_recalculate = 0 in value

    def re_sample(self, d: np.ndarray):
        if self.dti_plot <= 4:
            v = d
        else:
            v = ndimage.gaussian_filter1d(d, int(self.dti_plot / 4))

        if np.array_equal(d, self._t):
            return v[0:-1:self.dti_plot] * self.x_scale_factor
        else:
            return v[0:-1:self.dti_plot]

    def _update_ui_for_uo(self, uo_i, f, c):
        source_i = uo_i + 1
        # update sources
        inlet_source = self._data_source_list[source_i]
        inlet_source['f'].data = dict(t=self.re_sample(self._t),
                                      f=self.re_sample(f))
        self._plot_list[source_i].y_range.end = f.max()
        for i, s in enumerate(inlet_source['c']):
            if self.plot_first_component_as_sum and i == 0:
                inlet_source['c'][i].data = dict(t=self.re_sample(self._t),
                                                 c=self.re_sample(c.sum(0)))
            else:
                inlet_source['c'][i].data = dict(t=self.re_sample(self._t),
                                                 c=self.re_sample(c[i]))
        # set upper plot limit
        self._plot_list[source_i].y_range.end = c.max() * 1.05
        self._plot_list[source_i].extra_y_ranges['f'].end = f.max() * 1.2
        # # update log
        # if hasattr(self.rtd_model.dsp_uo_chain[uo_i], 'log_box'):
        #     self.rtd_model.dsp_uo_chain[uo_i].log_box.text =\
        #         self.get_text_from_log(self.rtd_model.dsp_uo_chain[uo_i].log_data)

    def _toggle_plot_visibility(self, hide, uo_id):
        uo_idx_list = [i for i, v in enumerate(self.rtd_model.dsp_uo_chain)
                       if v.uo_id == uo_id][0] + 1
        figure_id = self._plot_list[uo_idx_list].id
        root_id = self._plot_root_dict[figure_id]
        current_row = curdoc().get_model_by_id(root_id)
        if hide:
            if current_row.children[-1].id == figure_id:
                current_row.children.pop()
                current_row.children.append(
                    self._generate_is_hidden_text(
                        self.rtd_model.dsp_uo_chain[uo_idx_list - 1].title))
        else:
            if current_row.children[-1].id != figure_id:
                current_row.children.pop()
                current_row.children.append(self._plot_list[uo_idx_list])
        save(current_row)

    @staticmethod
    def setattr_with_index(obj, attr: str, val):
        if attr[-1] is ']':
            idx = int(attr[attr.find('[') + 1:-1])
            attr = attr[:attr.find('[')]
            attr_v = getattr(obj, attr)
            attr_v[idx] = val
        else:
            return setattr(obj, attr, val)

    # noinspection PyUnusedLocal
    def _mutable_parameters_callback(self,
                                     uo_id,
                                     par_idx,
                                     sf,
                                     attr_name,
                                     old_value,
                                     new_values):
        if not new_values.__class__ == tuple:
            new_values = [new_values]
        if sf:
            new_values = [sf * v for v in new_values]
        if uo_id == self.rtd_model.inlet.uo_id:
            var_list = self.rtd_model.inlet.adj_par_list[par_idx].var_list
            for i in range(new_values.__len__()):
                self.setattr_with_index(self.rtd_model.inlet,
                                        var_list[i],
                                        new_values[i])
                if var_list[i] == 'hidden':
                    self._toggle_plot_visibility(new_values[i], uo_id)
                else:
                    self.start_at = -1
        else:
            uo_idx_list = \
                [i for i, v in enumerate(self.rtd_model.dsp_uo_chain)
                 if v.uo_id == uo_id]
            if len(uo_idx_list) == 0:
                warn(f'Parameter could not be changed'
                     f' for {uo_id} - target missing.')
                return
            else:
                uo_idx = uo_idx_list[0]
            var_list = self.rtd_model.dsp_uo_chain[uo_idx]\
                .adj_par_list[par_idx].var_list
            for i in range(new_values.__len__()):
                self.setattr_with_index(self.rtd_model.dsp_uo_chain[uo_idx],
                                        var_list[i],
                                        new_values[i])
                if var_list[i] == 'hidden':
                    self._toggle_plot_visibility(new_values[i], uo_id)
                else:
                    self.rtd_model.dsp_uo_chain[uo_idx].new_config = True
                    self.start_at = min(self.start_at, uo_idx)
        # Recalculate if needed.
        if self.auto_recalculate:
            self.recalculate()

    def _mutable_checkbox_callback(self, uo_id, par_idx, n_par, active):
        for i in range(n_par):
            self._mutable_parameters_callback(uo_id, par_idx, None,
                                              [], [], i in active)

    def _mutable_radio_group_callback(self, uo_id, par_idx, n_par, active):
        for i in range(n_par):
            self._mutable_parameters_callback(uo_id, par_idx, None,
                                              [], [], i == active)

    @staticmethod
    def getattr_with_index(obj, attr: str):
        if attr[-1] is ']':
            idx = int(attr[attr.find('[') + 1:-1])
            attr = attr[:attr.find('[')]
            return getattr(obj, attr)[idx]
        else:
            return getattr(obj, attr)

    def _generate_controls(self, uo: Union[UnitOperation, Inlet]):
        controls = list()
        for i, r in enumerate(uo.adj_par_list):
            if r.gui_type == 'slider':
                if r.v_init:
                    initial_value = r.v_init
                else:
                    initial_value = self.getattr_with_index(uo, r.var_list[0])
                    if r.scale_factor:
                        initial_value /= r.scale_factor

                slider = Slider(title=r.par_name, value=initial_value,
                                start=r.v_range[0], end=r.v_range[1],
                                step=r.v_range[2])

                slider.on_change('value',
                                 partial(self._mutable_parameters_callback,
                                         uo.uo_id,
                                         i,
                                         r.scale_factor))
                controls.append(slider)
            elif r.gui_type == 'range':

                if r.v_init:
                    initial_value = r.v_init
                else:
                    initial_value = (
                        self.getattr_with_index(uo, r.var_list[0]),
                        self.getattr_with_index(uo, r.var_list[1])
                    )
                    if r.scale_factor:
                        initial_value = \
                            [v / r.scale_factor for v in initial_value]

                slider = RangeSlider(title=r.par_name, value=initial_value,
                                     start=r.v_range[0], end=r.v_range[1],
                                     step=r.v_range[2])
                slider.on_change('value',
                                 partial(self._mutable_parameters_callback,
                                         uo.uo_id,
                                         i,
                                         r.scale_factor))
                controls.append(slider)
            elif r.gui_type in ['checkbox', 'radio_group']:
                attrs = list(r.var_list) if hasattr(r.var_list, '__len__') \
                    else [r.var_list]
                if r.par_name is None:
                    pars = attrs
                else:
                    pars = list(r.par_name) if type(r.par_name) is tuple \
                        else [r.par_name]

                on_list = [i for i in range(len(pars)) if r.v_init[i]] \
                    if type(r.v_init) is tuple \
                    else list(range(len(pars))) if r.v_init is True \
                    else [] if r.v_init is False \
                    else [getattr(uo, attr) for attr in attrs]

                if r.gui_type == 'checkbox':
                    uie = CheckboxGroup(labels=pars, active=on_list)
                else:
                    active = on_list.index(True) if True in on_list else 0
                    uie = RadioGroup(labels=pars, active=active)

                uie.on_click(partial(self._mutable_checkbox_callback,
                                     uo.uo_id,
                                     i,
                                     len(pars)))
                controls.append(uie)
        return controls

    # noinspection DuplicatedCode
    def _generate_plot(self, uo):
        tmp_t = self.re_sample(self._t)
        tmp_zero = tmp_t.copy() * 0
        # Generate plot.
        plot = figure(plot_height=self.plot_height,
                      plot_width=self.plot_width,
                      title=uo.gui_title,
                      tools="crosshair,reset,save,wheel_zoom,box_zoom",
                      x_range=[0, self._t[-1] * self.x_scale_factor],
                      y_range=[0, self.c_y_max], output_backend='svg')
        plot.xaxis.axis_label = self.x_label
        plot.yaxis.axis_label = self.y_label_c
        if self.custom_x_ticks is not None:
            plot.xaxis.ticker = self.custom_x_ticks
        plot.hover.mode = 'vline'
        # add extra axis for flow
        plot.extra_y_ranges = {
            'f': Range1d(
                start=0,
                end=self.rtd_model.inlet.get_result()[0].max() * 1.05)}
        plot.add_layout(LinearAxis(y_range_name='f'), 'right')
        plot.yaxis[1].axis_label = self.y_label_f
        # generate sources and lines
        source_dict = dict()
        source_dict['f'] = ColumnDataSource(data=dict(t=tmp_t, f=tmp_zero))
        lf = plot.line('t', 'f',
                       source=source_dict['f'],
                       line_width=self.line_width,
                       line_alpha=self.line_alpha,
                       line_color=self.line_colors[0],
                       line_dash=self.line_dashes[0]
                       if len(self.line_dashes) > 0 else 'solid',
                       y_range_name='f',
                       legend_label=self.flow_label)
        plot.hover.renderers = [lf]
        source_dict['c'] = list()
        ht_c = HoverTool(tooltips=[('c', '@c{0.2f}')], mode='vline')
        for i in range(self.rtd_model.inlet.get_n_species()):
            source_dict['c'].append(ColumnDataSource(
                data=dict(t=self.re_sample(self._t), c=tmp_zero)
            ))
        rs = []
        for i, s in enumerate(source_dict['c']):
            cl = plot.line('t', 'c',
                           source=s,
                           line_width=self.line_width,
                           line_alpha=self.line_alpha,
                           line_color=self.line_colors[1 + i],
                           line_dash=self.line_dashes[1 + i]
                           if len(self.line_dashes) > 1 + i else 'solid',
                           legend_label=self.species_label[i]
                           if i < len(self.species_label) else '')
            rs.append(cl)
        ht_c.renderers = rs
        plot.add_tools(ht_c)
        plot.toolbar.logo = None
        font_size = str(self.font_size_pt) + 'pt'
        if isinstance(uo, Inlet) or not self.legend_only_on_first:
            plot.legend.location = 'center_right'
            plot.legend.title_text_font_size = font_size
            plot.legend.label_text_font_size = font_size
        else:
            plot.legend.visible = False

        # set font sizes
        plot.xaxis.axis_label_text_font_size = font_size
        plot.xaxis.major_label_text_font_size = font_size
        plot.yaxis.axis_label_text_font_size = font_size
        plot.yaxis.major_label_text_font_size = font_size
        plot.title.text_font_size = font_size

        # generate lines
        # return references
        return source_dict, plot

    # noinspection DuplicatedCode
    def _generate_plot_v2(self, uo):
        tmp_t = self.re_sample(self._t)
        tmp_zero = tmp_t.copy() * 0
        # generate plot
        plot = figure(plot_height=self.plot_height,
                      plot_width=self.plot_width,
                      title=uo.gui_title,
                      tools="save",
                      x_range=[0, self._t[-1] * self.x_scale_factor],
                      y_range=[0, self.c_y_max], output_backend='svg')
        plot.xaxis.axis_label = self.x_label
        plot.yaxis.axis_label = self.y_label_c
        plot.xaxis.ticker = FixedTicker(ticks=[0])
        plot.yaxis.ticker = FixedTicker(ticks=[0])
        plot.hover.mode = 'vline'
        # add extra axis for flow
        plot.extra_y_ranges = {
            'f': Range1d(
                start=0,
                end=self.rtd_model.inlet.get_result()[0].max() * 1.05)}
        plot.add_layout(LinearAxis(y_range_name='f'), 'right')
        plot.yaxis[1].visible = False
        # generate sources and lines
        source_dict = dict()
        source_dict['f'] = ColumnDataSource(data=dict(t=tmp_t, f=tmp_zero))
        lf = plot.line('t', 'f',
                       source=source_dict['f'],
                       line_width=self.line_width,
                       line_alpha=self.line_alpha,
                       line_color=self.line_colors[0],
                       line_dash=self.line_dashes[0]
                       if len(self.line_dashes) > 0 else 'solid',
                       y_range_name='f',
                       legend_label=self.flow_label)
        plot.hover.renderers = [lf]
        source_dict['c'] = list()
        for i in range(self.rtd_model.inlet.get_n_species()):
            source_dict['c'].append(ColumnDataSource(
                data=dict(t=self.re_sample(self._t), c=tmp_zero)
            ))
        rs = []
        for i, s in enumerate(source_dict['c']):
            cl = plot.line('t', 'c',
                           source=s,
                           line_width=self.line_width,
                           line_alpha=self.line_alpha,
                           line_color=self.line_colors[1 + i],
                           line_dash=self.line_dashes[1 + i]
                           if len(self.line_dashes) > 1 + i else 'solid',
                           legend_label=self.species_label[i]
                           if i < len(self.species_label) else '')
            rs.append(cl)
        plot.toolbar.logo = None

        # generate lines
        # return references
        return source_dict, plot
