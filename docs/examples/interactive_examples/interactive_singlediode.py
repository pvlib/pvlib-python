from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Select, CustomJS, ColorBar, LinearColorMapper
)
from bokeh.transform import transform
from bokeh.plotting import figure, show

import pandas as pd
import numpy as np
from pvlib import pvsystem

irradiance_range = np.arange(0, 1300, 50)
temperature_range = np.arange(-10, 70, 10)

parameters = {
    'alpha_sc': 0.004539,
    'a_ref': 2.6373,
    'I_L_ref': 5.114,
    'I_o_ref': 8.196e-10,
    'R_s': 1.065,
    'R_sh_ref': 381.68,
}

dataset = []
for irrad in irradiance_range:
    for temp in temperature_range:

        IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
            irrad,
            temp,
            alpha_sc=parameters['alpha_sc'],
            a_ref=parameters['a_ref'],
            I_L_ref=parameters['I_L_ref'],
            I_o_ref=parameters['I_o_ref'],
            R_sh_ref=parameters['R_sh_ref'],
            R_s=parameters['R_s'],
            EgRef=1.121,
            dEgdT=-0.0002677
        )
        curve_info = pvsystem.singlediode(
            photocurrent=IL,
            saturation_current=I0,
            resistance_series=Rs,
            resistance_shunt=Rsh,
            nNsVth=nNsVth,
            ivcurve_pnts=100,
            method='lambertw'
        )
        data = {
            'Geff': irrad,
            'Tcell': temp,
            'IL': IL,
            'I0': I0,
            'Rs': Rs,
            'Rsh': Rsh,
            'nNsVth': nNsVth,
            'Isc': curve_info['i_sc'],
            'Voc': curve_info['v_oc'],
            'Imp': curve_info['i_mp'],
            'Vmp': curve_info['v_mp'],
            'Pmp': curve_info['p_mp'],
        }
        dataset.append(data)

df = pd.DataFrame(dataset)
source = ColumnDataSource(df)

# default plot is Pmp vs Geff, colored by Tcell
source.data['x'] = source.data['Geff']
source.data['y'] = source.data['Pmp']
source.data['c'] = source.data['Tcell']

colormapper = LinearColorMapper(palette='Viridis256',
                                low=df['Tcell'].min(),
                                high=df['Tcell'].max())

tooltips = [(label, "@"+label) for label in df.columns]

# scatter plot using the 'x', 'y', and 'c' columns
plot = figure(tooltips=tooltips)
plot.circle(source=source, x='x', y='y',
            fill_color=transform('c', colormapper),
            radius=5, radius_units='screen')
plot.xaxis[0].axis_label = 'Geff'
plot.yaxis[0].axis_label = 'Pmp'

color_bar = ColorBar(color_mapper=colormapper, location=(0,0))
plot.add_layout(color_bar, 'right')

# set the x/y/c values when the user changes the selections
callback = CustomJS(args=dict(source=source,
                              colormapper=colormapper,
                              xaxis=plot.xaxis[0],
                              yaxis=plot.yaxis[0]),
                    code="""
    var name = cb_obj.title;
    var src = cb_obj.value;
    if(name == 'X-var:'){
        var dest = 'x';
        xaxis.axis_label = src;
    } else if(name == 'Y-var:'){
        var dest = 'y';
        yaxis.axis_label = src;
    } else if(name == 'Color-var:'){
        var dest = 'c';
        cmap.low = Math.min(source.data[src]);
        cmap.high = Math.max(source.data[src]);
    } else {
        throw "bad source object name!";
    }
    source.data[dest] = source.data[src];
    source.change.emit();
""")

# controls:
options = list(df.columns)
xselect = Select(title="X-var:", value="Geff", options=options)
yselect = Select(title="Y-var:", value="Pmp", options=options)
cselect = Select(title="Color-var:", value="Tcell", options=options)

xselect.js_on_change('value', callback)
yselect.js_on_change('value', callback)
cselect.js_on_change('value', callback)

layout = row(column(xselect, yselect, cselect, width=100), plot)
show(layout)
