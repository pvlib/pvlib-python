from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Slider, CustomJS, Span
)
from bokeh.plotting import figure, show

import pandas as pd
import numpy as np
import pvlib

times = pd.date_range('2019-06-01 05:15', '2019-06-01 21:00',
                      freq='1min', closed='left', tz='US/Eastern')
location = pvlib.location.Location(40, -80)
solpos = location.get_solarposition(times)

theta = pvlib.tracking.singleaxis(
    solpos['zenith'],
    solpos['azimuth'],
    axis_tilt=0,
    axis_azimuth=0,
    max_angle=90,
    backtrack=True,
    gcr=0.5,
)['tracker_theta'].fillna(0)

# prevent weird shadows at night
solpos['elevation'] = solpos['elevation'].clip(lower=0)

curves_source = ColumnDataSource(data={
    'times': times,
    'tracker_theta': theta,  # degrees
    'solar_elevation': np.radians(solpos['elevation']),
    'solar_azimuth': np.radians(solpos['azimuth']),
})

source = ColumnDataSource(data={
    # backtracking positions
    'tracker1_x': [-1.5, -0.5],
    'tracker1_y': [0.0, 0.0],
    'tracker2_x': [0.5, 1.5],
    'tracker2_y': [0.0, 0.0],

    # ground line
    'ground_x': [-3, 3],
    'ground_y': [-1, -1],

    # racking posts
    'post1_x': [-1, -1],
    'post1_y': [-1, 0],
    'post2_x': [1, 1],
    'post2_y': [-1, 0],
})

shadow_source = ColumnDataSource(data={
    'xs': [[]],
    'ys': [[]],
})

# Set up Scene diagram
scene = figure(plot_height=350, plot_width=350, title="Tracker Position",
               x_range=[-3, 3], y_range=[-3, 3])

plot_args = dict(
    source=source,
    line_width=3,
    line_alpha=1.0,
)

scene.line('tracker1_x', 'tracker1_y', **plot_args)
scene.line('tracker2_x', 'tracker2_y', **plot_args)

scene.line('ground_x', 'ground_y', source=source, line_width=1, line_color='black')
scene.line('post1_x', 'post1_y', source=source, line_width=5, line_color='black')
scene.line('post2_x', 'post2_y', source=source, line_width=5, line_color='black')

scene.patches(xs="xs", ys="ys", fill_color="#333333", alpha=0.2, source=shadow_source)

# Set up daily tracker angle curve
curves = figure(plot_height=350, plot_width=350, title="Tracker Angle",
                x_axis_type='datetime')

curves_args = dict(
    source=curves_source,
    line_width=3,
    line_alpha=1.0,
)
curves.line('times', 'tracker_theta', **curves_args)

scrubber = Span(location=0,
                dimension='height', line_color='black',
                line_dash='dashed', line_width=3)
curves.add_layout(scrubber)

time_slider = Slider(title="Minute of Day", value=0, start=0, end=len(times)-1, step=1)

time_slider.callback = CustomJS(args=dict(span=scrubber,
                                          slider=time_slider,
                                          curves_source=curves_source,
                                          position_source=source,
                                          shadow_source=shadow_source),
                                code="""
    // update time scrubber position and scene geometry

    // update time scrubber position
    var idx = slider.value;
    span.location = curves_source.data['times'][idx];

    // update tracker positions
    var tracker_theta = curves_source.data['tracker_theta'][idx];
    var dx = 0.5 * Math.cos(3.14159/180 * tracker_theta);
    var dy = 0.5 * Math.sin(3.14159/180 * tracker_theta);
    var data = position_source.data;
    data['tracker1_x'] = [-1 - dx, -1 + dx];
    data['tracker1_y'] = [ 0 - dy,  0 + dy];
    data['tracker2_x'] = [ 1 - dx,  1 + dx];
    data['tracker2_y'] = [ 0 - dy,  0 + dy];
    position_source.change.emit();

    // update shadow patches
    var solar_elevation = curves_source.data['solar_elevation'][idx];
    var solar_azimuth = curves_source.data['solar_azimuth'][idx];
    var z = Math.sin(solar_elevation);
    var x = -Math.cos(solar_elevation)*Math.sin(solar_azimuth);

    // shadow length
    z = z * 10;
    x = x * 10;

    var shadow1_x = [
            data['tracker1_x'][0],
            data['tracker1_x'][1],
            data['tracker1_x'][1] - x,
            data['tracker1_x'][0] - x
    ];
    var shadow1_y = [
            data['tracker1_y'][0],
            data['tracker1_y'][1],
            data['tracker1_y'][1] - z,
            data['tracker1_y'][0] - z
    ];
    var shadow2_x = [
            data['tracker2_x'][0],
            data['tracker2_x'][1],
            data['tracker2_x'][1] - x,
            data['tracker2_x'][0] - x
    ];
    var shadow2_y = [
            data['tracker2_y'][0],
            data['tracker2_y'][1],
            data['tracker2_y'][1] - z,
            data['tracker2_y'][0] - z
    ];
    shadow_source.data = {
        'xs': [shadow1_x, shadow2_x],
        'ys': [shadow1_y, shadow2_y],
    };
    shadow_source.change.emit();

""")

layout = column(
    time_slider,
    row(scene, curves)
)

show(layout)
