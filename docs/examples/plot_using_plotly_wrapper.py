"""
Plotly Wrapper for pvlib
====================

Example of preparing plots using clear sky model as an example
"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
from pvlib.location import Location


def make_plotly_figure(template='gridon', rows=1, columns=1, column_widths=None, title='plotly_figure',
                       x_axis_title='x-axis-title', y_axis_title='y-axis-title'):
    """
    This function is a helper that builds a template figure using the plotly library that can be further
    customized as needed by the user.

    :param str template: currently supports plotly default templates. acceptable arguments are:
    ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
    'ygridoff', 'gridon', 'none']. Defaults to 'gridon'
    :param int rows: indicates how many vertical subplots are desired. Defaults to 1
    :param int columns: indicates how many horizontal subplots are desired. Defaults to 1
    :param list column_widths: helps set the width of the subplots when there are multiple. Defaults to None
    :param str title: name of the figure. Defaults to 'plotly_figure'
    :param x_axis_title: name of x-axis title. Defaults to 'x-axis'
    :param y_axis_title: name of y-axis title. Defaults to 'y-axis'
    :return: plotly figure object
    :rtype: plotly.graph_object
    """

    # Set template for plotly figure
    pio.templates.default = template

    # Create figure object
    # Test whether subplots are desired or single plot is desired
    if rows > 1 or columns > 1:
        # Test whether size of each subplot has been specified or if defaults can be used
        if column_widths is None:
            figure = make_subplots(rows=rows, cols=columns, column_widths=[0.7, 0.3])
        else:
            if max([rows, columns]) == len(column_widths):
                figure = make_subplots(rows=rows, cols=columns, column_widths=column_widths)
            else:
                raise ValueError(
                    '\n In order to customize subplots, require same # of widths as the # of unique subplots ')
    else:
        figure = go.Figure()

    # Update figure with title, x-axis and y-axis titles
    figure.update_layout(
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title
    )

    # Return figure object
    return figure


def plot_time_series(input_data, y_axis, x_axis, title='', x_axis_title='', y_axis_title='', mode='lines'):
    """
    This function is used to build simple time-series plots in plotly

    :param pd.DataFrame input_data: data set which contains columns to plot
    :param list y_axis: column name of the data frame to plot on y-axis
    :param str x_axis: column name of the data frame to plot on x-axis
    :param str title: title for the plot, defaults to ''
    :param str x_axis_title: x-axis title in the plot, defaults to ''
    :param str y_axis_title: y-axis title in the plot, defaults to ''
    :param str mode: which mode to be used in preparation of the plot, defaults to 'lines'. Other common arguments are
    'lines+markers' and 'markers'
    :return: plotly figure object
    :rtype: plotly.graph_object
    """

    # Prepare a figure object
    figure = make_plotly_figure(title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

    # Add traces
    for column in y_axis:
        if x_axis is 'index':
            figure.add_trace(go.Scatter(x=input_data.index.values, y=input_data[column], name=column))
        else:
            figure.add_trace(go.Scatter(x=input_data[x_axis], y=input_data[column], name=column))

    # Update legend to show legend entries
    figure.update_layout(showlegend=True)

    # Return figure
    return figure


tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
times = pd.date_range(start='2016-07-01', end='2016-07-04', freq='1min', tz=tus.tz)
cs = tus.get_clearsky(times)  # ineichen with climatology table by default
# Format the index as pd.DatetimeIndex
cs.index = pd.DatetimeIndex(cs.index, freq=None)

# Prepare plot
# Get figure object
title = 'Ineichen, climatological turbidity'
x_axis_title = 'DateTime'
y_axis_title = 'Irradiance W/m^2'

test_figure = plot_time_series(input_data=cs, y_axis=['ghi', 'dhi', 'dni'], x_axis='index', title=title,
                               x_axis_title=x_axis_title, y_axis_title=y_axis_title)
test_figure.show()




