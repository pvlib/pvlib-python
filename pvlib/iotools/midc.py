"""Functions to read NREL MIDC data.
"""
from functools import partial
import pandas as pd

# VARIABLE_MAP is a dictionary mapping partial MIDC field names to their
# pvlib names. See docstring of read_midc for description.

VARIABLE_MAP = {
    'Direct': 'dni',
    'Global': 'ghi',
    'Diffuse': 'dhi',
    'Airmass': 'airmass',
    'Azimuth Angle': 'solar_azimuth',
    'Zenith Angle': 'solar_zenith',
    'Air Temperature': 'temp_air',
    'Temperature': 'temp_air',
    'Dew Point Temp': 'temp_dew',
    'Relative Humidity': 'relative_humidity',
}


def map_midc_to_pvlib(variable_map, field_name):
    """A mapper function to rename Dataframe columns to their pvlib counterparts.

    Parameters
    ----------
    variable_map: Dictionary
        A dictionary for mapping MIDC field nameto pvlib name. See VARIABLE_MAP
        for default value and description of how to construct this argument.
    field_name: string
        The Column to map.

    Returns
    -------
    label: string
        The pvlib variable name associated with the MIDC field or the input if
        a mapping does not exist.
    """
    new_field_name = field_name
    for midc_name, pvlib_name in variable_map.items():
        if field_name.startswith(midc_name):
            # extract the instument and units field and then remove units
            instrument_units = field_name[len(midc_name):]
            instrument = instrument_units[:instrument_units.find('[') - 1]
            new_field_name = pvlib_name + instrument.replace(' ', '_')
            break
    return new_field_name


def format_index(data):
    """Create DatetimeIndex for the Dataframe localized to the timezone provided
    as the label of the second (time) column.
    """
    timezone = data.columns[1]
    datetime = data['DATE (MM/DD/YYYY)'] + data[timezone]
    datetime = pd.to_datetime(datetime, format='%m/%d/%Y%H:%M')
    data = data.set_index(datetime)
    data = data.tz_localize(timezone)
    return data


def read_midc(filename, variable_map=VARIABLE_MAP):
    """Read in NREL MIDC [1]_ weather data.

    Parameters
    ----------
    filename: string
        Filename or url of data to read.
    variable_map: dictionary
        Dictionary for mapping MIDC field names to pvlib names. See variable
        `VARIABLE_MAP` for default and Notes section below for a description of
        its format. 

    Returns
    -------
    data: Dataframe
        A dataframe with DatetimeIndex localized to the provided timezone.

    Notes
    -----
    Keys of the `variable_map` dictionary should include the first part
    of a MIDC field name which indicates the variable being measured.
    
        e.g. 'Global PSP [W/m^2]' is entered as a key of 'Global'
    
    The 'PSP' indicating instrument is appended to the pvlib variable name
    after mapping to differentiate measurements of the same variable. For a full
    list of pvlib variable names see the `Variable Style Rules <https://pvlib-python.readthedocs.io/en/latest/variables_style_rules.html>`_.

    Be sure to check the units for the variables you will use on the
    `MIDC site <https://midcdmz.nrel.gov/>`_.

    References
    ----------
    .. [1] National Renewable Energy Laboratory: Measurement and Instrumentation Data Center
        `https://midcdmz.nrel.gov/ <https://midcdmz.nrel.gov/>`_
    """
    data = pd.read_csv(filename)
    data = format_index(data)
    mapper = partial(map_midc_to_pvlib, variable_map)
    data = data.rename(columns=mapper)
    return data
