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

# Maps problematic timezones to 'Etc/GMT' for parsing.

TZ_MAP = {
    'PST': 'Etc/GMT+8',
    'CST': 'Etc/GMT+6',
}


def map_midc_to_pvlib(variable_map, field_name):
    """A mapper function to rename Dataframe columns to their pvlib counterparts.

    Parameters
    ----------
    variable_map: Dictionary
        A dictionary for mapping MIDC field name to pvlib name. See
        VARIABLE_MAP for default value and description of how to construct
        this argument.
    field_name: string
        The Column to map.

    Returns
    -------
    label: string
        The pvlib variable name associated with the MIDC field or the input if
        a mapping does not exist.

    Notes
    -----
    Will fail if field_name to be mapped matches an entry in VARIABLE_MAP and
    does not contain brackets. This should not be an issue unless MIDC file
    headers are updated.

    """
    new_field_name = field_name
    for midc_name, pvlib_name in variable_map.items():
        if field_name.startswith(midc_name):
            # extract the instrument and units field and then remove units
            instrument_units = field_name[len(midc_name):]
            units_index = instrument_units.find('[')
            instrument = instrument_units[:units_index - 1]
            new_field_name = pvlib_name + instrument.replace(' ', '_')
            break
    return new_field_name


def format_index(data):
    """Create DatetimeIndex for the Dataframe localized to the timezone provided
    as the label of the second (time) column.

    Parameters
    ----------
    data: Dataframe
        Must contain 'DATE (MM/DD/YYYY)' column, second column must be labeled
        with the timezone and contain times in 'HH:MM' format.

    Returns
    -------
    data: Dataframe
        Dataframe with DatetimeIndex localized to the provided timezone.
    """
    tz_raw = data.columns[1]
    timezone = TZ_MAP.get(tz_raw, tz_raw)
    datetime = data['DATE (MM/DD/YYYY)'] + data[tz_raw]
    datetime = pd.to_datetime(datetime, format='%m/%d/%Y%H:%M')
    data = data.set_index(datetime)
    data = data.tz_localize(timezone)
    return data


def format_index_raw(data):
    """Create DatetimeIndex for the Dataframe localized to the timezone provided
    as the label of the third column.

    Parameters
    ----------
    data: Dataframe
        Must contain columns 'Year' and 'DOY'. Timezone must be found as the
        label of the third (time) column.

    Returns
    -------
    data: Dataframe
        The data with a Datetime index localized to the provided timezone.
    """
    tz_raw = data.columns[3]
    timezone = TZ_MAP.get(tz_raw, tz_raw)
    year = data.Year.apply(str)
    jday = data.DOY.apply(lambda x: '{:03d}'.format(x))
    time = data[tz_raw].apply(lambda x: '{:04d}'.format(x))
    index = pd.to_datetime(year + jday + time, format="%Y%j%H%M")
    data = data.set_index(index)
    data = data.tz_localize(timezone)
    return data


def read_midc(filename, variable_map=VARIABLE_MAP, raw_data=False):
    """Read in National Renewable Energy Laboratory Measurement and
    Instrumentation Data Center [1]_ weather data.

    Parameters
    ----------
    filename: string
        Filename or url of data to read.
    variable_map: dictionary
        Dictionary for mapping MIDC field names to pvlib names. See variable
        `VARIABLE_MAP` for default and Notes section below for a description of
        its format.
    raw_data: boolean
        Set to true to use format_index_raw to correctly format the date/time
        columns of MIDC raw data files.

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
    after mapping to differentiate measurements of the same variable. For a
    full list of pvlib variable names see the `Variable Style Rules
    <https://pvlib-python.readthedocs.io/en/latest/variables_style_rules.html>`_.

    Be sure to check the units for the variables you will use on the
    `MIDC site <https://midcdmz.nrel.gov/>`_.

    References
    ----------
    .. [1] NREL: Measurement and Instrumentation Data Center
        `https://midcdmz.nrel.gov/ <https://midcdmz.nrel.gov/>`_
    """
    data = pd.read_csv(filename)
    if raw_data:
        data = format_index_raw(data)
    else:
        data = format_index(data)
    mapper = partial(map_midc_to_pvlib, variable_map)
    data = data.rename(columns=mapper)
    return data


def read_midc_raw_data_from_nrel(site, start, end):
    """Request and read MIDC data directly from the raw data api.

    Parameters
    ----------
    site: string
        The MIDC station id.
    start: datetime
        Start date for requested data.
    end: datetime
        End date for requested data.

    Returns
    -------
    data:
        Dataframe with DatetimeIndex localized to the station location.

    Notes
    -----
    Requests spanning an instrumentation change will yield an error. See the
    MIDC raw data api page
    `here <https://midcdmz.nrel.gov/apps/data_api_doc.pl?_idtextlist>`_
    for more details and considerations.
    """
    args = {'site': site,
            'begin': start.strftime('%Y%m%d'),
            'end': end.strftime('%Y%m%d')}
    endpoint = 'https://midcdmz.nrel.gov/apps/data_api.pl?'
    url = endpoint + '&'.join(['{}={}'.format(k, v) for k, v in args.items()])
    return read_midc(url, raw_data=True)
