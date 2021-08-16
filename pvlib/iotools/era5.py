"""Functions to retreive and read ERA5 data from the CDS.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""
# The functions only support single-level 2D data and not 3D / pressure-level
# data. Also, monthly datasets and grib files are no supported.

import requests

try:
    import xarray as xr
except ImportError:
    class xr:
        @staticmethod
        def open_dataset(*a, **kw):
            raise ImportError(
                'Reading ERA5 data requires xarray to be installed.')

        @staticmethod
        def open_mfdataset(*a, **kw):
            raise ImportError(
                'Reading ERA5 data requires xarray to be installed.')

try:
    import cdsapi
except ImportError:
    class cdsapi:
        @staticmethod
        def Client(*a, **kw):
            raise ImportError(
                'Retrieving ERA5 data requires cdsapi to be installed.')


def _extract_metadata_from_dataset(ds):
    metadata = {}
    for v in list(ds.variables):
        metadata[v] = {
            'name': ds[v].name,
            'long_name': ds[v].long_name}
        if v.lower() != 'time':
            metadata[v]['units'] = ds[v].units
    metadata['dims'] = dict(ds.dims)
    metadata.update(ds.attrs)  # add arbitrary metadata
    return metadata


# The returned data uses shortNames, whereas the request requires variable
# names according to the CDS convention - passing shortNames results in an
# "Ambiguous" error being raised
ERA5_DEFAULT_VARIABLES = [
    '2m_temperature',  # t2m
    '10m_u_component_of_wind',  # u10
    '10m_v_component_of_wind',  # v10
    'surface_pressure',  # sp
    'mean_surface_downward_short_wave_radiation_flux',  # msdwswrf
    'mean_surface_downward_short_wave_radiation_flux_clear_sky',  # msdwswrfcs
    'mean_surface_direct_short_wave_radiation_flux',  # msdrswrf
    'mean_surface_direct_short_wave_radiation_flux_clear_sky',  # msdrswrfcs
]

ERA5_VARIABLE_MAP = {
    't2m': 'temp_air',
    'd2m': 'temp_dew',
    'sp': 'pressure',
    'msdwswrf': 'ghi',
    'msdwswrfcs': 'ghi_clear',
    'msdwlwrf': 'lwd',
    'msdwlwrfcs': 'lwd_clear',
    'msdrswrf': 'bhi',
    'msdrswrfcs': 'bhi_clear',
    'mtdwswrf': 'ghi_extra'}

ERA5_HOURS = [
    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
    '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
    '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']

CDSAPI_URL = 'https://cds.climate.copernicus.eu/api/v2'


def get_era5(latitude, longitude, start, end, api_key=None,
             variables=ERA5_DEFAULT_VARIABLES,
             dataset='reanalysis-era5-single-levels',
             product_type='reanalysis', grid=(0.25, 0.25), save_path=None,
             cds_client=None, output_format=None, map_variables=True):
    """
    Retrieve ERA5 reanalysis data from the Copernicus Data Store (CDS).

    * Temporal coverage: 1979 to present (latency of ~5 days)
    * Temporal resolution: hourly
    * Spatial coverage: global
    * Spatial resolution: 0.25° by 0.25°

    An overview of ERA5 is given in [1]_ and [2]_. Data is retrieved using the
    CDSAPI [3]_.

    .. admonition:: Time reference

        ERA5 time stamps are in UTC and corresponds to the end of the period
        (right labeled). E.g., the time stamp 12:00 for hourly data refers to
        the period from 11:00 to 12:00.

    .. admonition:: Usage notes

        To use this function the package CDSAPI [4]_ needs to be installed
        [3]_. The CDSAPI keywords are described in [5]_.

        Requested variables should be specified according to the naming
        convention used by the CDS. The returned data contains the short-name
        versions of the variables. See [2]_ for a list of variables names and
        units.

        Access to the CDS requires user registration, see [6]_. The obtaining
        API key can either be passed directly to the function or be saved in a
        local file as described in [3]_.

        It is possible to check your
        `request status <https://cds.climate.copernicus.eu/cdsapp#!/yourrequests>`_
        and the `status of all queued requests <https://cds.climate.copernicus.eu/live/queue>`_.

    Parameters
    ----------
    latitude: float or list
        in decimal degrees, between -90 and 90, north is positive (ISO 19115).
        If latitude is a list, it should have the format [S, N] and
        latitudes within the range are selected according to the grid.
    longitude: float or list
        in decimal degrees, between -180 and 180, east is positive (ISO 19115).
        If longitude is a list, it should have the format [W, E] and
        longitudes within the range are selected according to the grid.
    start: datetime like
        First day of the requested period
    end: datetime like
        Last day of the requested period
    api_key: str, optional
        Personal API key for the CDS with the format "uid:key" e.g.
        '00000:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    variables: list, default: ERA5_DEFAULT_VARIABLES
        List of variables to retrieve (according to CDS naming convention)
    dataset: str, default 'reanalysis-era5-single-levels'
        Name of the dataset to retrieve the variables from. Can be either
        'reanalysis-era5-single-levels' or 'reanalysis-era5-land'.
    product_type: str, {'reanalysis', 'ensemble_members', 'ensemble_mean', 'ensemble_spread'}, default: 'reanalysis'
        ERA5 product type
    grid: list or tuple, default: (0.25, 0.25)
        User specified grid resolution
    save_path: str or path-like, optional
        Filename of where to save data. Should have ".nc" extension.
    cds_client: CDS API client object, optional
        CDS API client
    output_format: {'dataframe', 'dataset'}, optional
        Type of data object to return. Default is to return a pandas DataFrame
        if file only contains one location and otherwise return an xarray
        Dataset.
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable ERA5_VARIABLE_MAP.

    Notes
    -----
    The returned data includes the following fields by default:

    ========================  ======  =========================================
    Key, mapped key           Format  Description
    ========================  ======  =========================================
    *Mapped field names are returned when the map_variables argument is True*
    ---------------------------------------------------------------------------
    2tm, temp_air             float   Air temperature at 2 m above ground [K]
    u10                       float   Horizontal airspeed towards east at 10 m [m/s]
    v10                       float   Horizontal airspeed towards north at 10 m [m/s]
    sp, pressure              float   Atmospheric pressure at the ground [Pa]
    msdwswrf, ghi             float   Mean surface downward short-wave radiation flux [W/m^2]
    msdwswrfcs, ghi_clear     float   Mean surface downward short-wave radiation flux, clear sky [W/m^2]
    msdrswrf, bhi             float   Mean surface direct short-wave radiation flux [W/m^2]
    msdrswrfcs, bhi_clear     float   Mean surface direct short-wave radiation flux, clear sky [W/m^2]
    ========================  ======  =========================================

    Returns
    -------
    data: DataFrame
        ERA5 time-series data, fields depend on the requested data. The
        returned object is either a pandas DataFrame or an xarray dataset, see
        the output_format parameter.
    metadata: dict
        Metadata for the time-series.

    See Also
    --------
    pvlib.iotools.read_era5

    References
    ----------
    .. [1] `ERA5 hourly data on single levels from 1979 to present
       <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_
    .. [2] `ERA5 data documentation
       <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`_
    .. [3] `How to use the CDS API
       <https://cds.climate.copernicus.eu/api-how-to>`_
    .. [4] `CDSAPI source code
       <https://github.com/ecmwf/cdsapi>`_
    .. [5] `Climate Data Store (CDS) API Keywords
       <https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+API+Keywords>`_
    .. [6] `Climate Data Storage user registration
       <https://cds.climate.copernicus.eu/user/register>`_
    """  # noqa: E501
    if cds_client is None:
        cds_client = cdsapi.Client(url=CDSAPI_URL, key=api_key)

    # Area is selected by a box made by the four coordinates: [N, W, S, E]
    try:
        area = [latitude[1], longitude[0], latitude[0], longitude[1]]
    except TypeError:
        area = [latitude+0.005, longitude-0.005,
                latitude-0.005, longitude+0.005]

    params = {
        'product_type': product_type,
        'format': 'netcdf',
        'variable': variables,
        'date': start.strftime('%Y-%m-%d') + '/' + end.strftime('%Y-%m-%d'),
        'time': ERA5_HOURS,
        'grid': grid,
        'area': area}

    # Retrieve path to the file
    file_location = cds_client.retrieve(dataset, params)

    # Load file into memory
    with requests.get(file_location.location) as res:

        # Save the file locally if save_path  has been specified
        if save_path is not None:
            with open(save_path, 'wb') as f:
                f.write(res.content)

        return read_era5(res.content, map_variables=map_variables,
                         output_format=output_format)


def read_era5(filename, output_format=None, map_variables=True):
    """Read one or more ERA5 netcdf files.

    Parameters
    ----------
    filename: str or path-like or list
        Filename of a netcdf file containing ERA5 data or a list of filenames.
    output_format: {'dataframe', 'dataset'}, optional
        Type of data object to return. Default is to return a pandas DataFrame
        if file only contains one location and otherwise return an xarray
        dataset.
    map_variables: bool, default: True
        When true, renames columns to pvlib variable names where applicable.
        See variable ERA5_VARIABLE_MAP.

    Returns
    -------
    data: DataFrame
        ERA5 time-series data, fields depend on the requested data. The
        returned object is either a pandas DataFrame or an xarray Dataset, see
        the output_format parameter.
    metadata: dict
        Metadata for the time-series.

    See Also
    --------
    pvlib.iotools.get_era5

    References
    ----------
    .. [1] `ERA5 hourly data on single levels from 1979 to present
       <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_
    .. [2] `ERA5 data documentation
       <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`_
    """
    # open multiple-files (mf) requires dask
    if isinstance(filename, (list, tuple)):
        ds = xr.open_mfdataset(filename)
    else:
        ds = xr.open_dataset(filename)

    metadata = _extract_metadata_from_dataset(ds)

    if map_variables:
        # Renaming of xarray datasets throws an error if keys are missing
        ds = ds.rename_vars(
            {k: v for k, v in ERA5_VARIABLE_MAP.items() if k in list(ds)})

    if (output_format == 'dataframe') or (
            (output_format is None) & (ds['latitude'].size == 1) &
            (ds['longitude'].size == 1)):
        data = ds.to_dataframe()
        if (ds['latitude'].size == 1) & (ds['longitude'].size == 1):
            data.droplevel(['latitude', 'longitude'])
        return data, metadata
    else:
        return ds, metadata
