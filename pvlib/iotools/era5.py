"""Functions to retreive and read ERA5 data from the CDS.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import requests

# Can probably remove the netCDF4 dependency
try:
    import netCDF4
except ImportError:
    class netCDF4:
        @staticmethod
        def Dataset(*a, **kw):
            raise ImportError(
                'Reading ERA5 data requires netCDF4 to be installed.')

try:
    import xarray as xr
except ImportError:
    class xarray:
        @staticmethod
        def open_dataset(*a, **kw):
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

CDSAPI_URL = 'https://cds.climate.copernicus.eu/api/v2'


ERA5_DEFAULT_VARIABLES = [
    '2m_temperature',  # t2m
    '10m_u_component_of_wind',  # u10
    '10m_v_component_of_wind',  # v10
    'surface_pressure',  # sp
    'mean_surface_downward_short_wave_radiation_flux',  # msdwswrf
    'mean_surface_downward_short_wave_radiation_flux_clear_sky',  # msdwswrfcs
    'mean_surface_direct_short_wave_radiation_flux', # msdrswrf
    'mean_surface_direct_short_wave_radiation_flux_clear_sky',  # msdrswrfcs
    ]

# The returned data uses short-names, whereas the request requires CDS variable
# names - passing short-names results in an "Ambiguous" error being raised
ERA5_VARIABLE_MAP = {
    't2m': 'temp_air',
    'd2m': 'temp_dew',
    'sp': 'pressure',
    'msdwswrf': 'ghi',
    'msdwswrfcs': 'ghi_clear',
    'msdwlwrf': 'lwd',
    'msdwlwrfcs': 'lwd_clear',
    'msdrswrf': 'bhi',  # check this
    'msdrswrfcs': 'bhi_clear',  # check this
    'mtdwswrf': 'ghi_extra'}

ERA5_EXTENSIONS = {
    'netcdf': '.nc',
    'grib': '.grib'}

ERA5_HOURS = [
    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
    '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
    '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']


def get_era5(latitude, longitude, start, end, api_key=None,
             variables=ERA5_DEFAULT_VARIABLES,
             dataset_name='reanalysis-era5-single-levels',
             product_type='reanalysis', grid=(0.25, 0.25), local_path=None,
             cds_client=None, map_variables=True):
    """
    Retrieve ERA5 reanalysis data from the Copernicus Data Store (CDS).

    An overview of ERA5 is given in [1]_. Data is retrieved using the CDSAPI
    [3]_.

    Temporal coverage: 1979 to present
    Temporal resolution: hourly
    Spatial resolution: 0.25° by 0.25°
    Spatial coverage: global

    Time-stamp: from the previous time stamp and backwards, i.e. end of period
    For the reanalysis, the accumulation period is over the 1 hour up to the validity date and time.

    Note the variables specified should be the variable names used by in the
    CDS. The returned data contains the short-name versions of the variables.
    See [4]_ for a list of variables names and units. 

    Hint
    ----
    In order to use the this function the package
    [cdsapi](https://github.com/ecmwf/cdsapi) needs to be installed [3]_.

    Access requires user registration [3]_. The obtaining API key can either be
    passed directly to the function or be saved in a local file as described in
    [4]_.

    You can check the status of your requests
    [here](https://cds.climate.copernicus.eu/cdsapp#!/yourrequests) and the
    status of all queued requests
    [here](https://cds.climate.copernicus.eu/live/queue),

    Parameters
    ----------
    latitude: float or list
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
        If latitude is a list, it should have the format [S, N] and
        latitudes within the range are selected according to the grid.
    longitude: float or list
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
        If longitude is a list, it should have the format [W, E] and
        longitudes within the range are selected according to the grid.
    start: datetime like
        First day of the requested period
    end: datetime like
        Last day of the requested period
    api_key: str, optional
        Personal API key for the CDS
    variables: list, default: ERA5_DEFAULT_VARIABLES
        List of variables to retrieve (requires CDS variable names)
    dataset: str, default: {'reanalysis-era5-single-levels', 'reanalysis-era5-land'}
        Name of the dataset
    product_type: str, {'reanalysis', 'ensemble_members', 'ensemble_mean',
                   'ensemble_spread'}, default: 'reanalysis'
        ERA5 product type
    grid: list or tuple, default: (0.25, 0.25)
        User specified grid resolution
    cds_client: CDS API client object, optional
        CDS API client
    map_variables : bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable ERA5_VARIABLE_MAP.

    Notes
    -----
    The returned data includes the following fields by default:

    ========================  ======  =========================================
    Key, mapped key           Format  Description
    ========================  ======  =========================================
    **Mapped field names are returned when the map_variables argument is True**
    ---------------------------------------------------------------------------
    2tm, temp_air             float   Air temperature at 2 m above ground (K)
    u10                       float   Horizontal air speed towards east at 10 m [m/s]
    v10                       float   Horizontal air speed towards north at 10 m [m/s]
    sp, pressure              float   Atmospheric pressure at the ground (Pa)
    msdwswrf, ghi             float   Mean surface downward short-wave radiation flux [W/m^2]
    msdwswrfcs, ghi_clear     float   Mean surface downward short-wave radiation flux, clear sky [W/m^2]
    msdrswrf, bhi             float   Mean surface direct short-wave radiation flux [W/m^2]
    msdrswrfcs, bhi_clear     float   Mean surface direct short-wave radiation flux, clear sky [W/m^2]
    ========================  ======  =========================================

    References
    ----------
    .. [1] `ERA5 hourly data on single levels from 1979 to present
       <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`
    .. [2] `ERA5: data documentation
       <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`
    .. [3] `How to use the CDS API
       <https://cds.climate.copernicus.eu/api-how-to>`
    .. [4] `Climate Data Storage user registration
       <https://cds.climate.copernicus.eu/user/register>`
    """  # noqa: E501
    if cds_client is None:
        cds_client = cdsapi.Client(url=CDSAPI_URL, key=api_key)

    # Area is selected by a box made by the four coordinates: [N, W, S, E]
    if type(latitude) == list:
        area = [latitude[1], longitude[0], latitude[0], longitude[1]]
    else:
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
    file_location = cds_client.retrieve(dataset_name, params)

    # Load file into memory
    with requests.get(file_location.location) as res:

        # Save the file locally if local_path has been specified
        if local_path is not None:
            with open(local_path, 'wb') as f:
                f.write(res.content)

        return read_era5(res.content, map_variables=map_variables)


def read_era5(filename, map_variables=True):
    """Read an ERA5 netcdf file.

    Hint
    ----
    The ERA5 time stamp convention is to label data periods by the right edge,
    e.g., the time stamp 12:00 for hourly data refers to the period from 11.00
    to 12:00.


    If the file contains only one location, then a DataFrame is returned,
    otherwise a xarray dataset is returned.

    """
    ds = xr.open_dataset(filename)

    metadata = ds.attrs

    if map_variables:
        # Renaming of xarray datasets throws an error if keys are missing
        ds = ds.rename_vars(
            {k: v for k, v in ERA5_VARIABLE_MAP.items() if k in list(ds)})

    if (ds['latitude'].size == 1) & (ds['longitude'].size == 1):
        data = ds.to_dataframe().droplevel(['latitude','longitude'])
        return data, metadata
    else:
        return ds, metadata
