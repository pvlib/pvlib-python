"""Functions to retreive and read ERA5 data from the CDS.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import requests
import warnings

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

ERA5_DEFAULT_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'surface_pressure',
    'mean_surface_downward_short_wave_radiation_flux',
    'mean_surface_downward_short_wave_radiation_flux_clear_sky',
    'mean_surface_direct_short_wave_radiation_flux',
    'mean_surface_direct_short_wave_radiation_flux_clear_sky',
    'mean_top_downward_short_wave_radiation_flux']


ERA5_VARIABLE_MAP = {
    '2m_temperature': 'temp_air',
    'surface_pressure': 'pressure',
    'mean_surface_downward_short_wave_radiation_flux': 'ghi',
    'mean_surface_downward_short_wave_radiation_flux_clear_sky': 'ghi_clear',
    'mean_surface_direct_short_wave_radiation_flux': 'bhi',
    'mean_surface_direct_short_wave_radiation_flux_clear_sky': 'bhi_clear',
    'mean_top_downward_short_wave_radiation_flux': 'ghi_extra'}


ERA5_EXTENSIONS = {
    'netcdf': '.nc',
    'grib': '.grib'}

ERA5_HOURS = [
    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
    '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
    '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']


def get_era5(latitude, longitude, start, end,
             variables=ERA5_DEFAULT_VARIABLES,
             dataset_name='reanalysis-era5-single-levels',
             product_type='reanalysis',
             grid=(0.25, 0.25), local_path=None, file_format='netcdf',
             cds_client=None, map_variables=True):
    """
    Retrieve ERA5 reanalysis data from the Copernicus Data Store (CDS).

    An overview of ERA5 is given in [1]_. Data is retrieved using the CDSAPI
    [3]_.

    Temporal coverage: 1979 to present
    Temporal resolution: hourly
    Spatial resolution: 0.25° by 0.25°
    Spatial coverage: global
    Projection: regular latitude-longitude grid

    Time-stamp: from the previous time stamp and backwards, i.e. end of period
    For the reanalysis, the accumulation period is over the 1 hour up to the validity date and time.

    Hint
    ----
    In order to use the this function the package [cdsapi](https://github.com/ecmwf/cdsapi)
    needs to be installed [3]_.

    Also, access requires user registration [3]_ and creating a local file with
    your personal API key, see [4]_ for instructions.

    Important
    ---------
    Retrieving data from the CDS data store is significanlty slower than
    other solar radiation services. Requests are first queued before being
    executed, and the process can easily exceed 20 minutes.

    You can check the status of your api at
    https://cds.climate.copernicus.eu/cdsapp#!/yourrequests

    The status of all queued requests can be seen at
    https://cds.climate.copernicus.eu/live/queue

    Parameters
    ----------
    start : datetime like
        First timestamp of the requested period
    end : datetime like
        Last timestamp of the requested period
    latitude: float or list
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
        If latitude is a list, it should have the format [S, N] and
        latitudes within the range are selected according to the grid.
    longitude : float or list
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
        If longitude is a list, it should have the format [W, E] and
        longitudes within the range are selected according to the grid.
    variables
    dataset_name: str, default: 'reanalysis-era5-single-levels'
        Name of the dataset. Options are 'reanalysis-era5-single-levels',
        'reanalysis-era5-single-levels-monthly-means',
        'reanalysis-era5-land', or 'reanalysis-era5-land-monthly-means'
    product_type: str, {'reanalysis', 'ensemble_members', 'ensemble_mean',
                   'ensemble_spread'}, default: 'reanalysis'
    grid : list or tuple, default: (0.25, 0.25)
    file_format : {grib, netcdf}, default: netcdf
        File format of retrieved data. Note NetCDF is experimental.
    cds_client: CDS API client object, optional
    map_variables : bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable ERA5_VARIABLE_MAP.

    Notes
    -----
    The returned data DataFrame includes the following fields by default:

    ========================  ======  =========================================
    Key, mapped key           Format  Description
    ========================  ======  =========================================
    **Mapped field names are returned when the map_variables argument is True**
    ---------------------------------------------------------------------------
    10m_u_component_of_wind   float   Horizontal air speed towards the east, at a height of 10 m [m/s]
    10m_v_component_of_wind   float   Horizontal air speed towards the north, at a height of 10 m [m/s]
    Clear sky GHI, ghi_clear  float   Clear sky global radiation on horizontal
    2m_temperature            float   Air temperature at 2m above the ground (K)
    surface_pressure          float   Atmospheric pressure at the ground (Pa)
    toa_incident_solar_radiation float Horizontal solar irradiation at top of atmosphere (J/m^2)
    clear_sky_direct_solar_radiation_at_surface float Clear-sky direct horizontal irradiance (J/m^2)
    surface_solar_radiation_downward_clear_sky float Clear-sky global iradiation (J/m^2)
    surface_solar_radiation_downwards float Global horizontal irradiation (GHI) (J/m^2)
    total_sky_direct_solar_radiation_at_surface float Direct/beam horizontal irradiation (BHI) (J/m^2)
    ========================  ======  =========================================

    References
    ----------
    .. [1] `ERA5 hourly data on single levels from 1979 to present
       <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`
    .. [2] `ERA5 hourly data on single levels from 1979 to present
       <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`
    .. [3] `How to use the CDS API
       <https://cds.climate.copernicus.eu/api-how-to>`
    .. [4] `Climate Data Storage user registration
       <https://cds.climate.copernicus.eu/user/register>`
    """  # noqa: E501
    if cds_client is None:
        cds_client = cdsapi.Client()

    # Area is selected by a box made by the four coordinates: [N,W,S,E]
    if type(latitude) == list:
        area = [latitude[1], longitude[0], latitude[0], longitude[1]]
    else:
        area = [latitude+0.001, longitude, latitude, longitude+0.001]

    params = {
        'product_type': product_type,
        'format': file_format,
        'variable': variables,
        'date': start.strftime('%Y-%m-%d') + '/' + end.strftime('%Y-%m-%d'),
        'time': ERA5_HOURS,
        'grid': grid,
        'area': area}

    # Retrieve path to the file
    file_location = cds_client.retrieve(dataset_name, params)

    if file_format not in list(ERA5_EXTENSIONS):
        raise ValueError(f"File format must be in {list(ERA5_EXTENSIONS)}")

    # Load file into memory
    with requests.get(file_location.location) as res:
        if file_format == 'netcdf':  # open netcdf files using xarray
            ds = xr.open_dataset(res.content)

        # Save the file locally if local_path has been specified
        if local_path is not None:
            with open(local_path, 'wb') as f:
                f.write(res.content)

    if (file_format == 'grib') & (local_path is not None):
        warnings.warn('Parsing of grib files is not supported.')
        ds = xr.open_dataset(local_path, engine='cfgrib')
    elif (file_format == 'grib') & (local_path is None):
        raise ValueError('local_path must be specified when file format is '
                         'grib.')

    return ds


def read_era5(filename, map_variables=True):
    """Read an ERA5 netcdf file.

    Hint
    ----
    The ERA5 time stamp convention is to label data periods by the right edge,
    e.g., the time stamp 12:00 for hourly data refers to the period from 11.00
    to 12:00.

    """
    # load into memory
    # with urlopen(filename) as f:
    #     ds = xr.open_dataset(f.read())

    ds = xr.open_dataset(filename)  # chunks=chunks)

    return ds
