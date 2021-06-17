"""Functions to retreive and read ERA5 data from the CDS.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import cdsapi
import pandas as pd

# %%
# ERA5 vs ERA-Land? - ERA-Land doesn not seem to have BHI, but only GHI
# reanalysis-era5-single-levels vs. reanalsys-era5-land?
# Other parameters: 'total_column_water_vapour', 'surface_net_solar_radiation'
# Variables are also avaiable as integrated values
start = pd.Timestamp('2018-12-31')
end = pd.Timestamp('2018-12-31 23:59')
product_type = 'reanalysis'
latitude = 55
longitude = 10
variables = ['mean_surface_downward_short_wave_radiation_flux','mean_surface_downward_short_wave_radiation_flux_clear_sky']

# %%
get_era5(start, end, latitude=latitude, longitude=longitude,
         variables=variables)

# %%
# If the getter is only a 'downloader' and does not parse, it should not have map_variables

# %%


DEFAULT_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'surface_pressure',
    'mean_surface_downward_short_wave_radiation_flux',
    'mean_surface_downward_short_wave_radiation_flux_clear_sky',
    'mean_surface_direct_short_wave_radiation_flux',
    'mean_surface_direct_short_wave_radiation_flux_clear_sky',
    'mean_top_downward_short_wave_radiation_flux'
    ]


ERA5_VARIABLE_MAP = {
    '2m_temperature': 'temp_air',
    'surface_pressure': 'pressure',
    'mean_surface_downward_short_wave_radiation_flux': 'ghi',
    'mean_surface_downward_short_wave_radiation_flux_clear_sky': 'ghi_clear',
    'mean_surface_direct_short_wave_radiation_flux': 'bhi',
    'mean_surface_direct_short_wave_radiation_flux_clear_sky': 'bhi_clear',
    'mean_top_downward_short_wave_radiation_flux': 'ghi_extra',
    }


def get_era5(start, end, latitude, longitude,
             dataset_name='reanalysis-era5-single-levels',
             product_type='reanalysis', variables=DEFAULT_VARIABLES,
             grid=[0.25, 0.25], file_format='netcdf', cds_client=None,
             map_variables=True):
    """
    Retrieve ERA5 reanalysis data from the Copernicus Data Store (CDS).
    
    [1]_.

    Access requires user registration [3]_ and creating a local file with your
    personal API key, see [4]_ for instructions.

    Temporal coverage: 1979 to present
    Temporal resolution: hourly
    Spatial resolution: 0.25° by 0.25°
    Spatial coverage: global
    Projection: regular latitude-longitude grid

    Time-stamp: from the previous time stamp and backwards, i.e. end of period
    For the reanalysis, the accumulation period is over the 1 hour up to the validity date and time.

    Parameters
    ----------
    start: datetime like
        First timestamp of the requested period
    end: datetime like
        Last timestamp of the requested period
    latitude: float or list
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float or list
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    dataset_name: str, default: 
        Name of the dataset. Options are 'reanalysis-era5-single-levels',
        'reanalysis-era5-single-levels-monthly-means',
        'reanalysis-era5-land', or 'reanalysis-era5-land-monthly-means'
    product_type: str, {'reanalysis', 'ensemble_members', 'ensemble_mean',
                   'ensemble_spread'}, default: 'reanalysis'
    variables
    grid
    file_format: {grib, netcdf}, default netcdf
        File format of retrieved data. Note NetCDF is experimental.
    cds_client: CDS API client object
    map_variables: bool, default: True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable ERA5_VARIABLE_MAP.



    The returned data DataFrame includes the following fields:

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
    """
    if cds_client is None:
        cds_client = cdsapi.Client()

    # Area is selected by a box made by the four coordinates: [N,W,S,E]
    if type(latitude) == list:
        area = latitude + longitude
    else:
        area = [latitude, longitude, latitude+0.001, longitude+0.001]

    cds_client.retrieve(dataset_name,
                        {'product_type': product_type,
                         'format': file_format,
                         'variable': variables,
                         'date': start.strftime('%Y-%m-%d %H:%M') + '/' \
                             + end.strftime('%Y-%m-%d %H:%M'),
                             'grid': grid,
                             'area': area,
                             },
                            f'ERA5_test_data.{file_format}')

# Use simpler variable names, e.g., 2t


# %%

start = pd.Timestamp(2020,1,1)
end = pd.Timestamp(2020,1,2)
latitude = 55
longitude = 10

get_era5(start, end, latitude, longitude,
             dataset_name='reanalysis-era5-single-levels',
             product_type='reanalysis', variables=DEFAULT_VARIABLES,
             grid=[0.25, 0.25], file_format='netcdf', cds_client=None,
             map_variables=True)

# %% Example official

c = cdsapi.Client()
c.retrieve("reanalysis-era5-single-levels",
           {"variable": variables,
            "product_type": "reanalysis",
            "year": ['2018'],
            "month": months,
            "day": days,
            "time": hours,
            "format": "grib"
            },
           "download.grib")

# %%

