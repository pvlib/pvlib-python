"""Functions to access data from Copernicus Climate Data Storage (CDS).
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import cdsapi
import pandas as pd

# %%
# ERA5 vs ERA-Land? - ERA-Land doesn not seem to have BHI, but only GHI
# reanalysis-era5-single-levels vs. reanalsys-era5-land?
# Other parameters: 'total_column_water_vapour', 'surface_net_solar_radiation'
# Variables are also avaiable as integrated values
start_time = pd.Timestamp('2018-01-01')
end_time = pd.Timestamp('2018-05-01')
product_type = 'reanalysis'

get_era5(start_time, end_time, latitude=55, longitude=10)

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


def get_era5(start_time, end_time, latitude, longitude,
             variables=DEFAULT_VARIABLES, map_variables=True,
             product_type='reanalysis', file_format='netcdf'):
    """
    Retrieve time-series of XXfrom ERA5 reanalysis [1]_.

    Access requires user registration [2]_ and creating a local file with your
    personal API key, see [3]_ for instructions.

    Temporal coverage: 1979 to present
    Temporal resolution: hourly
    Spatial resolution: 0.25° by 0.25°
    Spatial coverage: global
    Projection: regular latitude-longitude grid

    Time-stamp: from the previous time stamp and backwards, i.e. end of period
    For the reanalysis, the accumulation period is over the 1 hour up to the validity date and time.

    Parameters
    ----------
    start_time: datetime like
        First timestamp of the requested period
    end_time: datetime like
        Last timestamp of the requested period
    latitude: float
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    altitude: float, default: None
        Altitude in meters.
    file_format: {grib, netcdf}, default netcdf
        File format of retrieved data. Note NetCDF is experimental.
    product_type: str, {'reanalysis', 'ensemble_members', 'ensemble_mean',
                   'ensemble_spread'}, default: 'reanalysis'
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
    .. [2] `How to use CDS API
       <https://cds.climate.copernicus.eu/api-how-to>`
    .. [3] `Climate Data Storage user registration
       <https://cds.climate.copernicus.eu/user/register>`
    """
    c = cdsapi.Client()

    dataset_name = 'reanalysis-era5-single-levels'

    # This should be made faster!
    # Generate lists of the years, months, days, and hours of interest
    date_range = pd.date_range(start_time, end_time, freq='1h')
    years = date_range.strftime('%Y').unique()
    months = date_range.strftime('%m').unique()
    days = date_range.strftime('%d').unique()
    hours = date_range.strftime('%H:%M').unique()

    c.retrieve(dataset_name,
               {'product_type': product_type,
                'format': file_format,
                'variable': variables,
                'year': years,
                'month': months,
                'day': days,
                'time': hours,
                'area': [79, 11, 78.5, 13],
                },
               'ERA5_NYAA_solar_2015.grib')









