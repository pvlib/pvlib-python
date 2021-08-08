"""Functions to read and retrieve MERRA2 reanalysis data from NASA.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import xarray as xr  # Make funky import
from pydap.cas.urs import setup_session
import os

MERRA2_VARIABLE_MAP = {
    # Variables from the 'M2T1NXRAD' dataset
    # Hourly,Time-Averaged,Single-Level,Assimilation,Radiation Diagnostics
    'ALBEDO': 'albedo',
    #'surface_incoming_shortwave_flux': ,
    #'surface_incoming_shortwave_flux_assuming_clear_sky': ,
    #'surface_net_downward_longwave_flux': ,
    'SWGDN': 'ghi',
    'SWTDN': '_extra',
    'PS': 'pressure',
    'T2M': 'temp_air',
    'T2MDEW': 'temp_dew',
    
    }

# goldsmr4 contains the single-level 2D MERRA-2 data files
MERRA2_BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/dods'


def get_merra2(latitude, longitude, start, end, dataset, variables, username,
               password, local_path=None):
    """
    Retrieve MERRA2 reanalysis data from the NASA GES DISC repository.

    Regular latitude-longitude grid of 0.5° x 0.625°.

    Parameters
    ----------
    start: datetime-like
        First day of the requested period
    end: datetime-like
        Last day of the requested period
    local_path: str or path-like, optional
        If specified, path (abs. or relative) of where to save files

    Returns
    -------
    data: DataFrame
        Dataframe containing MERRA2 timeseries data, see [3]_ for variable units.
    metadata: dict
        metadata

    Notes
    -----
    In order to obtain MERRA2 data, it is necessary to registre for an
    Earthdata account and link it to the GES DISC as described in [2]_.

    MERRA-2 contains 14 single-level 2D datasets with an hourly resolution. The
    most important ones are 'M2T1NXAER' which contains aerosol data, 'M2T1NXRAD'
    which contains radiation related parameters, and 'M2T1NXSLV' which contains
    general variables (e.g., temperature and wind speed).

    Warning
    -------
    Known error in calculation of radiation, hence it is strongly adviced that
    radiation from MERRA-2 should not be used. Users interested in radiation
    from reanalysis datasets are referred to pvlib.iotools.get_era5.

    See Also
    --------
    pvlib.iotools.read_merra2, pvlib.iotools.get_era5

    References
    ----------
    .. [1] `NASA MERRA-2 Project overview
        <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_
    .. [2] `Account registration and data access to NASA's GES DISC
        <https://disc.gsfc.nasa.gov/data-access>`
    .. [3] `MERRa-2 File specification
        <https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf>`

    """  # noqa: E501
    url = MERRA2_BASE_URL + '/' + dataset
    session = setup_session(username, password, check_url=url)
    store = xr.backends.PydapDataStore.open(url, session=session)

    ds = xr.open_dataset(store).sel(
        {'lat': latitude,
         'lon': longitude,
         'times': slice(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')),
         })

    data = ds[variables].to_dataframe()

    metadata = ds.attrs  # Gives overall metadata but not variable stuff

    if local_path is not None:
        ds.to_netcdf(os.path.join(local_path, metadata['Filename']))

    return data, metadata


# Shoudl read_merra2 use open_mfdataset?
def read_merra2(filenames, latitude, longitude, variables, map_variables=True):
    """Reading a MERRA-2 file into a pandas dataframe.
    
    """
    ds = xr.open_dataset(filenames).sel(lat=latitude, lon=longitude,
                                        method='nearest')

    data = ds[variables].to_dataframe().drop(columns=['lon', 'lat'])
    metadata = ds.attrs  # Gives overall metadata but not variable stuff

    return data, metadata
