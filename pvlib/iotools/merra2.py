"""Functions to read and retrieve MERRA2 reanalysis data from NASA.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import xarray as xr  # Make funky import
from pydap.cas.urs import setup_session

MERRA2_BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/dods'


def get_merra2(latitude, longitude, start, end, dataset, variables, username,
               password, local_path=None):
    """
    Retrieve MERRA2 reanalysis data from the NASA GESDISC repository.

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

    data = ds[[variables]].to_dataframe()

    metadata = {}

    return data, metadata


