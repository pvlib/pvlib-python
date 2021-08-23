"""Functions to read and retrieve MERRA-2 reanalysis data from NASA.
.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""
from pvlib.tools import (_extract_metadata_from_dataset,
                         _convert_C_to_K_in_dataset)

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from pydap.cas.urs import setup_session
except ImportError:
    setup_session = None

try:
    import cftime
except ImportError:
    cftime = None

MERRA2_VARIABLE_MAP = {
    # Variables from M2T1NXRAD - radiation diagnostics
    'LWGEM': 'lwu',  # longwave flux emitted from surface [W/m^2]
    'SWGDN': 'ghi',  # surface incoming shortwave flux [W/m^2]
    'SWGDNCLR': 'ghi_clear',  # SWGDN assuming clear sky [W/m^2]
    'SWTDN': 'toa',  # toa incoming shortwave flux [W/m^2]
    # Variables from M2T1NXSLV - single-level diagnostics
    'PS': 'pressure',  # surface pressure [Pa]
    'T2M': 'temp_air',  # 2-meter air temperature [K converted to C]
    'T2MDEW': 'temp_dew',  # dew point temperature at 2 m [K converted to C]
}

# goldsmr4 contains the single-level 2D hourly MERRA-2 data files
MERRA2_BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/dods'


def get_merra2(latitude, longitude, start, end, dataset, variables, username,
               password, save_path=None, output_format=None,
               map_variables=True):
    """
    Retrieve MERRA-2 reanalysis data from the NASA GES DISC repository.

    The function supports downloading of MERRA-2 [1]_ hourly 2-dimensional
    time-averaged variables. A list of the available datasets and parameters
    is given in [2]_.

    * Temporal coverage: 1980 to present (latency of 2-7 weeks)
    * Temporal resolution: hourly
    * Spatial coverage: global
    * Spatial resolution: 0.5° latitude by 0.625° longitude

    Parameters
    ----------
    latitude: float or list
        in decimal degrees, between -90 and 90, north is positive (ISO 19115).
        If latitude is a list, it should have the format [S, N], and
        latitudes within the range are selected according to the grid.
    longitude: float or list
        in decimal degrees, between -180 and 180, east is positive (ISO 19115).
        If longitude is a list, it should have the format [W, E], and
        longitudes within the range are selected according to the grid.
    start: datetime-like
        First day of the requested period.
    end: datetime-like
        Last day of the requested period.
    variables: list
        List of variables to retrieve, e.g., ['TAUHGH', 'SWGNT'].
    dataset: str
        Name of the dataset to retrieve the variables from, e.g., 'M2T1NXRAD'
        for radiation parameters and 'M2T1NXAER' for aerosol parameters.
    output_format: {'dataframe', 'dataset'}, optional
        Type of data object to return. Default is to return a pandas DataFrame
        if data for a single location is requested and otherwise return an
        xarray Dataset.
    map_variables: bool, default: True
        When true, renames columns to pvlib variable names where applicable.
        See variable MERRA2_VARIABLE_MAP.

    Returns
    -------
    data: DataFrame
        MERRA-2 time-series data, fields depend on the requested data. The
        returned object is either a pandas DataFrame or an xarray dataset,
        depending on the output_format parameter.
    metadata: dict
        Metadata extracted from the netcdf files.

    Notes
    -----
    To obtain MERRA-2 data, it is necessary to register for an EarthData
    account and link it to the GES DISC as described in [3]_.

    MERRA-2 contains 14 single-level 2D datasets with an hourly resolution. The
    most important ones are 'M2T1NXAER', which contains aerosol data,
    'M2T1NXRAD', which contains radiation related parameters, and 'M2T1NXSLV',
    which contains general variables (e.g., temperature and wind speed).

    Warning
    -------
    There is a known error in the calculation of radiation, hence it is
    strongly adviced that radiation from MERRA-2 should not be used. Users
    interested in radiation from reanalysis datasets are referred to
    :func:`pvlib.iotools.get_era5`.

    See Also
    --------
    pvlib.iotools.read_merra2, pvlib.iotools.get_era5

    References
    ----------
    .. [1] `NASA MERRA-2 Project overview
        <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_
    .. [2] `MERRA-2 File specification
        <https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf>`_
    .. [3] `Account registration and data access to NASA's GES DISC
        <https://disc.gsfc.nasa.gov/data-access>`_
    """  # noqa: E501
    if xr is None:
        raise ImportError('Retrieving MERRA-2 data requires xarray')
    if setup_session is None:
        raise ImportError('Retrieving MERRA-2 data requires PyDap')
    if cftime is None:
        raise ImportError('Retrieving MERRA-2 data requires cftime')

    url = MERRA2_BASE_URL + '/' + dataset

    session = setup_session(username, password, check_url=url)
    store = xr.backends.PydapDataStore.open(url, session=session)

    start_float = cftime.date2num(start, units='days since 1-1-1 00:00:0.0')
    end_float = cftime.date2num(end, units='days since 1-1-1 00:00:0.0')

    try:
        sel_dict = {
            'lat': slice(latitude[0], latitude[1]),
            'lon': slice(longitude[0], longitude[1]),
            'time': slice(start_float, end_float)}
    except TypeError:
        sel_dict = {
            'lat': latitude,
            'lon': longitude,
            'time': slice(start_float, end_float)}

    # Setting decode_times=False results in a time saving of up to some minutes
    ds = xr.open_dataset(store, decode_times=False).sel(sel_dict)

    ds = xr.decode_cf(ds)  # Decode timestamps

    variables = [v.lower() for v in variables]  # Make all variables lower-case

    ds = ds[variables]  # select sub-set of variables

    if map_variables:
        # Renaming of xarray datasets throws an error if keys are missing
        ds = ds.rename_vars(
            {k: v for k, v in MERRA2_VARIABLE_MAP.items() if k in list(ds)})

    ds = _convert_C_to_K_in_dataset(ds)
    metadata = _extract_metadata_from_dataset(ds)

    if (output_format == 'dataframe') or (
            (output_format is None) & (ds['lat'].size == 1) &
            (ds['lon'].size == 1)):
        data = ds.to_dataframe()
        # Localize timezone to UTC
        if data.index.nlevels > 1:  # if dataframe has a multi-index
            data.index = data.index.set_levels(data.index.get_level_values('time').tz_localize('utc'), level='time')  # noqa: E501
        else:  # for single location dataframes (only time as index)
            data.index = data.index.tz_localize('UTC')
            data = data.drop(columns = ['lat', 'lon'])
        return data, metadata
    else:
        return ds, metadata


def read_merra2(filename, output_format=None, map_variables=True):
    """Reading a MERRA-2 file into a pandas dataframe.

    MERRA-2 is described in [1]_ and a list of variables can be found in [2]_.

    Parameters
    ----------
    filename: str or path-like or list
        Filename of a netcdf file containing MERRA-2 data or a list of
        filenames.
    output_format: {'dataframe', 'dataset'}, optional
        Type of data object to return. Default is to return a pandas DataFrame
        if data for a single location is requested and otherwise return an
        xarray Dataset.
    map_variables: bool, default: True
        When true, renames columns to pvlib variable names where applicable.
        See variable MERRA2_VARIABLE_MAP.

    Returns
    -------
    data: DataFrame
        MERRA-2 time-series data, fields depend on the requested data. The
        returned object is either a pandas DataFrame or an xarray dataset,
        depending on the output_format parameter.
    metadata: dict
        Metadata extracted from the netcdf files.

    See Also
    --------
    pvlib.iotools.get_merra2, pvlib.iotools.get_era5

    References
    ----------
    .. [1] `NASA MERRA-2 Project overview
        <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_
    .. [2] `MERRA-2 File specification
        <https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf>`_
    """
    if xr is None:
        raise ImportError('Reading MERRA-2 data requires xarray to be installed.')  # noqa: E501

    # open multiple-files (mf) requires dask
    if isinstance(filename, (list, tuple)):
        ds = xr.open_mfdataset(filename)
    else:
        ds = xr.open_dataset(filename)

    if map_variables:
        # Renaming of xarray datasets throws an error if keys are missing
        ds = ds.rename_vars(
            {k: v for k, v in MERRA2_VARIABLE_MAP.items() if k in list(ds)})

    ds = _convert_C_to_K_in_dataset(ds)
    metadata = _extract_metadata_from_dataset(ds)

    if (output_format == 'dataframe') or (
            (output_format is None) & (ds['lat'].size == 1) &
            (ds['lon'].size == 1)):
        data = ds.to_dataframe()
        # Remove lat and lon from multi-index
        if (ds['lat'].size == 1) & (ds['lon'].size == 1):
            data = data.droplevel(['lat', 'lon'])
        # Localize timezone to UTC
        if data.index.nlevels > 1:  # if dataframe has a multi-index
            data.index = data.index.set_levels(data.index.get_level_values('time').tz_localize('utc'), level='time')  # noqa: E501
        else:  # for single location dataframes (only time as index)
            data.index = data.index.tz_localize('UTC')
        return data, metadata
    else:
        return ds, metadata
