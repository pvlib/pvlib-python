import pandas as pd
import requests
from io import StringIO


VARIABLE_MAP = {
    'SWGDN': 'ghi',
    'SWGDNCLR': 'ghi_clear',
    'ALBEDO': 'albedo',
    'T2M': 'temp_air',
    'T2MDEW': 'temp_dew',
    'PS': 'pressure',
    'TOTEXTTAU': 'aod550',
}

def get_merra2(latitude, longitude, start, end, username, password, dataset,
               variables, map_variables=True):
    """
    Retrieve MERRA-2 time-series irradiance and meteorological data from
    NASA's GESDISC data archive.

    MERRA-2 [1]_ offers modeled data for many atmospheric quantities at hourly
    resolution on a 0.5° x 0.625° global grid.

    Access must be granted to the GESDISC data archive before EarthData
    credentials will work.  See [2]_ for instructions.

    Parameters
    ----------
    latitude : float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start : datetime like or str
        First timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    end : datetime like or str
        Last timestamp of the requested period. If a timezone is not
        specified, UTC is assumed.
    username : str
        NASA EarthData username.
    password : str
        NASA EarthData password.
    dataset : str
        Dataset name (with version), e.g. "M2T1NXRAD.5.12.4".
    variables : list of str
        List of variable names to retrieve.  See the documentation of the
        specific dataset you are accessing for options.
    map_variables : bool, default True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.

    Raises
    ------
    ValueError
        If ``start`` and ``end`` are in different years, when converted to UTC.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the middle of the interval.
    meta : dict
        Metadata.

    Notes
    -----
    The following datasets provide quantities useful for PV modeling:

    - M2T1NXRAD.5.12.4: SWGDN, SWGDNCLR, ALBEDO
    - M2T1NXSLV.5.12.4: T2M, U10M, V10M, T2MDEW, PS
    - M2T1NXAER.5.12.4: TOTEXTTAU

    Note that MERRA2 does not currently provide DNI or DHI.

    References
    ----------
    .. [1] https://gmao.gsfc.nasa.gov/gmao-products/merra-2/
    .. [2] https://disc.gsfc.nasa.gov/earthdata-login
    """

    # general API info here:
    # https://docs.unidata.ucar.edu/tds/5.0/userguide/netcdf_subset_service_ref.html  # noqa: E501

    def _to_utc_dt_notz(dt):
        dt = pd.to_datetime(dt)
        if dt.tzinfo is None:  # convert everything to UTC
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.tz_localize(None)  # drop tz so that isoformat() is clean

    start = _to_utc_dt_notz(start)
    end = _to_utc_dt_notz(end)

    if (year := start.year) != end.year:
        raise ValueError("start and end must be in the same year (in UTC)")

    url = (
        "https://goldsmr4.gesdisc.eosdis.nasa.gov/thredds/ncss/grid/"
        f"MERRA2_aggregation/{dataset}/{dataset}_Aggregation_{year}.ncml"
    )

    parameters = {
        'var': ",".join(variables),
        'latitude': latitude,
        'longitude': longitude,
        'time_start': start.isoformat() + "Z",
        'time_end': end.isoformat() + "Z",
        'accept': 'csv',
    }

    auth = (username, password)

    with requests.Session() as session:
        session.auth = auth
        login = session.request('get', url, params=parameters)
        response = session.get(login.url, auth=auth, params=parameters)

    response.raise_for_status()

    content = response.content.decode('utf-8')
    buffer = StringIO(content)
    df = pd.read_csv(buffer)

    df.index = pd.to_datetime(df['time'])

    meta = {}
    meta['dataset'] = dataset
    meta['station'] = df['station'].values[0]
    meta['latitude'] = df['latitude[unit="degrees_north"]'].values[0]
    meta['longitude'] = df['longitude[unit="degrees_east"]'].values[0]

    # drop the non-data columns
    dropcols = ['time', 'station', 'latitude[unit="degrees_north"]',
        'longitude[unit="degrees_east"]']
    df = df.drop(columns=dropcols)

    # column names are like T2M[unit="K"] by default.  extract the unit
    # for the metadata, then rename col to just T2M
    units = {}
    rename = {}
    for col in df.columns:
        name, _ = col.split("[", maxsplit=1)
        unit = col.split('"')[1]
        units[name] = unit
        rename[col] = name

    meta['units'] = units
    df = df.rename(columns=rename)

    if map_variables:
        df = df.rename(columns=VARIABLE_MAP)

    return df, meta
