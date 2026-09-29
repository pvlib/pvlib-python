import pandas as pd
import numpy as np
import requests
from io import StringIO


VARIABLE_MAP = {
    'SWGDN': 'ghi',
    'SWGDNCLR': 'ghi_clear',
    'ALBEDO': 'albedo',
    'LWGNT': 'longwave_net',
    'LWGAB': 'longwave_down',
    'T2M': 'temp_air',
    'PS': 'pressure',
    'TOTEXTTAU': 'aod550',
    'TQV': 'precipitable_water',
}


def _k_to_c(temp_k):
    return temp_k - 273.15


UNITS = {
    'T2M': _k_to_c,
}


def get_merra2(latitude, longitude, start, end, username, password, dataset,
               variables, map_variables=True):
    """
    Retrieve MERRA-2 time-series irradiance and meteorological reanalysis data
    from NASA's GESDISC data archive.

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
        specified, UTC is assumed. Must be in the same year as ``start``.
    username : str
        NASA EarthData username.
    password : str
        NASA EarthData password.
    dataset : str or list of str
        Dataset name (with version), e.g. "M2T1NXRAD.5.12.4". If all
        variables are in the same dataset, this can be a single string.
        Otherwise, pass a list of dataset names corresponding to
        the list of requested variables.
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

    +------------------------------------+-----------+--------------------+
    | Dataset                            | Variable  | pvlib name         |
    +====================================+===========+====================+
    | `M2T1NXRAD.5.12.4 <M2T1NXRAD_>`_   | SWGDN     | ghi                |
    |                                    +-----------+--------------------+
    |                                    | SWGDNCLR  | ghi_clear          |
    |                                    +-----------+--------------------+
    |                                    | ALBEDO    | albedo             |
    |                                    +-----------+--------------------+
    |                                    | LWGNT     | longwave_net       |
    +------------------------------------+-----------+--------------------+
    | `M2T1NXLFO.5.12.4 <M2T1NXLFO_>`_   | LWGAB     | longwave_down      |
    +------------------------------------+-----------+--------------------+
    | `M2T1NXSLV.5.12.4 <M2T1NXSLV_>`_   | T2M       | temp_air           |
    |                                    +-----------+--------------------+
    |                                    | U10M      | n/a                |
    |                                    +-----------+--------------------+
    |                                    | V10M      | n/a                |
    |                                    +-----------+--------------------+
    |                                    | PS        | pressure           |
    |                                    +-----------+--------------------+
    |                                    | TO3       | n/a                |
    |                                    +-----------+--------------------+
    |                                    | TQV       | precipitable_water |
    +------------------------------------+-----------+--------------------+
    | `M2T1NXAER.5.12.4 <M2T1NXAER_>`_   | TOTEXTTAU | aod550             |
    |                                    +-----------+--------------------+
    |                                    | TOTSCATAU | n/a                |
    |                                    +-----------+--------------------+
    |                                    | TOTANGSTR | n/a                |
    +------------------------------------+-----------+--------------------+

    .. _M2T1NXRAD: https://disc.gsfc.nasa.gov/datasets/M2T1NXRAD_5.12.4/summary
    .. _M2T1NXSLV: https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary
    .. _M2T1NXAER: https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary
    .. _M2T1NXLFO: https://disc.gsfc.nasa.gov/datasets/M2T1NXLFO_5.12.4/summary

    A complete list of datasets and their documentation is available at [3]_.

    Note that MERRA2 does not currently provide DNI or DHI.

    References
    ----------
    .. [1] https://gmao.gsfc.nasa.gov/gmao-products/merra-2/
    .. [2] https://disc.gsfc.nasa.gov/earthdata-login
    .. [3] https://disc.gsfc.nasa.gov/datasets?project=MERRA-2
    """

    def _to_utc_dt_notz(dt):
        dt = pd.to_datetime(dt)
        if dt.tzinfo is not None:
            # convert to utc, then drop tz so that isoformat() is clean
            dt = dt.tz_convert("UTC").tz_localize(None)
        return dt

    start = _to_utc_dt_notz(start)
    end = _to_utc_dt_notz(end)

    # login
    login_url = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
    response = requests.post(
        login_url,
        auth=(username, password),
        headers={"Accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    token = response.json()["access_token"]

    # data query
    if isinstance(dataset, str):
        datasets = [dataset] * len(variables)
    else:
        datasets = dataset

    data_url = "https://api.giovanni.earthdata.nasa.gov/timeseries"
    parameters = {
        "location": "[{},{}]".format(round(latitude, 4), round(longitude, 4)),
        "time": "{}/{}".format(start.isoformat(), end.isoformat())
    }
    query_headers = {
        'Authorization': f'Bearer {token}'
    }
    meta = {'dataset': dataset}
    data = {}
    for variable, dataset in zip(variables, datasets):
        name = dataset.replace(".", "_") + "_" + variable
        query_parameters = parameters.copy()
        query_parameters["data"] = name

        response = requests.get(data_url, params=query_parameters,
                                headers=query_headers)
        response.raise_for_status()
        buffer = StringIO(response.text)

        var_meta = {}
        while (line := buffer.readline().rstrip()) != "":
            key, value = line.split(",", maxsplit=1)
            var_meta[key] = value

        meta[variable] = var_meta
        df = pd.read_csv(buffer, index_col=0, parse_dates=True)
        df = df.replace(float(var_meta["undef"]), np.nan)

        data[variable] = df["Data"]

    # copy lat/lon to the top level, for consistency
    # with other iotools functions
    meta["latitude"] = float(var_meta["lat"])
    meta["longitude"] = float(var_meta["lon"])

    df = pd.DataFrame(data)
    df.index = df.index.tz_localize("UTC")

    if map_variables:
        for col in df.columns:
            if col in UNITS:
                convert = UNITS[col]
                df[col] = convert(df[col])

        df = df.rename(columns=VARIABLE_MAP)

    return df, meta
