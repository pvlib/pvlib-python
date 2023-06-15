import requests
import pandas as pd
import numpy as np


def get_acis_precipitation(latitude, longitude, start, end, dataset,
                           url="https://data.rcc-acis.org/GridData", **kwargs):
    """
    Retrieve estimated daily precipitation data from the Applied Climate
    Information System (ACIS).

    The Applied Climate Information System (ACIS) was developed and is
    maintained by the NOAA Regional Climate Centers (RCCs) and brings together
    climate data from many sources.  This function accesses precipitation
    datasets covering the United States, although the exact domain
    varies by dataset [1]_.

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive
    longitude : float
        in decimal degrees, between -180 and 180, east is positive
    start : datetime-like
        First day of the requested period
    end : datetime-like
        Last day of the requested period
    dataset : int
        A number indicating which gridded dataset to query.  Options include:

        * 1: NRCC Interpolated
        * 2: Multi-Sensor Precipitation Estimates
        * 3: NRCC Hi-Res
        * 21: PRISM
    
        See [2]_ for the full list of options.

    url : str, default: 'https://data.rcc-acis.org/GridData'
        API endpoint URL
    kwargs:
        Optional parameters passed to ``requests.get``.

    Returns
    -------
    data : pandas.Series
        Daily rainfall [mm]
    metadata : dict
        Coordinates for the selected grid cell

    Raises
    ------
    requests.HTTPError
        A message from the ACIS server if the request is rejected

    Notes
    -----
    The returned precipitation values are 24-hour aggregates, but
    the aggregation period may not be midnight to midnight in local time.
    For example, PRISM data is aggregated from 12:00 to 12:00 UTC,
    meaning PRISM data labeled May 26 reflects to the 24 hours ending at
    7:00am Eastern Standard Time on May 26.

    Examples
    --------
    >>> prism, metadata = get_acis_precipitation(40.0, -80.0, '2020-01-01',
    >>>                                          '2020-12-31', dataset=21)

    References
    ----------
    .. [1] `ACIS Gridded Data <http://www.rcc-acis.org/docs_gridded.html>`_
    .. [2] `ACIS Web Services <http://www.rcc-acis.org/docs_webservices.html>`_
    .. [3] `NRCC <http://www.nrcc.cornell.edu/>`_
    .. [4] `Multisensor Precipitation Estimates
           <https://www.weather.gov/marfc/Multisensor_Precipitation>`_
    .. [5] `PRISM <https://prism.oregonstate.edu/>`_
    """
    elems = [
        # other variables exist, but are not of interest for PV modeling
        {"name": "pcpn", "interval": "dly", "units": "mm"},
    ]
    params = {
        'loc': f"{longitude},{latitude}",
        # use pd.to_datetime so that strings (e.g. '2021-01-01') are accepted
        'sdate': pd.to_datetime(start).strftime('%Y-%m-%d'),
        'edate': pd.to_datetime(end).strftime('%Y-%m-%d'),
        'grid': str(dataset),
        'elems': elems,
        'output': 'json',
        # [2]_ lists "ll" (lat/lon) and "elev" (elevation) as the available
        # options for "meta".  However, including "elev" when dataset=2
        # results in an "unknown meta elev" error.  "ll" works on all
        # datasets.  There doesn't seem to be any other metadata available.
        'meta': ["ll"],
    }
    response = requests.post(url, json=params,
                             headers={"Content-Type": "application/json"},
                             **kwargs)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        raise requests.HTTPError(payload['error'], response=response)

    metadata = payload['meta']
    metadata['latitude'] = metadata.pop('lat')
    metadata['longitude'] = metadata.pop('lon')

    df = pd.DataFrame(payload['data'], columns=['date', 'precipitation'])
    rainfall = df.set_index('date')['precipitation']
    rainfall = rainfall.replace(-999, np.nan)
    rainfall.index = pd.to_datetime(rainfall.index)
    return rainfall, metadata
