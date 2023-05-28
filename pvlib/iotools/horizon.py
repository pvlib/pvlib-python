"""Functions to access horizon data from MINES ParisTech.

.. codeauthor:: Adam R. Jensen<adam-r-j@hotmail.com>
"""

import requests
import pandas as pd
import io


def get_mines_horizon(latitude, longitude, altitude=None, ground_offset=0,
                      url='http://toolbox.1.webservice-energy.org/service/wps',
                      **kwargs):
    """Retrieve horizon profile from Shuttle Radar Topography Mission (SRTM).

    Service is provided by MINES ParisTech - Armines (France)

    Parameters
    ----------
    latitude: float
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    altitude: float, optional
        Altitude in meters. If None, then the altitude is determined from the
        NASA SRTM database.
    ground_offset: float, optional
        Vertical offset in meters for the point of view for which to calculate
        horizon profile.
    url: str, default: 'http://toolbox.1.webservice-energy.org/service/wps'
        Base URL for MINES ParisTech horizon profile API
    kwargs:
        Optional parameters passed to requests.get.

    Returns
    -------
    data : pd.Series
        Pandas Series of the retrived horizon elevation angles. Index is the
        corresponding horizon azimuth angles.
    metadata : dict
        Metadata

    Notes
    -----
    The azimuthal resolution is one degree. Also, the reutned horizon
    elevations can also be negative.
    """
    if altitude is None:  # API will then infer altitude
        altitude = -999

    # Manual formatting of the input parameters seperating each by a semicolon
    data_inputs = f"latitude={latitude};longitude={longitude};altitude={altitude}"  # noqa: E501

    params = {
        'service': 'WPS',
        'request': 'Execute',
        'identifier': 'compute_horizon_srtm',
        'version': '1.0.0',
        'ground_offset': ground_offset,
        }

    # The DataInputs parameter of the URL has to be manually formatted and
    # added to the base URL as it contains sub-parameters seperated by
    # semi-colons, which gets incorrectly formatted by the requests function
    # if passed using the params argument.
    res = requests.get(url + '?DataInputs=' + data_inputs, params=params,
                       **kwargs)

    # The response text is first converted to a StringIO object as otherwise
    # pd.read_csv raises a ValueError stating "Protocol not known:
    # <!-- PyWPS 4.0.0 --> <wps:ExecuteResponse xmlns:gml="http"
    # Alternatively it is possible to pass the url straight to pd.read_csv
    horizon = pd.read_csv(io.StringIO(res.text), skiprows=27, nrows=360,
                          delimiter=';', index_col=0,
                          names=['horizon_azimuth', 'horizon_elevation'])
    horizon = horizon['horizon_elevation']  # convert to series
    # Note, there is no way to detect if the request is correct. In all cases,
    # the API always returns a status code of OK/200 and no useful error
    # message.

    meta = {'data_provider': 'MINES ParisTech - Armines (France)',
            'databse': 'Shuttle Radar Topography Mission (SRTM)',
            'latitude': latitude, 'longitude': longitude, 'altitude': altitude,
            'ground_offset': ground_offset}

    return horizon, meta

# %%
latitude = 55.70
longitude = 12.567
altitude = -999

latitude, longitude = 22.019026196747035, -159.75502400435346

data_mines, meta_mines = get_mines_horizon(latitude, longitude)
data_pvgis, meta_pvgis = get_pvgis_horizon(latitude, longitude)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data_mines, label='MINES')
ax.plot(data_pvgis, label='PVGIS')
ax.legend()
