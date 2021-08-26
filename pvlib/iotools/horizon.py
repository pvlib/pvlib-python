
import requests
import pandas as pd
import io


def get_srtm_horizon(latitude, longitude, altitude, timeout=30):
    """Retrieve horizon profile from Shuttle Radar Topography Mission (SRTM)

    Service is provided by MINES ParisTech - Armines (France)

    Parameters
    ----------
    latitude: float
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    altitude: float, optional
        Altitude in meters. If None, then the altitude is determined from the
        NASA SRTM database
    timeout : int, default: 30
        Time in seconds to wait for server response before timeout

    Returns
    -------
    horizon: dataframe
        A dataframe with the first column being the azimuth (0,359) in 1 degree
        steps and the second column being the corresponding horizon elevation
        in degrees.

    Notes
    -----
    Horizon elevation can also be negative.
    """
    # Different servers?
    base_url = "http://toolbox.webservice-energy.org/service/wps"

    data_inputs_dict = {
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude}

    # Manual formatting of the input parameters seperating each by a semicolon
    data_inputs = ";".join([f"{key}={value}" for key, value in
                            data_inputs_dict.items()])

    params = {
        'service': 'WPS',
        'request': 'Execute',
        'identifier': 'compute_horizon_srtm',
        'version': '1.0.0'}

    # The DataInputs parameter of the URL has to be manually formatted and
    # added to the base URL as it contains sub-parameters seperated by
    # semi-colons, which gets incorrectly formatted by the requests function
    # if passed using the params argument.
    res = requests.get(base_url + '?DataInputs=' + data_inputs, params=params,
                       timeout=timeout)

    # The response text is first converted to a StringIO object as otherwise
    # pd.read_csv raises a ValueError stating "Protocol not known:
    # <!-- PyWPS 4.0.0 --> <wps:ExecuteResponse xmlns:gml="http"
    horizon = pd.read_csv(io.StringIO(res.text), skiprows=25, nrows=360,
                          delimiter=';')

    return horizon
