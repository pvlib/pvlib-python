import requests
import pandas as pd
from io import BytesIO, StringIO
import zipfile
import time


VARIABLE_MAP = {
    # short names
    'd2m': 'temp_dew',
    't2m': 'temp_air',
    'sp': 'pressure',
    'ssrd': 'ghi',
    'tp': 'precipitation',

    # long names
    '2m_dewpoint_temperature': 'temp_dew',
    '2m_temperature': 'temp_air',
    'surface_pressure': 'pressure',
    'surface_solar_radiation_downwards': 'ghi',
    'total_precipitation': 'precipitation',
}


def same(x):
    return x


def k_to_c(temp_k):
    return temp_k - 273.15


def j_to_w(j):
    return j / 3600


def m_to_cm(m):
    return m / 100

UNITS = {
    'u100': same,
    'v100': same,
    'u10': same,
    'v10': same,
    'd2m': k_to_c,
    't2m': k_to_c,
    'msl': same,
    'sst': k_to_c,
    'skt': k_to_c,
    'sp': same,
    'ssrd': j_to_w,
    'strd': j_to_w,
    'tp': m_to_cm,
}


def get_era5(latitude, longitude, start, end, variables, api_key,
             map_variables=True, timeout=60,
             url='https://cds.climate.copernicus.eu/api/retrieve/v1/'):
    """
    Retrieve ERA5 reanalysis data from the ECMWF's Copernicus Data Store.

    A CDS API key is needed to access this API.  Register for one at [1]_.

    This API [2]_ provides a subset of the full ERA5 dataset.  See [3]_ for
    the available variables.  Data are available on a 0.25° x 0.25° grid.

    Parameters
    ----------
    latitude : float
        In decimal degrees, north is positive (ISO 19115).
    longitude: float
        In decimal degrees, east is positive (ISO 19115).
    start : datetime like or str
        First day of the requested period.
    end : datetime like or str
        Last day of the requested period.
    variables : list of str
        List of variable names to retrieve.  See [1]_ for options.
    api_key : str
        ECMWF CDS API key.
    map_variables : bool, default True
        When true, renames columns of the DataFrame to pvlib variable names
        where applicable. See variable :const:`VARIABLE_MAP`.
    timeout : int, default 60
        Number of seconds to wait for the requested data to become available
        before timeout.
    url : str, optional
        API endpoint URL.

    Raises
    ------
    Exception
        If ``timeout`` is reached without the job finishing.

    Returns
    -------
    data : pd.DataFrame
        Time series data. The index corresponds to the start of the interval.
    meta : dict
        Metadata.

    References
    ----------
    .. [1] https://cds.climate.copernicus.eu/
    .. [2] https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview
    .. [3] https://confluence.ecmwf.int/pages/viewpage.action?pageId=505390919
    """  # noqa: E501
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")

    headers = {'PRIVATE-TOKEN': api_key}

    # allow variables to be specified with pvlib names
    reverse_map = {v: k for k, v in VARIABLE_MAP.items()}
    variables = [reverse_map.get(k, k) for k in variables]

    # Step 1: submit data request (add it to the queue)
    params = {
        "inputs": {
            "variable": variables,
            "location": {"longitude": longitude, "latitude": latitude},
            "date": [f"{start}/{end}"],
            "data_format": "csv"
        }
    }
    slug = "processes/reanalysis-era5-single-levels-timeseries/execution"
    response = requests.post(url + slug, json=params, headers=headers)
    submission_response = response.json()
    if not response.ok:
        raise Exception(response.json())  # likely need to accept license
    job_id = submission_response['jobID']
    
    # Step 2: poll until the data request is ready
    slug = "jobs/" + job_id
    poll_interval = 1
    num_polls = 0
    while True:
        response = requests.get(url + slug, headers=headers)
        poll_response = response.json()
        job_status = poll_response['status']

        if job_status == 'successful':
            break  # ready to proceed to next step
        elif job_status == 'failed':
            msg = (
                'Request failed. Please check the ECMWF website for details: '
                'https://cds.climate.copernicus.eu/requests?tab=all'
            )
            raise Exception(msg)

        num_polls += 1
        if num_polls * poll_interval > timeout:
            raise Exception(
                'Request timed out. Try increasing the timeout parameter or '
                'reducing the request size.'
            )

        time.sleep(1)

    # Step 3: get the download link for our requested dataset
    slug = "jobs/" + job_id + "/results"
    response = requests.get(url + slug, headers=headers)
    results_response = response.json()
    download_url = results_response['asset']['value']['href']

    # Step 4: finally, download our dataset.  it's a zipfile of one CSV
    response = requests.get(download_url)
    zipbuffer = BytesIO(response.content)
    archive = zipfile.ZipFile(zipbuffer)
    filename = archive.filelist[0].filename
    csvbuffer = StringIO(archive.read(filename).decode('utf-8'))
    df = pd.read_csv(csvbuffer)

    # and parse into the usual formats
    metadata = submission_response['metadata']  # include messages from ECMWF
    metadata['jobID'] = job_id
    if not df.empty:
        metadata['latitude'] = df['latitude'].values[0]
        metadata['longitude'] = df['longitude'].values[0]

    df.index = pd.to_datetime(df['valid_time']).dt.tz_localize('UTC')
    df = df.drop(columns=['valid_time', 'latitude', 'longitude'])

    if map_variables:
        # convert units and rename
        for shortname in df.columns:
            converter = UNITS[shortname]
            df[shortname] = converter(df[shortname])
        df = df.rename(columns=VARIABLE_MAP)

    return df, metadata
