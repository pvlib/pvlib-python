"""
Get, read, and parse data from 
`CGIAR GeoPortal <https://srtm.csi.cgiar.org/>`_.

For more information, see the following links:
* `FAQ <https://srtm.csi.cgiar.org/faq/>`_


"""
import math
from zipfile import ZipFile
import requests
import io


def _lat_lon_to_query(longitude, latitude, srtm_arc_sec=3):
    r'''Converts latitude, longitude from the format used as
    input to the other functions to format used by CGIAR
    ----------
    longitude : numeric
        longitude value, negative W of prime meridian
    latitude: numeric
        latitude value
    srtm_arc_sec: numeric {1,3}
        The resolution (arc-seconds) of the desired DEM.
        Either 1 or 3

    Returns
    -------
    rounded_lon : numeric
        Rounded/adjusted longitude value
    rounded_lat : numeric
        Rounded/adjusted latitude value

    '''
    rounded = (int(math.floor(longitude)), int(math.floor(latitude)))
    if srtm_arc_sec == 1:
        return rounded
    elif srtm_arc_sec == 3:
        rounded_lon, rounded_lat = rounded
        return (rounded_lon + 180) // 5 + 1, (64 - rounded_lat) // 5
    else:
        raise Exception("Please use SRTM 1 Arc Second or SRTM 3 Arc Second")


def download_SRTM(latitude, longitude, srtm_arc_sec=3, 
                  path_to_save="./", proxies=None):
    r'''Downloads a SRTM DEM tile from CGIAR,
    saves it to a path, and loads it as an array
    ----------
    latitude: numeric
        latitude value to be included in the DEM
    longitude : numeric
        longitude value to be included in the DEM,
        negative W of prime meridian
    srtm_arc_sec: numeric, {1,3}
        The resolution (arc-seconds) of the desired DEM.
        Either 1 or 3
    path_to_save: string
        The base path to save the DEM as a .tif file
    proxies: dict
        Proxy table for a corporate network

    Returns
    -------
    img : np.array
        Numpy array filled with elevation values in [m]
    path : string
        Path and filename where the DEM .tif filed was saved

    '''
    import skimage
    long, lat = _lat_lon_to_query(longitude, latitude, srtm_arc_sec)
    base_url = 'https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/'
    query_URL = base_url + f'srtm_{ long:02d }_{ lat:02d }.zip'
    res = requests.get(query_URL, proxies=proxies, verify=False)
    res.raise_for_status()
    zipfile = ZipFile(io.BytesIO(res.content))
    ext = '.tif'
    files = zipfile.namelist()
    file = [f for f in files if ext in f][0]
    path= zipfile.extract(file, path=path_to_save)
    img = skimage.io.imread(path)
    return img, path
