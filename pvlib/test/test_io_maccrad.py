import os

from pvlib.io.maccrad import read_maccrad


maccrad_url_base = "https://raw.githubusercontent.com/dacoex/pvlib_data/master/MACC-RAD/carpentras/"

maccrad_csv = "irradiation-0e2a19f2-abe7-11e5-a880-5254002dbd9b.csv"
maccrad_url_full = maccrad_url_base + maccrad_csv
maccrad_csv_dir = os.path.join("..", "..", "..", "pvlib_data", "MACC-RAD", "carpentras")
maccrad_csv_path = os.path.join(maccrad_csv_dir, maccrad_csv)
                               
data_maccrad = read_maccrad(maccrad_csv_path, output='loc')

maccrad_loc = data_maccrad[0]
maccrad_df = data_maccrad[1]

def test_location_coord():
    assert (44.0830, 5.0590, 97.00) == (maccrad_loc.latitude, maccrad_loc.longitude, 
                             maccrad_loc.altitude)
    

def test_location_tz():
    assert 'Europe/Paris' == maccrad_loc.tz
    
    
def test_maccrad_recolumn():
    assert 'Clear sky GHI' in maccrad_df.columns
    
def test_maccrad_norecolumn():
    assert 'Clear sky GHI' in maccrad_df.columns
    
def test_maccrad_coerce_year():
    coerce_year = 2010
    assert (maccrad_df.index.year[0] == coerce_year)
    
    
def test_maccrad():
    read_maccrad(maccrad_csv_path)

##FIXME: this still crashes
### if data is on remotely on github
#
#
#import urllib
#
##f = urllib.request.urlopen(maccrad_url_full)
#
##from urllib.parse import urlparse
##response = urlparse(maccrad_url_full)
#
#
##from urllib.request import urlopen
##response = urlopen(maccrad_url_full)
##response = response.decode('utf-8')
#
#
#
## http://stackoverflow.com/questions/4981977/how-to-handle-response-encoding-from-urllib-request-urlopen
#req=urllib.request.urlopen(maccrad_url_full)
#charset=req.info().get_content_charset()
#response=req.read().decode(charset)
#
##data = urllib.request.urlopen(maccrad_url_full).read()
#
#lines = response.splitlines(True)
## http://stackoverflow.com/questions/23131227/how-to-readlines-from-urllib
#
#
#
##import requests
##response = requests.get(maccrad_url_full).text
#
#for line in response:
##    print (line)
#    if line.startswith( "# Latitude"):
##        lat_line = line
##        lat = float(lat_line.split(':')[1])
#        lat = float(line.split(':')[1])
#    if line.startswith( "# Longitude"):
##        lon_line = line
##        lon = float(lon_line.split(':')[1])
#        lon = float(line.split(':')[1])
#    if line.startswith( "# Altitude"):
##    if "Altitude" in line:
#        alt_line = line
#        alt = float(alt_line.split(':')[1])
##        alt = float(line.split(':')[1])


