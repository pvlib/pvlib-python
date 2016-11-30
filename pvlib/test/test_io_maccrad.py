import os

from pvlib.io.maccrad import read_maccrad


maccrad_url_base = "https://raw.githubusercontent.com/dacoex/pvlib_data/master/MACC-RAD/carpentras/"

maccrad_csv = "irradiation-0e2a19f2-abe7-11e5-a880-5254002dbd9b.csv"
maccrad_url_full = maccrad_url_base + maccrad_csv
maccrad_csv_dir = os.path.join("..", "..", "..", "pvlib_data", "MACC-RAD", "carpentras")
maccrad_csv_path = os.path.join(maccrad_csv_dir, maccrad_csv)
                               
#data_maccrad = read_maccrad(maccrad_csv, output='loc')


## if data is on remotely on github

import urllib

#f = urllib.request.urlopen(maccrad_url_full)

#from urllib.parse import urlparse
#response = urlparse(maccrad_url_full)


from urllib.request import urlopen
response = urlopen(maccrad_url_full)
response = response.decode('utf-8')




req=urllib.request.urlopen(maccrad_url_full)
charset=req.info().get_content_charset()
content=req.read().decode(charset)

data = urllib.request.urlopen(maccrad_url_full).read()

lines = data.splitlines(True)
# http://stackoverflow.com/questions/23131227/how-to-readlines-from-urllib



import requests

response = requests.get(maccrad_url_full).text

for line in response:
#    print (line)
    if line.startswith( "# Latitude"):
#        lat_line = line
#        lat = float(lat_line.split(':')[1])
        lat = float(line.split(':')[1])
    if line.startswith( "# Longitude"):
#        lon_line = line
#        lon = float(lon_line.split(':')[1])
        lon = float(line.split(':')[1])
#    if line.startswith( "# Altitude"):
    if "Altitude" in line:
        alt_line = line
        alt = float(alt_line.split(':')[1])
#        alt = float(line.split(':')[1])


## if file is on local drive
f = open(maccrad_csv_path)
for line in f:
    if "Latitude" in line:
#    print (line)
#    if line.startswith( "# Latitude"):
        lat_line = line
        lat = float(lat_line.split(':')[1])
#        lat = float(line.split(':')[1])
    if "Longitude" in line:
#    if line.startswith( "# Longitude"):
        lon_line = line
        lon = float(lon_line.split(':')[1])
        lon = float(line.split(':')[1])
#    if line.startswith( "# Altitude"):
    if "Altitude" in line:
        alt_line = line
        alt = float(alt_line.split(':')[1])
#        alt = float(line.split(':')[1])


from timezonefinder import TimezoneFinder
tf = TimezoneFinder()
tz = tf.timezone_at(lng=lon, lat=lat)

from pvlib.location import Location

location = Location(lat, lon, name=maccrad_csv.split('.')[0], altitude=alt,
                       tz=tz)