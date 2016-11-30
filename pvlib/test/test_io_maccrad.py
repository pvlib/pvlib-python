from pvlib.io.maccrad import read_maccrad


maccrad_url_base = "https://raw.githubusercontent.com/dacoex/pvlib_data/master/MACC-RAD/carpentras/"
maccrad_csv = "irradiation-0e2a19f2-abe7-11e5-a880-5254002dbd9b.csv"
maccrad_url_full = maccrad_url_base + maccrad_csv
#maccrad_csv = r"C:\Users\hzr\Documents\GitHub\pvlib_data\MACC-RAD\carpentras\irradiation-0e2a19f2-abe7-11e5-a880-5254002dbd9b.csv"

#data_maccrad = read_maccrad(maccrad_csv, output='loc')

import urllib

f = urllib.request.urlopen(maccrad_url_full)

#from urllib.parse import urlparse
#response = urlparse(maccrad_url_full)


from urllib.request import urlopen
response = urlopen(maccrad_url_full)

#from urllib.request import urlretrieve
#urlretrieve(url, "result.txt")

f = response.readlines()
for line in f:
    print (line)
    if "Latitude" in line:
#        lat_line = line
#        lat = float(lat_line.split(':')[1])
        lat = float(line.split(':')[1])
    if "Longitude" in line:
#        lon_line = line
#        lon = float(lon_line.split(':')[1])
        lon = float(line.split(':')[1])
    if "Altitude" in line:
#        alt_line = line
#        alt = float(alt_line.split(':')[1])
        alt = float(line.split(':')[1])
        
        
from timezonefinder import TimezoneFinder
tf = TimezoneFinder()
tz = tf.timezone_at(lng=lon, lat=lat)

from pvlib.location import Location

location = Location(lat, lon, name=maccrad_csv.split('.')[0], altitude=alt,
                       tz=tz)