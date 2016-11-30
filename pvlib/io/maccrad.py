import pandas as pd

# required dateconverters
def dtp_soda_pro_macc_rad(date):
    "datetime converter for MACC-RAD data and others
    datetime_stamp = date.split('/')[0]

    return datetime_stamp
   
#XXX read metadata / header

def read_maccrad_metadata(file_csv, name='maccrad'):
    """
    Read metadata from commented lines of the file
    * coordinates
    * altitude
    
    Retrieve timezone
    """
    if not name:
        name = file_csv.split('.')[0]
    
    ## if file is on local drive
    f = open(file_csv)
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
    
    # How to get a time zone from a location using 
    # latitude and longitude coordinates?
    # http://stackoverflow.com/a/16086964
    # upstream: https://github.com/MrMinimal64/timezonefinder
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    
    from pvlib.location import Location
    
    location = Location(lat, lon, name=name, altitude=alt,
                           tz=tz)
    
    return location
    
#XXX read data
def read_maccrad(file_csv, output='df'):
    """
    Read MACC-RAD current format for files into a pvlib-ready dataframe
    """
    df = pd.read_csv(file_csv, sep=';', skiprows=40, header=0,
                      index_col=0, parse_dates=True,
                      date_parser=dtp_soda_pro_macc_rad)
    
    if output == 'loc':
        loc = read_maccrad_metadata(file_csv)
        res = (loc, df)
    else:
        res = df
    
    
    return res

 