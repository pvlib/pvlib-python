# standard library imports 
import logging

# related third party imports
import pandas as pd
import pytz

# local application/library specific imports 
from pvlib.location import Location
from pvlib.iotools.iotools import get_loc_latlon, localise_df

# required dateconverters
def dtp_soda_pro_macc_rad(date):
    """
    datetime converter for MACC-RAD data and others
    """
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
        if "Title" in line:
            title = line.split(':')[1].split('(')[0].strip()
            name = title
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
        if "Time reference" in line:
            if "Universal time (UT)" in line:
                tz_raw = 'UTC'
            else:
                logging.debug('No metadata on timezone found in input file')
                logging.debug('Assuming UTC as default timezone')
                tz_raw = 'UTC'

    tz_loc = get_loc_latlon(lat, lon)
    
    
    
    location = Location(lat, lon, name=name, altitude=alt,
                           tz=tz_loc)
    
    #XXX
    metadata = None
    
    return tz_raw, location, metadata

#XXX convert to pvlib conventions
def maccrad_df_to_pvlib(df_raw, tz_raw, loc, localise=True):
    """Change some properties of the dataframe to be more compliant with pvlib
    
    * localisation
    * column renaming
    * setting dataframe name description according to datasource
    
    """

    if localise:
        # timezone localisations
        df_pvlib = localise_df(df_raw, tz_source_str=tz_raw, 
                               tz_target_str=loc.tz)
    # column renaming
    df_pvlib.index.name = 'datetime'
    
    # name the dataframe according to data source 
    df_pvlib.df_name = loc.name
  
    return df_pvlib
    
#def maccrad_df_to_pvlib(df):
#    
#    
#    pass
   
    
#XXX read data
def read_maccrad(file_csv, loc_name=None, skiprows=40, output='all'):
    """
    Read MACC-RAD current format for files into a pvlib-ready dataframe
    
    Parameters
    ----------
    file_csv : a csv file corresponding to the reader format   
    skiprows : skiprows as in pandas.io.read_csv
        The example files require skipping 40 rows.
    output : all / loc / df_pvlib        
        df_raw returns only the a pandas.DataFrame using the raw data from the 
                        file (this can be helpful for debugging and comparison 
                        with restuls obtained with other programs, e.g. 
                        spreadsheet)
        all returns df_raw, returns a pandas.DataFrame reformatted to match the 
                         `variable naming convention <variables_style_rules>`
                         for pvliball outputs, and a location created from 
                         the metadata in raw input file header
                         as tuple

    
    """
    df_raw = pd.read_csv(file_csv, sep=';', skiprows=skiprows, header=0,
                      index_col=0, parse_dates=True,
                      date_parser=dtp_soda_pro_macc_rad)
#TODO: add loc_name
#TODO: add reformat needs loc!
#TODO: add simplify output options raw or all
#    print (output)
    if output == 'df_raw':
        res = df_raw
    if output == 'all':
        tz_raw, loc, metadata = read_maccrad_metadata(file_csv)
        loc.name = (loc.name + ' @ ' + 'lat (deg. N), lon (deg. E): ' + 
                    str(loc.latitude) + ', ' + str(loc.longitude))
        df_pvlib = maccrad_df_to_pvlib(df_raw, tz_raw, loc, localise=True)
#        res = df_pvlib
        res = (df_raw, df_pvlib, loc)
#    if output == 'loc':
#        
#        res = loc, df
#    if output == 'all':
#        # not calculated outside conditional to reduce overhead of metadata 
#        # reading if not desired
#        loc = read_maccrad_metadata(file_csv)
#        res = (df_raw, df_pvlib, loc)
    
    
    return res

 