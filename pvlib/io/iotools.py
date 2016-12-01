import pytz
# How to get a time zone from a location using 
# latitude and longitude coordinates?
# http://stackoverflow.com/a/16086964
# upstream: https://github.com/MrMinimal64/timezonefinder
from timezonefinder import TimezoneFinder

def get_loc_latlon(lat, lon):
    """Returns the timezone for a coordinate pair.
    """
    
    
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    
    return tz
    
def localise_df(df_notz, tz_source_str='UTC', tz_target_str):
    """
    localises a pandas.DataFrame (df) to the target time zone of the pvlib-Location    
    
    Assumes that the input df does not have a timezone
    
    """
    
    tz_source_str = pytz.timezone(tz_source_str)
    tz_target = pytz.timezone(tz_target_str)
    
    df_tz_target = df_notz.tz_convert(tz_source)
    
    return df_tz_target
