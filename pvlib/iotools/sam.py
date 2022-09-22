"""Functions for reading and writing SAM data files."""

import pandas as pd


def saveSAM_WeatherFile(data, metadata, savefile='SAM_WeatherFile.csv',
                        standardSAM=True, includeminute=False):
    """
    Saves dataframe with weather data from pvlib format on SAM-friendly format.

    Parameters
    -----------
    data : pandas.DataFrame
        timeseries data in PVLib format. Should be TZ converted (not UTC).
        Ideally it is one sequential year data; if not suggested to use
        standardSAM = False.
    metdata : dictionary
        Dictionary with 'latitude', 'longitude', 'elevation', 'source',
        and 'TZ' for timezone.
    savefile : str
        Name of file to save output as.
    standardSAM : boolean
        This checks the dataframe to avoid having a leap day, then averages it
        to SAM style (closed to the right),
        and fills the years so it starst on YEAR/1/1 0:0 and ends on
        YEAR/12/31 23:00.
    includeminute ; Bool
        For hourly data, if SAM input does not have Minutes, it calculates the
        sun position 30 minutes prior to the hour (i.e. 12 timestamp means sun
         position at 11:30).
        If minutes are included, it will calculate the sun position at the time
        of the timestamp (12:00 at 12:00)
        Set to true if resolution of data is sub-hourly.

    Returns
    -------
    Nothing, it just writes the file.

    """

    def _is_leap_and_29Feb(s):
        ''' Creates a mask to help remove Leap Years. Obtained from:
            https://stackoverflow.com/questions/34966422/remove-leap-year-day-
            from-pandas-dataframe/34966636
        '''
        return (s.index.year % 4 == 0) & \
               ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & \
               (s.index.month == 2) & (s.index.day == 29)

    def _averageSAMStyle(df, interval='60T', closed='right', label='right'):
        ''' Averages subhourly data into hourly data in SAM's expected format.
        '''
        df = df.resample(interval, closed=closed, label=label).mean()
        return df

    def _fillYearSAMStyle(df, freq='60T'):
        ''' Fills year
        '''
        # add zeros for the rest of the year
        if freq is None:
            freq = pd.infer_freq(df.index)
        # add a timepoint at the end of the year
        # idx = df.index
        # apply correct TZ info (if applicable)
        tzinfo = df.index.tzinfo
        starttime = pd.to_datetime('%s-%s-%s %s:%s' % (df.index.year[0], 1, 1,
                                                       0, 0)
                                   ).tz_localize(tzinfo)
        endtime = pd.to_datetime('%s-%s-%s %s:%s' % (df.index.year[-1], 12, 31,
                                                     23, 60-int(freq[:-1]))
                                 ).tz_localize(tzinfo)

        df2 = _averageSAMStyle(df, freq)
        df2.iloc[0] = 0  # set first datapt to zero to forward fill w zeros
        df2.iloc[-1] = 0  # set last datapt to zero to forward fill w zeros
        df2.loc[starttime] = 0
        df2.loc[endtime] = 0
        df2 = df2.resample(freq).ffill()
        return df2

    # Modify this to cut into different years. Right now handles partial year
    # and sub-hourly interval.
    if standardSAM:
        filterdatesLeapYear = ~(_is_leap_and_29Feb(data))
        data = data[filterdatesLeapYear]
        data = _fillYearSAMStyle(data)

    # metadata
    latitude = metadata['latitude']
    longitude = metadata['longitude']
    elevation = metadata['elevation']
    timezone_offset = metadata['TZ']
    source = metadata['source']

    # make a header
    header = '\n'.join(['Source,Latitude,Longitude,Time Zone,Elevation',
                        source + ',' + str(latitude) + ',' + str(longitude)
                        + ',' + str(timezone_offset) + ',' +
                        str(elevation)])+'\n'

    savedata = pd.DataFrame({'Year': data.index.year,
                             'Month': data.index.month,
                             'Day': data.index.day,
                             'Hour': data.index.hour})

    if includeminute:
        savedata['Minute'] = data.index.minute

    windspeed = list(data.wind_speed)
    temp_amb = list(data.temp_air)
    savedata['Wspd'] = windspeed
    savedata['Tdry'] = temp_amb

    if 'dni' in data:
        dni = list(data.dni)
        savedata['DHI'] = dni

    if 'dhi' in data:
        dhi = list(data.dhi)
        savedata['DNI'] = dhi

    if 'ghi' in data:
        ghi = list(data.ghi)
        savedata['GHI'] = ghi

    if 'poa' in data:
        poa = list(data.poa)
        savedata['POA'] = poa

    if 'albedo' in data:
        albedo = list(data.albedo)
        savedata['Albedo'] = albedo
        
        # Not elegant but seems to work for the standardSAM format
        if standardSAM and savedata.Albedo.iloc[0] == 0:
            savedata.loc[savedata.index[0],'Albedo'] = savedata.loc[savedata.index[1]]['Albedo']
            savedata.loc[savedata.index[-1],'Albedo'] = savedata.loc[savedata.index[-2]]['Albedo']

    with open(savefile, 'w', newline='') as ict:
        # Write the header lines, including the index variable for
        # the last one if you're letting Pandas produce that for you.
        # (see above).
        for line in header:
            ict.write(line)

        savedata.to_csv(ict, index=False)


def tz_convert(df, tz_convert_val, metadata=None):
    """
    Support function to convert metdata to a different local timezone.
    Particularly for GIS weather files which are returned in UTC by default.

    Parameters
    ----------
    df : DataFrame
        A dataframe in UTC timezone
    tz_convert_val : int
        Convert timezone to this fixed value, following ISO standard
        (negative values indicating West of UTC.)
    Returns: metdata, metadata


    Returns
    -------
    df : DataFrame
        Dataframe in the converted local timezone.
    metadata : dict
        Adds (or updates) the existing Timezone in the metadata dictionary

    """
    import pytz
    if (type(tz_convert_val) == int) | (type(tz_convert_val) == float):
        df = df.tz_convert(pytz.FixedOffset(tz_convert_val*60))

        if metadata is not None:
            metadata['TZ'] = tz_convert_val
            return df, metadata
    return df
