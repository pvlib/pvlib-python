"""
Import TMY2 and TMY3 data.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import pdb
import re
import datetime
import dateutil
import csv 

import pandas as pd
import numpy as np

from . import pvl_tools



def readtmy3(filename=None):
    '''
    Read a TMY3 file in to a pandas dataframe

    Read a TMY3 file and make a pandas dataframe of the data. Note that values
    contained in the struct are unchanged from the TMY3 file (i.e. units 
    are retained). In the case of any discrepencies between this
    documentation and the TMY3 User's Manual ([1]), the TMY3 User's Manual
    takes precedence.

    If a FileName is not provided, the user will be prompted to browse to
    an appropriate TMY3 file.

    Parameters
    ----------

    FileName : string 
    An optional argument which allows the user to select which
    TMY3 format file should be read. A file path may also be necessary if
    the desired TMY3 file is not in the MATLAB working path.

    Returns
    -------

    TMYDATA : DataFrame

    A pandas dataframe, is provided with the components in the table below. Note
    that for more detailed descriptions of each component, please consult
    the TMY3 User's Manual ([1]), especially tables 1-1 through 1-6. 

    meta : struct
    struct of meta data is created, which contains all 
    site metadata available in the file

    Notes
    -----

    ===============   ======  ===================  
    meta field        format  description
    ===============   ======  ===================  
    meta.altitude     Float   site elevation
    meta.latitude     Float   site latitudeitude
    meta.longitude    Float   site longitudeitude
    meta.Name         String  site name
    meta.State        String  state
    meta.TZ           Float   timezone
    meta.USAF         Int     USAF identifier
    ===============   ======  ===================  

    =============================       ======================================================================================================================================================
    TMYData field                       description
    =============================       ======================================================================================================================================================
    TMYData.Index                       A pandas datetime index. NOTE, the index is currently timezone unaware, and times are set to local standard time (daylight savings is not indcluded)
    TMYData.ETR                         Extraterrestrial horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.ETRN                        Extraterrestrial normal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.GHI                         Direct and diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.GHISource                   See [1], Table 1-4
    TMYData.GHIUncertainty              Uncertainty based on random and bias error estimates                        see [2]
    TMYData.DNI                         Amount of direct normal radiation (modeled) recv'd during 60 mintues prior to timestamp, Wh/m^2
    TMYData.DNISource                   See [1], Table 1-4
    TMYData.DNIUncertainty              Uncertainty based on random and bias error estimates                        see [2]
    TMYData.DHI                         Amount of diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
    TMYData.DHISource                   See [1], Table 1-4
    TMYData.DHIUncertainty              Uncertainty based on random and bias error estimates                        see [2]
    TMYData.GHillum                     Avg. total horizontal illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.GHillumSource               See [1], Table 1-4
    TMYData.GHillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]
    TMYData.DNillum                     Avg. direct normal illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.DNillumSource               See [1], Table 1-4
    TMYData.DNillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]
    TMYData.DHillum                     Avg. horizontal diffuse illuminance recv'd during the 60 minutes prior to timestamp, lx
    TMYData.DHillumSource               See [1], Table 1-4
    TMYData.DHillumUncertainty          Uncertainty based on random and bias error estimates                        see [2]
    TMYData.Zenithlum                   Avg. luminance at the sky's zenith during the 60 minutes prior to timestamp, cd/m^2
    TMYData.ZenithlumSource             See [1], Table 1-4
    TMYData.ZenithlumUncertainty        Uncertainty based on random and bias error estimates                        see [1] section 2.10
    TMYData.TotCld                      Amount of sky dome covered by clouds or obscuring phenonema at time stamp, tenths of sky
    TMYData.TotCldSource                See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.TotCldUnertainty            See [1], Table 1-6
    TMYData.OpqCld                      Amount of sky dome covered by clouds or obscuring phenonema that prevent observing the sky at time stamp, tenths of sky
    TMYData.OpqCldSource                See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.OpqCldUncertainty           See [1], Table 1-6
    TMYData.DryBulb                     Dry bulb temperature at the time indicated, deg C
    TMYData.DryBulbSource               See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.DryBulbUncertainty          See [1], Table 1-6
    TMYData.DewPoint                    Dew-point temperature at the time indicated, deg C
    TMYData.DewPointSource              See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.DewPointUncertainty         See [1], Table 1-6
    TMYData.RHum                        Relatitudeive humidity at the time indicated, percent
    TMYData.RHumSource                  See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.RHumUncertainty             See [1], Table 1-6
    TMYData.Pressure                    Station pressure at the time indicated, 1 mbar
    TMYData.PressureSource              See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.PressureUncertainty         See [1], Table 1-6
    TMYData.Wdir                        Wind direction at time indicated, degrees from north (360 = north; 0 = undefined,calm) 
    TMYData.WdirSource                  See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.WdirUncertainty             See [1], Table 1-6
    TMYData.Wspd                        Wind speed at the time indicated, meter/second
    TMYData.WspdSource                  See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.WspdUncertainty             See [1], Table 1-6
    TMYData.Hvis                        Distance to discernable remote objects at time indicated (7777=unlimited), meter
    TMYData.HvisSource                  See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.HvisUncertainty             See [1], Table 1-6
    TMYData.CeilHgt                     Height of cloud base above local terrain (7777=unlimited), meter
    TMYData.CeilHgtSource               See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.CeilHgtUncertainty          See [1], Table 1-6
    TMYData.Pwat                        Total precipitable water contained in a column of unit cross section from earth to top of atmosphere, cm
    TMYData.PwatSource                  See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.PwatUncertainty             See [1], Table 1-6
    TMYData.AOD                         The broadband aerosol optical depth per unit of air mass due to extinction by aerosol component of atmosphere, unitless
    TMYData.AODSource                   See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.AODUncertainty              See [1], Table 1-6
    TMYData.Alb                         The ratio of reflected solar irradiance to global horizontal irradiance, unitless
    TMYData.AlbSource                   See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.AlbUncertainty              See [1], Table 1-6
    TMYData.Lprecipdepth                The amount of liquid precipitation observed at indicated time for the period indicated in the liquid precipitation quantity field, millimeter
    TMYData.Lprecipquantity             The period of accumulatitudeion for the liquid precipitation depth field, hour
    TMYData.LprecipSource               See [1], Table 1-5, 8760x1 cell array of strings
    TMYData.LprecipUncertainty          See [1], Table 1-6
    =============================       ======================================================================================================================================================  

    References
    ----------

    [1] Wilcox, S and Marion, W. "Users Manual for TMY3 Data Sets".
    NREL/TP-581-43156, Revised May 2008.

    [2] Wilcox, S. (2007). National Solar Radiation Database 1991 2005 
    Update: Users Manual. 472 pp.; NREL Report No. TP-581-41364.

    See also
    ---------

    pvl_makelocationstruct
    pvl_readtmy2

    '''
    
    if filename is None: 					#If no filename is input
        try:
            import Tkinter 
            from tkFileDialog import askopenfilename
            Tkinter.Tk().withdraw() 				 #Start interactive file input
            kwargs = {'FileName': askopenfilename()} 	#read in file name
            var = pvl_tools.Parse(kwargs,Expect) 		#Parse filename 
        except:
            raise Exception('Interactive load failed. Tkinter not supported on this system. Try installing X-Quartz and reloading')
    else:
        var = pvl_tools.Parse(Vars,Expect)		#Parse filename

    head = ['USAF','Name','State','TZ','latitude','longitude','altitude']
    headerfile = open(filename,'r')
    meta = dict(zip(head,headerfile.readline().rstrip('\n').split(","))) #Read in file metadata
    meta['altitude'] = float(meta['altitude'])
    meta['latitude'] = float(meta['latitude'])
    meta['longitude'] = float(meta['longitude'])
    meta['TZ'] = float(meta['TZ'])
    meta['USAF'] = int(meta['USAF'])
    #meta = pvl_tools.repack(meta) #repack dict as a struct

    TMYData = pd.read_csv(filename, header=1,
                          parse_dates={'datetime':['Date (MM/DD/YYYY)','Time (HH:MM)']},
                          date_parser=parsedate, index_col='datetime')

    TMYData = recolumn(TMYData)												#rename to standard column names

    #retreive Timezone for pandas NOTE: TMY3 is currently given in local standard time. Pandas and pytz can only handle DST timezones, and so to keep consistency, the time index will be input as TZ unaware for the moment
    TZ = parsetz(float(meta['TZ']))
    #pdb.set_trace()
    #TMYData.index = TMYData.index.tz_localize(TZ)

    return TMYData, meta



def parsedate(ymd, hour):
    date = ymd + ' ' + hour
    date = pd.date_range(date[0], freq='H', periods=len(date))
    #print ymd+hour
    #yr = '2001'											#Move all data to 2001 datum
    #hour[hour =  = '24:00'] = '0:00'							#Set to pandas 24hr clock
    #ymd = [l[:-4]+yr+' ' for l in ymd]					#remove year from TMY data
    #date = ymd+hour
    #date = pd.to_datetime(date)							#convert to pandas datetime
    #date[date.hour =  = 0] = date[date.hour =  = 0]+pd.tseries.offsets.DateOffset(day = 1)
    return date



def parsetz(UTC):
    #currently not used, need to make these daylight savings unaware
    TZinfo = {-5:'EST',
              -6:'CST',
              -7:'MST',
              -8:'PST',
              -9:'AKST',
              -10:'HAST'}
    return TZinfo[UTC]



def recolumn(TMY3):
    TMY3.columns = ('ETR','ETRN','GHI','GHISource','GHIUncertainty',
    'DNI','DNISource','DNIUncertainty','DHI','DHISource','DHIUncertainty',
    'GHillum','GHillumSource','GHillumUncertainty','DNillum','DNillumSource',
    'DNillumUncertainty','DHillum','DHillumSource','DHillumUncertainty',
    'Zenithlum','ZenithlumSource','ZenithlumUncertainty','TotCld','TotCldSource',
    'TotCldUnertainty','OpqCld','OpqCldSource','OpqCldUncertainty','DryBulb',
    'DryBulbSource','DryBulbUncertainty','DewPoint','DewPointSource',
    'DewPointUncertainty','RHum','RHumSource','RHumUncertainty','Pressure',
    'PressureSource','PressureUncertainty','Wdir','WdirSource','WdirUncertainty',
    'Wspd','WspdSource','WspdUncertainty','Hvis','HvisSource','HvisUncertainty',
    'CeilHgt','CeilHgtSource','CeilHgtUncertainty','Pwat','PwatSource',
    'PwatUncertainty','AOD','AODSource','AODUncertainty','Alb','AlbSource',
    'AlbUncertainty','Lprecipdepth','Lprecipquantity','LprecipSource',
    'LprecipUncertainty') 

    return TMY3
