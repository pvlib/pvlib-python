'''
PVL_READTMY2 Read a TMY2 file in to a DataFrame

Syntax
  TMYData, meta = pvl_readtmy2()
  TMYData, meta = pvl_readtmy2(FileName)

Description
  Read a TMY2 file and make a DataFrame of the data. Note that values
  contained in the DataFrame are unchanged from the TMY2 file (i.e. units 
  are retained). Time/Date and Location data imported from the TMY2 file
  have been modified to a "friendlier" form conforming to modern
  conventions (e.g. N latitude is postive, E longitude is positive, the
  "24th" hour of any day is technically the "0th" hour of the next day).
  In the case of any discrepencies between this documentation and the 
  TMY2 User's Manual ([1]), the TMY2 User's Manual takes precedence.

  If a FileName is not provided, the user will be prompted to browse to
  an appropriate TMY2 file.

  Input
    FileName - an optional argument which allows the user to select which
    TMY2 format file should be read. A file path may also be necessary if
    the desired TMY2 file is not in the working path. If FileName
    is not provided, the user will be prompted to browse to an
    appropriate TMY2 file.

  Output
  TMYData- A dataframe, is provided with the following components.  Note
    that for more detailed descriptions of each component, please consult
    the TMY2 User's Manual ([1]), especially tables 3-1 through 3-6, and 
    Appendix B. 

   meta- A struct containing the metadata from the TMY2 file.

      meta.SiteID - Site identifier code (WBAN number), scalar unsigned integer
      meta.StationName - Station name, 1x1 cell string
      meta.StationState - Station state 2 letter designator, 1x1 cell string
      meta.SiteTimeZone - Hours from Greenwich, scalar double
      meta.latitude - Latitude in decimal degrees, scalar double
      meta.longitude - Longitude in decimal degrees, scalar double
      meta.SiteElevation - Site elevation in meters, scalar double
      TMYData.index- Pandas timeseries object containing timestamps
      TMYData.year - 
      TMYData.month -
      TMYData.day -
      TMYData.hour-
      TMYData.ETR - Extraterrestrial horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
      TMYData.ETRN - Extraterrestrial normal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
      TMYData.GHI - Direct and diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
      TMYData.GHISource - See [1], Table 3-3
      TMYData.GHIUncertainty - See [1], Table 3-4
      TMYData.DNI - Amount of direct normal radiation (modeled) recv'd during 60 mintues prior to timestamp, Wh/m^2
      TMYData.DNISource - See [1], Table 3-3
      TMYData.DNIUncertainty - See [1], Table 3-4
      TMYData.DHI - Amount of diffuse horizontal radiation recv'd during 60 minutes prior to timestamp, Wh/m^2
      TMYData.DHISource - See [1], Table 3-3
      TMYData.DHIUncertainty - See [1], Table 3-4
      TMYData.GHillum - Avg. total horizontal illuminance recv'd during
        the 60 minutes prior to timestamp, units of 100 lux (e.g. value
        of 50 = 5000 lux)
      TMYData.GHillumSource - See [1], Table 3-3
      TMYData.GHillumUncertainty - See [1], Table 3-4
      TMYData.DNillum - Avg. direct normal illuminance recv'd during the
        60 minutes prior to timestamp, units of 100 lux
      TMYData.DNillumSource - See [1], Table 3-3
      TMYData.DNillumUncertainty - See [1], Table 3-4
      TMYData.DHillum - Avg. horizontal diffuse illuminance recv'd during
        the 60 minutes prior to timestamp, units of 100 lux
      TMYData.DHillumSource - See [1], Table 3-3
      TMYData.DHillumUncertainty - See [1], Table 3-4
      TMYData.Zenithlum - Avg. luminance at the sky's zenith during the
        60 minutes prior to timestamp, units of 10 Cd/m^2 (e.g. value of
        700 = 7,000 Cd/m^2)
      TMYData.ZenithlumSource - See [1], Table 3-3
      TMYData.ZenithlumUncertainty - See [1], Table 3-4
      TMYData.TotCld - Amount of sky dome covered by clouds or obscuring phenonema at time stamp, tenths of sky
      TMYData.TotCldSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.TotCldUnertainty - See [1], Table 3-6
      TMYData.OpqCld - Amount of sky dome covered by clouds or obscuring phenonema that prevent observing the 
        sky at time stamp, tenths of sky
      TMYData.OpqCldSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.OpqCldUncertainty - See [1], Table 3-6
      TMYData.DryBulb - Dry bulb temperature at the time indicated, in
        tenths of degree C (e.g. 352 = 35.2 C).
      TMYData.DryBulbSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.DryBulbUncertainty - See [1], Table 3-6
      TMYData.DewPoint - Dew-point temperature at the time indicated, in
        tenths of degree C (e.g. 76 = 7.6 C).
      TMYData.DewPointSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.DewPointUncertainty - See [1], Table 3-6
      TMYData.RHum - Relative humidity at the time indicated, percent
      TMYData.RHumSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.RHumUncertainty - See [1], Table 3-6
      TMYData.Pressure - Station pressure at the time indicated, 1 mbar
      TMYData.PressureSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.PressureUncertainty - See [1], Table 3-6
      TMYData.Wdir - Wind direction at time indicated, degrees from east
        of north (360 = 0 = north; 90 = East; 0 = undefined,calm) 
      TMYData.WdirSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.WdirUncertainty - See [1], Table 3-6
      TMYData.Wspd - Wind speed at the time indicated, in tenths of
        meters/second (e.g. 212 = 21.2 m/s)
      TMYData.WspdSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.WspdUncertainty - See [1], Table 3-6
      TMYData.Hvis - Distance to discernable remote objects at time 
        indicated (7777=unlimited, 9999=missing data), in tenths of
        kilometers (e.g. 341 = 34.1 km).
      TMYData.HvisSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.HvisUncertainty - See [1], Table 3-6
      TMYData.CeilHgt - Height of cloud base above local terrain
        (7777=unlimited, 88888=cirroform, 99999=missing data), in meters
      TMYData.CeilHgtSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.CeilHgtUncertainty - See [1], Table 3-6
      TMYData.Pwat - Total precipitable water contained in a column of unit cross section from 
        Earth to top of atmosphere, in millimeters
      TMYData.PwatSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.PwatUncertainty - See [1], Table 3-6
      TMYData.AOD - The broadband aerosol optical depth (broadband
        turbidity) in thousandths on the day indicated (e.g. 114 = 0.114)
      TMYData.AODSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.AODUncertainty - See [1], Table 3-6
      TMYData.SnowDepth - Snow depth in centimeters on the day indicated,
        (999 = missing data).
      TMYData.SnowDepthSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.SnowDepthUncertainty - See [1], Table 3-6
      TMYData.LastSnowfall - Number of days since last snowfall (maximum 
        value of 88, where 88 = 88 or greater days; 99 = missing data)
      TMYData.LastSnowfallSource - See [1], Table 3-5, 8760x1 cell array of strings
      TMYData.LastSnowfallUncertainty - See [1], Table 3-6
      TMYData.PresentWeather - See [1], Appendix B, an 8760x1 cell array
        of strings. Each string contains 10 numeric values. The string
        can be parsed to determine each of 10 observed weather metrics.

Reference
  [1] Marion, W and Urban, K. "Wilcox, S and Marion, W. "User's Manual
    for TMY2s". NREL 1995.

See also
  DATEVEC  PVL_MAKELOCATIONSTRUCT  PVL_MAKETIMESTRUCT  PVL_READTMY3
'''
import pandas as pd
import numpy as np
import re
import datetime
import Tkinter 
import pvl_tools
import pdb
from tkFileDialog import askopenfilename

def pvl_readtmy2(FileName):
	Vars=locals()
	Expect={'FileName':('open')}
	var=[]
	if len(kwargs.keys())==0:
		Tkinter.Tk().withdraw() 
		kwargs={'FileName': askopenfilename()} 
		var=pvl_tools.Parse(kwargs,Expect)
	else:
		var=pvl_tools.Parse(Vars,Expect)

	string='%2d%2d%2d%2d%4d%4d%4d%1s%1d%4d%1s%1d%4d%1s%1d%4d%1s%1d%4d%1s%1d%4d%1s%1d%4d%1s%1d%2d%1s%1d%2d%1s%1d%4d%1s%1d%4d%1s%1d%3d%1s%1d%4d%1s%1d%3d%1s%1d%3d%1s%1d%4d%1s%1d%5d%1s%1d%10d%3d%1s%1d%3d%1s%1d%3d%1s%1d%2d%1s%1d'
	columns='year,month,day,hour,ETR,ETRN,GHI,GHISource,GHIUncertainty,DNI,DNISource,DNIUncertainty,DHI,DHISource,DHIUncertainty,GHillum,GHillumSource,GHillumUncertainty,DNillum,DNillumSource,DNillumUncertainty,DHillum,DHillumSource,DHillumUncertainty,Zenithlum,ZenithlumSource,ZenithlumUncertainty,TotCld,TotCldSource,TotCldUnertainty,OpqCld,OpqCldSource,OpqCldUncertainty,DryBulb,DryBulbSource,DryBulbUncertainty,DewPoint,DewPointSource,DewPointUncertainty,RHum,RHumSource,RHumUncertainty,Pressure,PressureSource,PressureUncertainty,Wdir,WdirSource,WdirUncertainty,Wspd,WspdSource,WspdUncertainty,Hvis,HvisSource,HvisUncertainty,CeilHgt,CeilHgtSource,CeilHgtUncertainty,PresentWeather,Pwat,PwatSource,PwatUncertainty,AOD,AODSource,AODUncertainty,SnowDepth,SnowDepthSource,SnowDepthUncertainty,LastSnowfall,LastSnowfallSource,LastSnowfallUncertaint'
	hdr_columns='WBAN,City,State,TimeZone,Latitude,Longitude,Elevation'

	TMY2,TMY2_meta=readTMY(string,columns,hdr_columns,var.FileName)	

	return TMY2,TMY2_meta

def parsemeta(columns,line):
	rawmeta=" ".join(line.split()).split(" ") #Remove sduplicated spaces, and read in each element
	meta=rawmeta[:4] #take the first string entries
	longitude=(float(rawmeta[5])+float(rawmeta[6])/60)*(2*(rawmeta[4]=='N')-1)#Convert to decimal notation with S negative
	latitude=(float(rawmeta[8])+float(rawmeta[9])/60)*(2*(rawmeta[7]=='E')-1) #Convert to decimal notation with W negative
	meta.append(longitude)
	meta.append(latitude)
	meta.append(float(rawmeta[10]))	
	return dict(zip(columns.split(','),meta)) #Creates a dictionary of metadata

def readTMY(string,columns,hdr_columns,fname):
	head=1
	date=[]
	with open(fname) as infile:
		fline=0
		for line in infile:
			#Skip the header
			if head!=0:
				meta=parsemeta(hdr_columns,line)
				head-=1
				continue
			#Reset the cursor and array for each line    
			cursor=1
			part=[]
			for marker in string.split('%'):
				#Skip the first line of markers
				if marker=='':
					continue
				
				#Read the next increment from the marker list    
				increment=int(re.findall('\d+',marker)[0])

				#Extract the value from the line in the file
				val=(line[cursor:cursor+increment])
				#increment the cursor by the length of the read value
				cursor=cursor+increment
			 

				# Determine the datatype from the marker string
				if marker[-1]=='d':
					try:
						val=float(val)
					except:
						raise Exception('WARNING: In'+__name__+' Read value is not an integer" '+val+' " ')
				elif marker[-1]=='s':
					try:
						val=str(val)
					except:
						raise Exception('WARNING: In'+__name__+' Read value is not a string" '+val+' " ')
				else: 
					raise Exception('WARNING: In'+__name__+'Improper column DataFrameure " %'+marker+' " ')
					
				part.append(val)
				
			if fline==0:
				axes=[part]
				year=part[0]+1900
				fline=1
			else:
				axes.append(part)
				
			#Create datetime objects from read data
			date.append(datetime.datetime(year=int(year),month=int(part[1]),day=int(part[2]),hour=int(part[3])-1))
	
	TMYData=pd.DataFrame(axes,index=date,columns=columns.split(','))

	return TMYData,meta
			
			
