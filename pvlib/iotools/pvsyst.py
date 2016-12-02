# -*- coding: utf-8 -*-
"""
Read & Write PVSyst Output
"""
# standard library imports 
import logging

# related third party imports
import numpy as np
import pandas as pd
import pytz

# local application/library specific imports 
from pvlib.location import Location
from pvlib.iotools.iotools import get_loc_latlon, localise_df



#XXX read metadata / header
def read_pvsyst_h_metadata(file_csv, name='pvsyst'):
    if not name:
        name = file_csv.split('.')[0]
    
    ## if file is on local drive
    f = open(file_csv)
    for line in f:
        if "Geographical Site" in line:
            site = line.split(";")[3]
            country = line.split(";")[4]
            continent = line.split(";")[5]
            name = site
        
        meta_project = "Project"
        if meta_project in line:
            project = line.split(";")[1]
        meta_meteo = "Meteo data"
        if meta_meteo in line:
            meteo_data = line.split(";")[1]
        meta_variant = "Simulation variant"
        if  meta_variant in line:
            variant = line.split(";")[1]


    lat = np.nan
    logging.debug('PVSyst output CSV file has not latitue information. \
                   Check the site site file of the PVSyst project.')
    lon = np.nan
    logging.debug('PVSyst output CSV file has not longitude information. \
                   Check the site site file of the PVSyst project.')
    alt = np.nan
    logging.debug('PVSyst output CSV file has not altitude information. \
                   Check the site site file of the PVSyst project.')
    
    tz_raw = ''
    logging.debug('PVSyst output CSV file has not timezone information. \
                   Check the site site file of the PVSyst project.')
    
    location = Location(lat, lon, name=name, altitude=alt,
#                           tz=tz_raw
                           )
    
    #XXX other metadata
    metadata = { 
                meta_project : project,
                meta_meteo : meteo_data,
                meta_variant : variant,
                }
    
    return tz_raw, location, metadata
    
#XXX convert to pvlib conventions
def pvsyst_h_to_pvlib(df_raw, tz_raw, loc, localise=False):
    """Change some properties of the dataframe to be more compliant with pvlib
    
    * localisation
    * column renaming
    * setting dataframe name description according to datasource
    
    """

    if localise:
        # timezone localisations
        df_pvlib = localise_df(df_raw, tz_source_str=tz_raw, 
                               tz_target_str=loc.tz)
    else:
        df_pvlib = df_raw.copy()
    # TODO: adjust column renaming
    # column renaming
    df_pvlib.index.name = 'datetime'
    df_pvlib.rename(columns={'GlobHor': 'ghi', 
                        'T Amb': 'temp_air',
                        'GlobEff': 'g_poa_effective',
                        'EArrMPP': 'pdc, dc',
                        'FTransp': 'transposition_factor',
                        'AngInc': 'surface_tilt',
                        'E_Grid': 'pac, ac',                        
                        }, 
              inplace=True)
    
    # name the dataframe according to data source 
    df_pvlib.df_name = loc.name
  
    return df_pvlib

#XXX read data
def read_pvsyst_hourly(file_csv, output='all', localise=False):
    df_raw = pd.read_csv(file_csv, sep=';', skiprows=[11,12], index_col=0, 
                              parse_dates=True, header=8)
    
    
    if output == 'df_raw':
        res = df_raw
    if output == 'all':
        tz_raw, loc, metadata = read_pvsyst_h_metadata(file_csv)
        loc.name = (loc.name + ' of ' + metadata['Project'] + 
                    ' using meteo data input "' + metadata['Meteo data'] + 
                    '" for simlation variant "' + 
                    metadata['Simulation variant'] + '"')
                    
        df_pvlib = pvsyst_h_to_pvlib(df_raw, tz_raw, loc, localise=localise)
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