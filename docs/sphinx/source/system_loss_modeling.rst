
Modeling multiple systems with system losses
============================================

This tutorial provides an outline of techniques to model single inverter
systems with multiple strings.

A number of complexities are included:

-  Converting a CSV to a dictionary of system losses
-  Modeling multiple systems at a time
-  Modeling inverters with temperature derating applied to maximum power
   output

Author:

::

    Jessica Forbess (@jforbess), Sunshine Analytics, September 2015

.. code:: python

    # built-in imports
    import sys
    import datetime
    import os.path
    import inspect
    
    # add-on imports
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    %matplotlib inline
    try:
        import seaborn as sns
        sns.set(rc={"figure.figsize": (12, 6)})
        sns.set_color_codes()
    except ImportError:
        print('We suggest you install seaborn using conda or pip and rerun this cell')
    
    import numpy as np
    
    # pvlib imports
    from pvlib.location import Location
    import pvlib.solarposition
    import pvlib.clearsky
    import pvlib.tmy
    
    import logging
    logging.getLogger('pvlib').setLevel(logging.WARNING) # WARNING to avoid clearsky output, INFO or DEBUG for more
    
    

Import site configuration data
==============================

The dictionary holding the configuration data below is a beginning point
to discuss a structure for system losses. It will likely be improved
upon and standardized in the future.

.. code:: python

    # Replaces all punctuation in equipment (module/inverter) names with underscores to match Sandia databases
    def clean_punc(equip):
        equip = equip.replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace('/', '_')
        equip = equip.replace(')', '_').replace('[', '_').replace(']', '_').replace(':', '_').replace('+', '_')
        equip = equip.replace('"', '_')
        return equip
    
    
    # Find the absolute file path to your pvlib installation
    pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))
    
    # absolute path to a data file
    datapath = os.path.join(pvlib_abspath, 'data', 'site_config.csv')
    
    # Import site config data, skipping any row starting with #
    # In this file, each line represents a site with one inverter
    site_config = pd.read_csv(datapath, comment='#')
    # Load list of site names to iterate over
    site_names = list(set(site_config['Plant Name'].tolist()))
    site_names.sort()
    
    # initialize config dictionary
    meta = {}
    
    # Load site config data from csv file into meta dict variable
    for n in site_names:
        # Solar position code needs a Location object
        m = pvlib.location.Location(site_config[site_config['Plant Name'] == n].iloc[0]['Latitude'], 
                                    site_config[site_config['Plant Name'] == n].iloc[0]['Longitude'], 
                                    site_config[site_config['Plant Name'] == n].iloc[0]['Timezone'], 
                                    float(site_config[site_config['Plant Name'] == n].iloc[0]['Elev (meters)']), 
                                    site_config[site_config['Plant Name'] == n].iloc[0]['Plant Name']) ## One common name for site
        # Systemdef needs a dict instead of a Location object
        s = pvlib.pvsystem.systemdef({'latitude': site_config[site_config['Plant Name'] == n].iloc[0]['Latitude'],
                                      'longitude': site_config[site_config['Plant Name'] == n].iloc[0]['Longitude'],
                                      'altitude': float(site_config[site_config['Plant Name'] == n].iloc[0]['Elev (meters)']),
                                      'TZ': site_config[site_config['Plant Name'] == n].iloc[0]['Timezone'],
                                      'Name': n}, ## Specific name for subarray
                                     int(site_config[site_config['Plant Name'] == n].iloc[0]['surface_tilt']),
                                     int(site_config[site_config['Plant Name'] == n].iloc[0]['surface_azimuth']),
                                     float(site_config[site_config['Plant Name'] == n].iloc[0]['albedo']),
                                     int(site_config[site_config['Plant Name'] == n].iloc[0]['series_modules']),
                                     float(site_config[site_config['Plant Name'] == n].iloc[0]['parallel_modules']))
        # Other equipment information that isn't stored in systemdef
        e = {'module_name': clean_punc(site_config[site_config['Plant Name'] == n].iloc[0]['module_name']), 
             'module_count': int(site_config[site_config['Plant Name'] == n].iloc[0]['module_count']),
             'module_wattage': float(site_config[site_config['Plant Name'] == n].iloc[0]['module_wattage']),
             'inverter_name': clean_punc(site_config[site_config['Plant Name'] == n].iloc[0]['inverter_name']), 
             'inverter_count': int(site_config[site_config['Plant Name'] == n].iloc[0]['inverter_count']), 
             'inverter_pmax': int(site_config[site_config['Plant Name'] == n].iloc[0]['inv_pmax']),
             'inverter_pnom': int(site_config[site_config['Plant Name'] == n].iloc[0]['inv_pnom']*1000), # in kW in config file
             'inverter_pmax_default': int(site_config[site_config['Plant Name'] == n].iloc[0]['inv_pmax_default']*1000), # in kW in config file
             'inverter_pmax_temp': int(site_config[site_config['Plant Name'] == n].iloc[0]['inv_pmax_temp']),
             'inverter_pnom_temp': int(site_config[site_config['Plant Name'] == n].iloc[0]['inv_pnom_temp']),
             'row_pitch': float(site_config[site_config['Plant Name'] == n].iloc[0]['pitch']),
             'collector': float(site_config[site_config['Plant Name'] == n].iloc[0]['collector']),}
        # System Losses
        q = {'soiling': float(site_config[site_config['Plant Name'] == n].iloc[0]['Soiling']),
             'dcohmic': float(site_config[site_config['Plant Name'] == n].iloc[0]['DCOhmic']),
             'mqf': float(site_config[site_config['Plant Name'] == n].iloc[0]['MQF']),
             'lid': float(site_config[site_config['Plant Name'] == n].iloc[0]['LID']),
             'mismatch': float(site_config[site_config['Plant Name'] == n].iloc[0]['Mismatch']),
             'acohmic': float(site_config[site_config['Plant Name'] == n].iloc[0]['ACOhmic']),
             'mvload': float(site_config[site_config['Plant Name'] == n].iloc[0]['MVLoad']),
            }
        meta[n] = [m,s,e,q]
    
    # Load Module and Inverter dbs
    ## Direct from Sandia, or can download locally and provide filename -- much faster
    moddb = pvlib.pvsystem.retrieve_sam(name='CECMod')
    invdb = pvlib.pvsystem.retrieve_sam(name='CECInverter')
    
    # Confirm sites loaded, and Lat/Long
    for n in site_names:
        print(meta[n][1]['name'],": ",meta[n][1]['latitude'], meta[n][1]['longitude'])


.. parsed-literal::

    Site 1: 500kWac :  38.6 -121.5
    Site 2: 880kWac :  38.6 -121.5
    Site 3: 660kWac :  38.6 -121.5
    

Define key system model method
==============================

This method takes system configuration information and applies it to the
environmental data to create modeled power for each time sample. It can
be used with standard TMY3 hour data, or finer grain environmental data,
if available.

Note that the meta defined in this method is a subset of the meta
defined in the overarching script.

.. code:: python

    def subarray(meta, d, module_type, inverter_type, 
                 inv_pmax, inv_pnom, inv_pmax_default, inv_pmax_temp, inv_pnom_temp,
                 pitch, col, losses):
        df = d
        # Find Isc at STC for DC ohmic loss estimate
        IL,I0,Rs,Rsh,nNsVth=pvlib.pvsystem.calcparams_desoto(poa_global=1000,
                                                             temp_cell=25,
                                                             alpha_isc=.003,
                                                             module_parameters=module_type,
                                                             EgRef=1.121,
                                                             dEgdT= -0.0002677)
        DFOut = pvlib.pvsystem.singlediode(module_type, IL, I0, Rs, Rsh, nNsVth)
        stc_i_sc = DFOut['i_sc']
        # Calculate Isc for entire array by multiplying Isc by number of parallel modules (actually strings)
        # Used as basis to scale DC and AC ohmic losses
        stc_array_i_sc = stc_i_sc * meta['parallel_modules']
    
        # Calculate IAM
        df['IAMa'] = pvlib.pvsystem.ashraeiam(b = 0.05, aoi = df['AOI'])
        df['IAMa'].fillna(0, inplace=True)
    
        # Soiling
        df['Soiling'] = 1 - losses['soiling']
    
        # Calculate module IV params for POA and temps experienced
        IL,I0,Rs,Rsh,nNsVth=pvlib.pvsystem.calcparams_desoto(poa_global=df['poa_global'],
                                                             temp_cell=df['tcell'],
                                                             alpha_isc=.003,
                                                             module_parameters=module_type,
                                                             EgRef=1.121,
                                                             dEgdT= -0.0002677)
        DFOut = pvlib.pvsystem.singlediode(module_type, IL, I0, Rs, Rsh, nNsVth)
        df['sd_i_mp'] = DFOut['i_mp']
        df['sd_v_oc'] = DFOut['v_oc']
        df['sd_v_mp'] = DFOut['v_mp']
        df['sd_p_mp'] = DFOut['p_mp']
        df['sd_i_x'] = DFOut['i_x']
        df['sd_i_xx'] = DFOut['i_xx']
    
    
        # Module Quality
        df['MQF'] = 1 - losses['mqf']
        # LID
        df['LID'] = 1 - losses['lid']
        # Mismatch
        df['Mismatch'] = 1 - losses['mismatch']
        # DC kW
        df['sd_string_v_mp'] = meta['series_modules'] * df.sd_v_mp
        df['sd_array_i_mp'] = meta['parallel_modules'] * df.sd_i_mp
        # DC ohmic
        df['DCohmic'] = 1 - losses['dcohmic'] * pow(df.sd_array_i_mp / stc_array_i_sc, 2)
        df['sd_array_p_mp'] = df.sd_string_v_mp * df.sd_array_i_mp * df.DCohmic * df.MQF * df.LID * df.Mismatch
        df['sd_array_kp_mp'] = df.sd_array_p_mp / 1000
    
        # Temperature Derate of Inverter
        # Approximating warmer temps input to inverter
        df['Tinv'] = df['DryBulb'] + 4 
        ## Model doesn't include sharp drop limit over 50C
        # Calculate maximum clipping level based on temperature. Clip level is never lower than nominal inverter level (pmax).
        def clip_level(tamb):
            return min(inv_pmax,
                       inv_pmax_default - max(tamb - inv_pmax_temp,0) * (inv_pmax_default - inv_pnom) / (inv_pnom_temp - inv_pmax_temp))/1000
    
        df['ACPowerClipLevel'] = df['Tinv'].apply(clip_level)
        # AC kW
        df['ACPower'] = pvlib.pvsystem.snlinverter(v_dc = df.sd_string_v_mp,
                                                   p_dc = df.sd_array_p_mp,
                                                   inverter = inverter_type) / 1000
        # Clip any ACPower values that exceed temperature bounded maximum
        df['ACPowerClipped'] = pd.DataFrame([df['ACPower'], df['ACPowerClipLevel']]).min()
    
        # AC ohmic
        df['ACohmic'] = 1 - losses['acohmic'] * pow(df.sd_array_i_mp / stc_array_i_sc, 2) # Scaled by DC current as rough approximation
    
        # MV Transformer
        df['MVLoad'] = 1 - losses['mvload']
    
        df['ArrayPower'] = df.ACPowerClipped * df.ACohmic * df.MVLoad
    

Load environmental data
=======================

Data is a dictionary of DataFrames, with each site\_name as a key. Plane
of Array irradiance and module temperatures are calculated here as well.

.. code:: python

    # Initialize the dataframe dictionary
    Data = {}
    
    # Find the absolute file path to your pvlib installation
    pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))
    
    # absolute path to a data file
    datapath = os.path.join(pvlib_abspath, 'data', '724839TYA.csv')
    
    # read tmy data with year values coerced to a single year
    tmy_data, meta_tmy = pvlib.tmy.readtmy3(datapath, coerce_year=2015)
    tmy_data.index.name = 'Time'
    
    # TMY data seems to be given as hourly data with time stamp at the end
    # shift the index 30 Minutes back for calculation of sun positions
    tmy_data = tmy_data.shift(freq='-30Min')
    
    for n in site_names:
        print('System Name: ' + n + '\n')
        # Initialize dataset with environmental data
        Data[n] = tmy_data
        # Generate clear sky data for site to compare to POA sensors
        # Calculate solar position using Ephemeris Calculations
        solpos = pvlib.solarposition.get_solarposition(Data[n].index, meta[n][0], method='pyephem')
        Data[n] = pd.concat([Data[n], solpos], axis=1)
    
        # Calculate extraterrestrial radiation and air mass and AOI
        Data[n]['HExtra']=pvlib.irradiance.extraradiation(Data[n].index)
        Data[n]['AM']=pvlib.atmosphere.relativeairmass(zenith=Data[n].apparent_zenith)
        Data[n]['AOI'] = pvlib.irradiance.aoi(meta[n][1]['surface_tilt'], 
                                              meta[n][1]['surface_azimuth'], 
                                              Data[n]['apparent_zenith'], 
                                              Data[n]['azimuth'])
        # Calculate GHI clear sky
        csky = pvlib.clearsky.ineichen(Data[n].index, meta[n][0])
        csky.rename(columns=lambda x: 'cs_'+x, inplace=True)
        Data[n] = pd.concat([Data[n], csky], axis=1)
        # Generate POA clear sky
        cs_poa = pvlib.irradiance.total_irrad(surface_tilt = meta[n][1]['surface_tilt'],
                                              surface_azimuth = meta[n][1]['surface_azimuth'],
                                              solar_zenith = Data[n].apparent_zenith,
                                              solar_azimuth = Data[n].apparent_azimuth,
                                              albedo = meta[n][1]['albedo'],
                                              ghi = Data[n].cs_ghi,
                                              dhi = Data[n].cs_dhi,
                                              dni = Data[n].cs_dni,
                                              dni_extra = Data[n].HExtra,
                                              airmass=Data[n].AM, 
                                              model='perez')
        cs_poa.rename(columns=lambda x: 'cs_poa_' + x, inplace=True)
        Data[n] = pd.concat([Data[n], cs_poa], axis=1)       
        
        # Generate POA from GHI data
        poa_sky_diffuse = pvlib.irradiance.haydavies(meta[n][1]['surface_tilt'], 
                                                     meta[n][1]['surface_azimuth'],
                                                     Data[n]['DHI'], Data[n]['DNI'], Data[n]['HExtra'],
                                                     Data[n]['apparent_zenith'], Data[n]['azimuth'])
        poa_ground_diffuse = pvlib.irradiance.grounddiffuse(meta[n][1]['surface_tilt'], 
                                                            Data[n]['GHI'], 
                                                            albedo=meta[n][1]['albedo'])
        poa_total = pvlib.irradiance.globalinplane(Data[n]['AOI'], Data[n]['DNI'], 
                                                   poa_sky_diffuse, poa_ground_diffuse)
        Data[n] = pd.concat([Data[n], poa_total], axis=1)    
    
        # Calculate model cell temperature
        tcells = pvlib.pvsystem.sapm_celltemp(irrad = Data[n]['poa_global'],
                                              wind = Data[n]['Wspd'],
                                              temp = Data[n]['DryBulb'],
                                              model = 'Open_rack_cell_polymerback')
        Data[n]['tcell'] = tcells['temp_cell']
        Data[n]['tmodule'] = tcells['temp_module']
                 
            
    


.. parsed-literal::

    System Name: Site 1: 500kWac
    
    System Name: Site 2: 880kWac
    
    System Name: Site 3: 660kWac
    
    

Run performance simulation for each array
-----------------------------------------

This method calls the modeling method, and compiles a smaller DataFrame
of all the sites for graphing.

Note that because Data[n] is a dictionary, it is modified by the
modeling method, and doesn't need to be explicitly returned.

.. code:: python

    # Initialize DataFrame for graphing
    graph_data = pd.DataFrame()
    
    for n in site_names:
        print('System Name: ' + n)
    
        # Load equipment coefficients
        module = moddb[meta[n][2]['module_name']]
        inverter = invdb[meta[n][2]['inverter_name']]
        # Adjust max inverter output to match field settings
        inverter.Paco = meta[n][2]['inverter_pmax']
        
        # Call modeling method 
        subarray(meta[n][1], Data[n], module, inverter, 
                 meta[n][2]['inverter_pmax'], meta[n][2]['inverter_pnom'], meta[n][2]['inverter_pmax_default'],
                 meta[n][2]['inverter_pmax_temp'], meta[n][2]['inverter_pnom_temp'],
                 meta[n][2]['row_pitch'], meta[n][2]['collector'], meta[n][3])     
    
        # Calculate kWh for each time sample
        Data[n]['ModelEnergy'] = Data[n]['ArrayPower'] # hour data samples only -- e.g., divide by 4 for 15 minute samples
        Data[n]['Name'] = n
        
        # Load key data points into a smaller data frame with all sites   
        graph_data_n = pd.concat([Data[n]['Name'],Data[n]['poa_global'],Data[n]['ModelEnergy']],axis=1)
        graph_data = pd.concat([graph_data,graph_data_n],axis=0)
    
    
        
        
       


.. parsed-literal::

    System Name: Site 1: 500kWac
    System Name: Site 2: 880kWac
    System Name: Site 3: 660kWac
    

Plot plane of array irradiance vs output energy
===============================================

.. code:: python

    # Note that linear regression fit includes inverter clipping data as well as linear region
    # Usually the clipping region is removed from a linear regression fit of system performance. 
    sns.lmplot(x="poa_global", y="ModelEnergy", col="Name", hue="Name", data=graph_data,
                ci=None, palette="muted", size=4,
               scatter_kws={"s": 50, "alpha": 1})
    
    
    




.. parsed-literal::

    <seaborn.axisgrid.FacetGrid at 0x1160ec4e0>




.. image:: system_loss_modeling_files%5Csystem_loss_modeling_11_1.png


