"""
Stub documentation for the module.
"""

from pvlib import atmosphere

class ModelChain(object):
    """
    A class that represents all of the modeling steps necessary for
    calculating power or energy for a PV system at a given location.

    Consider an abstract base class.

    Parameters
    ----------
    system : PVSystem
        The connected set of modules, inverters, etc.

    location : location
        The physical location at which to evaluate the model.

    times : DatetimeIndex
        Times at which to evaluate the model.

    orientation_strategy : None or str
        The strategy for aligning the modules.
        If not None, sets the ``surface_azimuth`` and ``surface_tilt``
        properties of the ``system``.

    clearsky_model : str
        Passed to location.get_clearsky.

    transposition_model : str
        Passed to system.get_irradiance.

    solar_position_method : str
        Passed to location.get_solarposition.

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.

    See also
    --------
    location.Location
    pvsystem.PVSystem
    """

    def __init__(self, system, location,
                 orientation_strategy='south_at_latitude_tilt',
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 **kwargs):

        self.system = system
        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model
        
        # calls setter
        self.orientation_strategy = orientation_strategy
    
    @property
    def orientation_strategy(self):
        return self._orientation_strategy
    
    
    @orientation_strategy.setter
    def orientation_strategy(self, strategy):
        if strategy is None or strategy == 'None':
            pass
        elif strategy == 'south_at_latitude_tilt':
            self.system.surface_azimuth = 180
            self.system.surface_tilt = self.location.latitude
        elif strategy == 'flat':
            self.system.surface_azimuth = 0
            self.system.surface_tilt = 0
        else:
            raise ValueError('invalid orientation strategy. strategy must ' +
                             'be one of south_at_latitude, flat,')

        self._orientation_strategy = strategy
        

    def run_model(self, times=None, irradiance=None, weather=None):
        """
        Run the model.
        
        Parameters
        ----------
        times : None or DatetimeIndex
        irradiance : None or DataFrame
            If None, calculates clear sky data.
            Columns must be 'dni', 'ghi', 'dhi'
        weather : None or DataFrame
            If None, assumes air temperature is 20 C and
            wind speed is 0 m/s.
            Columns must be 'wind_speed', 'temp_air'.

        Returns
        -------
        output : DataFrame
            Some combination of AC power, DC power, POA irrad, etc.
        """
        solar_position = self.location.get_solarposition(times)
        
        if irradiance is None:
            irradiance = self.location.get_clearsky(solar_position.index,
                                                    self.clearsky_model)

        total_irrad = self.system.get_irradiance(solar_position['apparent_zenith'],
                                                 solar_position['azimuth'],
                                                 irradiance['dni'],
                                                 irradiance['ghi'],
                                                 irradiance['dhi'],
                                                 model=self.transposition_model)
                                                
        if weather is None:
            weather = {'wind_speed': 0, 'temp_air': 20}

        temps = self.system.sapm_celltemp(total_irrad['poa_global'],
                                          weather['wind_speed'],
                                          weather['temp_air'])
        
        aoi = self.system.get_aoi(solar_position['apparent_zenith'],
                                  solar_position['azimuth'])

        am_rel = atmosphere.relativeairmass(solar_position['apparent_zenith'],
                                            self.airmass_model)
        am_abs = self.location.get_absoluteairmass(am_rel)

        dc = self.system.sapm(total_irrad['poa_direct'],
                              total_irrad['poa_diffuse'],
                              temps['temp_cell'],
                              am_abs, aoi)

        ac = self.system.snlinverter(dc['v_mp'], dc['p_mp'])

        return dc, ac
    

    def model_system(self):
        """
        Model the system?
        
        I'm just copy/pasting example code...
        
        Returns
        -------
        ???
        """
        
        final_output = self.run_model()
        input = self.prettify_input()
        modeling_steps = self.get_modeling_steps()



class MoreSpecificModelChain(ModelChain):
    """
    Something more specific.
    """
    def __init__(self, *args, **kwargs):
        super(MoreSpecificModelChain, self).__init__(**kwargs)
    
    def run_model(self):
        # overrides the parent ModelChain method
        pass