"""
Stub documentation for the module.
"""

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
        
    orientation_strategy : str
        The strategy for aligning the modules.
    
    clearsky_method : str
        Passed to location.get_clearsky.
    
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

    def __init__(self, system, location, times,
                 orientation_strategy='south_at_latitude',
                 clearsky_method='ineichen',
                 solar_position_method='nrel_numpy',
                 **kwargs):

        self.system = system
        self.location = location
        self.times = times
        self.clearsky_method = clearsky_method
        self.solar_position_method = solar_position_method
        
        self._orientation_strategy = orientation_strategy
    
    @property
    def orientation_strategy(self):
        return self._orientation_strategy
    
    
    @property.setter
    def orientation_strategy(self, strategy):
        if strategy == 'south_at_latitude':
            self.surface_azimuth = 180
            self.surface_tilt = self.location.latitude
        elif strategy == 'flat':
            self.surface_azimuth = 0
            self.surface_tilt = 0
        else:
            raise ValueError('invalid orientation strategy. strategy must ' +
                             'be one of south_at_latitude, flat,')

        self._orientation_strategy = strategy
        

    def run_model(self):
        """
        Run the model.
        
        Returns
        -------
        output : DataFrame
            Column names???
        """
        solar_position = self.location.get_solarposition(self.times)
        clearsky = self.system.get_clearsky(solar_position,
                                            self.clearsky_method)
        final_output = self.system.calculate_final_yield(args)
        return final_output
    

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