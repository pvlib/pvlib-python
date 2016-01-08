'''
The 'forecast' module contains class definitions for 
retreiving forecasted data from UNIDATA Thredd servers.
'''
import datetime
from xml.etree.ElementTree import ParseError
from netCDF4 import num2date
import pandas as pd
import numpy as np

from pvlib.location import Location
from pvlib.tools import localize_to_utc
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import liujordan
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS


class ForecastModel(object):
    '''
    An object for holding forecast model information for use within the 
    pvlib library.

    Simplifies use of siphon library on a THREDDS server.

    Parameters
    ----------
    model_type: string
        UNIDATA category in which the model is located.
    model_name: string
        Name of the UNIDATA forecast model.
    set_type: string
        Model dataset type.

    Attributes
    ----------
    access_url: string
        URL specifying the dataset from data will be retrieved.
    base_tds_url : string
        The top level server address
    catalog_url : string
        The url path of the catalog to parse.
    columns: list
        List of headers used to create the data DataFrame.
    data: pd.DataFrame
        Data returned from the query.
    data_format: string
        Format of the forecast data being requested from UNIDATA.
    dataset: Dataset
        Object containing information used to access forecast data.
    dataframe_variables: list
        Model variables that are present in the data.
    datasets_list: list
        List of all available datasets.
    fm_models: Dataset
        Object containing all available foreast models.
    fm_models_list: list
        List of all available forecast models from UNIDATA.
    latitude: list
        A list of floats containing latitude values.
    location: Location
        A pvlib Location object containing geographic quantities.
    longitude: list
        A list of floats containing longitude values.
    lbox: boolean
        Indicates the use of a location bounding box.
    ncss: NCSS object
        NCSS    model_name: string
        Name of the UNIDATA forecast model.    
    model: Dataset
        A dictionary of Dataset object, whose keys are the name of the
        dataset's name.
    model_url: string
        The url path of the dataset to parse.
    modelvariables: list
        Common variable names that correspond to queryvariables.
    query: NCSS query object
        NCSS object used to complete the forecast data retrival.
    queryvariables: list
        Variables that are used to query the THREDDS Data Server.
    rad_type: dictionary
        Dictionary labeling the method used for calculating radiation values.
    time: datetime
        Time range specified for the NCSS query.
    utctime: DatetimeIndex
        Time range in UTC.
    var_stdnames: dictionary
        Dictionary containing the standard names of the variables in the
        query, where the keys are the common names.
    var_units: dictionary
        Dictionary containing the unites of the variables in the query,
        where the keys are the common names.
    variables: dictionary
        Dictionary that translates model specific variables to 
        common named variables.
    vert_level: float or integer
        Vertical altitude for query data.
    wind_type: string
        Quantity that was used to calculate wind_speed.
    zenith: numpy.array
        Solar zenith angles for the given time range.
    '''

    access_url_key = 'NetcdfSubset'
    catalog_url = 'http://thredds.ucar.edu/thredds/catalog.xml'
    base_tds_url = catalog_url.split('/thredds/')[0]
    data_format = 'netcdf'
    vert_level = 100000
    columns = np.array(['temperature',
                        'wind_speed',
                        'total_clouds',
                        'low_clouds',
                        'mid_clouds',
                        'high_clouds',
                        'dni',
                        'dhi',
                        'ghi', ])

    def __init__(self, model_type, model_name, set_type):
        self.model_type = model_type
        self.model_name = model_name
        self.set_type = set_type
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(self.catalog.catalog_refs[model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))
        
        try:
            model_url = self.fm_models.catalog_refs[model_name].href
        except ParseError:
            raise ParseError(self.model_name + ' model may be unavailable.')

        self.model = TDSCatalog(model_url)
        self.datasets_list = list(self.model.datasets.keys())
        self.set_dataset()


    def set_dataset(self):
        '''
        Retreives the designated dataset, creates NCSS object, and 
        initiates a NCSS query.

        '''
        keys = list(self.model.datasets.keys())
        labels = [item.split()[0].lower() for item in keys]
        if self.set_type == 'best':
            self.dataset = self.model.datasets[keys[labels.index('best')]]
        elif self.set_type == 'latest':
            self.dataset = self.model.datasets[keys[labels.index('latest')]]
        elif self.set_type == 'full':
            self.dataset = self.model.datasets[keys[labels.index('full')]]

        self.access_url = self.dataset.access_urls[self.access_url_key]
        self.ncss = NCSS(self.access_url)
        self.query = self.ncss.query()        

    def set_query_latlon(self):
        '''
        Sets the NCSS query location latitude and longitude.

        '''
        if isinstance(self.longitude, list):
            self.lbox = True
            # west, east, south, north
            self.query.lonlat_box(self.latitude[0], self.latitude[1], 
                                    self.longitude[0], self.longitude[1])
        else:
            self.lbox = False
            self.query.lonlat_point(self.longitude, self.latitude)

    def set_query_time(self):
        '''
        Sets the NCSS query time range.

        as: single or range

        '''        
        if len(self.utctime) == 1:
            self.query.time(pd.to_datetime(self.utctime)[0])
        else:
            self.query.time_range(pd.to_datetime(self.utctime)[0], 
                pd.to_datetime(self.utctime)[-1])
    
    def set_location(self, time):
        '''
        Sets the location for 

        Parameters
        ----------
        time: datetime or DatetimeIndex
            Time range of the query.
        '''
        if isinstance(time, datetime.datetime):
            tzinfo = time.tzinfo
        else:
            tzinfo = time.tz

        if tzinfo is None:
            self.location = Location(self.latitude, self.longitude)
        else:
            self.location = Location(self.latitude, self.longitude, tz=tzinfo)

    def get_query_data(self, latitude, longitude, time, vert_level=None, 
        variables=None):
        '''
        Submits a query to the UNIDATA servers using siphon NCSS and 
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ----------
        latitude: list
            A list of floats containing latitude values.
        longitude: list
            A list of floats containing longitude values.
        time: pd.datetimeindex
            Time range of interest.
        vert_level: float or integer
            Vertical altitude of interest.
        variables: dictionary
            Variables and common names being queried.

        Returns
        -------
        pd.DataFrame
        '''
        if vert_level is not None:
            self.vert_level = vert_level
        if variables is not None:
            self.variables = variables
            self.modelvariables = list(self.variables.keys())
            self.queryvariables = [self.variables[key] for key in \
                self.modelvariables]
            self.columns = self.modelvariables
            self.dataframe_variables = self.modelvariables
        

        self.latitude = latitude
        self.longitude = longitude
        self.set_query_latlon()
        self.set_location(time)

        self.utctime = localize_to_utc(time, self.location)
        self.set_query_time()

        self.query.vertical_level(self.vert_level)
        self.query.variables(*self.queryvariables)
        self.query.accept(self.data_format)
        netcdf_data = self.ncss.get_data(self.query)

        try:
            time_var = 'time'
            self.set_time(netcdf_data.variables[time_var])
        except KeyError:
            time_var = 'time1'
            self.set_time(netcdf_data.variables[time_var])

        self.data = self.netcdf2pandas(netcdf_data)

        self.set_variable_units(netcdf_data)
        self.set_variable_stdnames(netcdf_data)
        if self.__class__.__name__ is 'HRRR':
            self.calc_temperature(netcdf_data)
        self.convert_temperature()
        self.calc_wind(netcdf_data)
        self.calc_radiation(netcdf_data)

        self.data = self.data.tz_convert(self.location.tz)

        netcdf_data.close()        

        return self.data       

    def netcdf2pandas(self, data):
        '''
        Transforms data from netcdf  to pandas DataFrame.

        Currently only supports one-dimensional netcdf data.

        Parameters
        ----------
        data: netcdf
            Data returned from UNIDATA NCSS query.

        Returns
        -------
        pd.DataFrame
        '''
        if not self.lbox:
            ''' one-dimensional data '''
            data_dict = {}
            for var in self.dataframe_variables:
                data_dict[var] = pd.Series(
                    data[self.variables[var]][:].squeeze(), index=self.utctime)
            return pd.DataFrame(data_dict, columns=self.columns)
        else:
            return pd.DataFrame(columns=self.columns, index=self.utctime)

    def set_time(self, time):
        '''
        Converts time data into a pandas date object.

        Parameters
        ----------
        time: netcdf
            Contains time information.

        Returns
        -------
        pandas.DatetimeIndex
        '''
        times = num2date(time[:].squeeze(), time.units)
        self.time = pd.DatetimeIndex(pd.Series(times), tz='UTC')
        self.time = self.time.tz_convert(self.location.tz)
        self.utctime = localize_to_utc(self.time, self.location.tz)

    def set_variable_units(self, data):
        '''
        Extracts variable unit information from netcdf data.

        Parameters
        ----------
        data: netcdf
            Contains queried variable information.

        '''
        self.var_units = {}
        for var in self.variables:
            self.var_units[var] = data[self.variables[var]].units

    def set_variable_stdnames(self, data):
        '''
        Extracts standard names from netcdf data.

        Parameters
        ----------
        data: netcdf
            Contains queried variable information.

        '''
        self.var_stdnames = {}
        for var in self.variables:
            try:
                self.var_stdnames[var] = \
                    data[self.variables[var]].standard_name
            except AttributeError:
                self.var_stdnames[var] = var

    def calc_radiation(self, data, cloud_type='total_clouds'):
        '''
        Determines shortwave radiation values if they are missing from 
        the model data.

        Parameters
        ----------
        data: netcdf
            Query data formatted in netcdf format.
        cloud_type: string
            Type of cloud cover to use for calculating radiation values.
        '''
        self.rad_type = {}
        if not self.lbox and cloud_type in self.modelvariables:           
            cloud_prct = self.data[cloud_type]
            solpos = get_solarposition(self.time, self.location)
            self.zenith = np.array(solpos.zenith.tz_convert('UTC'))
            for rad in ['dni','dhi','ghi']:
                if self.model_name is 'HRRR_ESRL':
                    # HRRR_ESRL is the only model with the 
                    # correct equation of time.
                    if rad in self.modelvariables:
                        self.data[rad] = pd.Series(
                            data[self.variables[rad]][:].squeeze(), 
                            index=self.time)
                        self.rad_type[rad] = 'forecast'
                        self.data[rad].fillna(0, inplace=True)
                else:
                    for rad in ['dni','dhi','ghi']:
                        self.rad_type[rad] = 'liujordan'
                        self.data[rad] = liujordan(self.zenith, cloud_prct)[rad]
                        self.data[rad].fillna(0, inplace=True)

            for var in ['dni', 'dhi', 'ghi']:
                self.data[var].fillna(0, inplace=True)
                self.var_units[var] = '$W m^{-2}$'

    def convert_temperature(self):
        '''
        Converts Kelvin to celsius.

        '''
        if 'Temperature_surface' in self.queryvariables or 'Temperature_isobaric' in self.queryvariables:
            self.data['temperature'] -= 273.15
            self.var_units['temperature'] = 'C'

    def calc_temperature(self, data):
        '''
        Calculates temperature (in degrees C) from isobaric temperature.

        Parameters
        ----------
        data: netcdf
            Query data in netcdf format.
        '''
        P = data['Pressure_surface'][:].squeeze() / 100.0
        Tiso = data['Temperature_isobaric'][:].squeeze()
        Td = data['Dewpoint_temperature_isobaric'][:].squeeze() - 273.15
        e = 6.11 * 10**((7.5 * Td) / (Td + 273.3))
        w = 0.622 * (e / (P - e))

        T = Tiso - ((2.501 * 10.**6) / 1005.7) * w

        self.data['temperature'] = T

    def calc_wind(self, data):
        '''
        Computes wind speed.

        Parameters
        ----------
        data: netcdf
            Query data in netcdf format.
        '''
        if not self.lbox:
            if 'u-component_of_wind_isobaric' in self.queryvariables and \
                'v-component_of_wind_isobaric' in self.queryvariables:
                wind_data = np.sqrt(\
                    data['u-component_of_wind_isobaric'][:].squeeze()**2 +
                    data['v-component_of_wind_isobaric'][:].squeeze()**2)
                self.wind_type = 'component'
            elif 'Wind_speed_gust_surface' in self.queryvariables:
                wind_data = data['Wind_speed_gust_surface'][:].squeeze()
                self.wind_type = 'gust'

            if 'wind_speed' in self.data:
                self.data['wind_speed'] = pd.Series(wind_data, index=self.time)
                self.var_units['wind_speed'] = 'm/s'


class GFS(ForecastModel):
    '''
    Subclass of the ForecastModel class representing GFS forecast model.

    Model data corresponds to 0.25 degree resolution forecasts.

    Parameters
    ----------
    res: string
        Resolution of the model.
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self, res='half', set_type='best'):
        model_type = 'Forecast Model Data'
        if res == 'half':
            model = 'GFS Half Degree Forecast'
        elif res == 'quarter':
            model = 'GFS Quarter Degree Forecast'
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'wind_speed_u':'u-component_of_wind_isobaric',
        'wind_speed_v':'v-component_of_wind_isobaric',
        'total_clouds':
        'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
        'low_clouds':
        'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
        'mid_clouds':
        'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
        'high_clouds':
        'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
        'boundary_clouds':
        'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
        'convect_clouds':'Total_cloud_cover_convective_cloud',
        'ghi':
        'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds',
        'boundary_clouds',
        'convect_clouds',
        'ghi', ]
        super(GFS, self).__init__(model_type, model, set_type)


class HRRR_ESRL(ForecastModel):
    '''
    Subclass of the ForecastModel class representing
    NOAA/GSD/ESRL's HRRR forecast model. This is not an operational product.

    Model data corresponds to NOAA/GSD/ESRL HRRR CONUS 3km resolution
    surface forecasts.

    Parameters
    ----------
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self, set_type='best'):
        import warnings
        warnings.warn('HRRR_ESRL is an experimental model and is not always '
                        + 'available.')       
        model_type = 'Forecast Model Data'
        model = 'GSD HRRR CONUS 3km surface'
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere',
        'low_clouds':'Low_cloud_cover_UnknownLevelType-214',
        'mid_clouds':'Medium_cloud_cover_UnknownLevelType-224',
        'high_clouds':'High_cloud_cover_UnknownLevelType-234',
        'ghi':'Downward_short-wave_radiation_flux_surface', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds',
        'ghi', ]
        super(HRRR_ESRL, self).__init__(model_type, model, set_type)


        
class NAM(ForecastModel):
    '''
    Subclass of the ForecastModel class representing NAM forecast model.

    Model data corresponds to NAM CONUS 12km resolution forecasts
    from CONDUIT.

    Parameters
    ----------
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self,set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere_single_layer',
        'low_clouds':'Low_cloud_cover_low_cloud',
        'mid_clouds':'Medium_cloud_cover_middle_cloud',
        'high_clouds':'High_cloud_cover_high_cloud',
        'ghi':'Downward_Short-Wave_Radiation_Flux_surface', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds',
        'ghi', ]
        super(NAM, self).__init__(model_type, model, set_type)


class HRRR(ForecastModel):
    '''
    Subclass of the ForecastModel class representing HRRR forecast model.

    Model data corresponds to NCEP HRRR CONUS 2.5km resolution 
    forecasts.

    Parameters
    ----------
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NCEP HRRR CONUS 2.5km'            
        self.variables = {
        'temperature_iso':'Dewpoint_temperature_isobaric',
        'temperature_dew_iso':'Temperature_isobaric',
        'pressure':'Pressure_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere',
        'low_clouds':'Low_cloud_cover_low_cloud',
        'mid_clouds':'Medium_cloud_cover_middle_cloud',
        'high_clouds':'High_cloud_cover_high_cloud',
        'condensation_height':'Geopotential_height_adiabatic_condensation_lifted'}
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds', ]
        super(HRRR, self).__init__(model_type, model, set_type)


class NDFD(ForecastModel):
    '''
    Subclass of the ForecastModel class representing NDFD forecast model.

    Model data corresponds to NWS CONUS CONDUIT forecasts.

    Parameters
    ----------
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed':'Wind_speed_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'total_clouds':'Total_cloud_cover_surface', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        'total_clouds', ]
        super(NDFD, self).__init__(model_type, model, set_type)


class RAP(ForecastModel):
    '''
    Subclass of the ForecastModel class representing RAP forecast model.

    Model data corresponds to Rapid Refresh CONUS 20km resolution 
    forecasts.

    Parameters
    ----------
    set_type: string
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    modelvariables: list
        Common variable names.
    queryvariables: list
        Names of default variables specific to the model.
    variables: dictionary
        Dictionary of common variables that reference the model
        specific variables.
    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'Rapid Refresh CONUS 20km'
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere_single_layer',
        'low_clouds':'Low_cloud_cover_low_cloud',
        'mid_clouds':'Medium_cloud_cover_middle_cloud',
        'high_clouds':'High_cloud_cover_high_cloud', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds', ]
        super(RAP, self).__init__(model_type, model, set_type)
