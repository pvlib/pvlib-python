'''
The 'forecast' module contains class definitions for 
retreiving forecasted data from UNIDATA Thredd servers.
'''

from netCDF4 import num2date
import pandas as pd
import pytz
import numpy as np

from pvlib import *
from pvlib.location import Location
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

    Attributes
    ----------
    access_url: string
        URL specifying the dataset from data will be retrieved.
    access_url_key: string
        Key spcificying the type of data to be returned with the query.
    base_tds_url : string
        The top level server address
    catalog_url : string
        The url path of the catalog to parse.
    data: pd.DataFrame
        Data returned from the query.
    data_labels: dictionary
        Dictionary that translates model specific variables to 
        common named variables.
    dataformat: string
        Format of the forecast data being requested from UNIDATA.
    dataset: Dataset
        Object containing information used to access forecast data.
    datasets_list: list
        List of all available datasets.
    fm_models: Dataset
        Object containing all available foreast models.
    fm_models_list: list
        List of all available forecast models from UNIDATA.
    latlon: list
        The extent of the areal coverage of the dataset.
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
    query: NCSS query object
        NCSS object used to complete the forecast data retrival.
    set_type: string
        Model dataset type.
    time: datetime
        Time range specified for the NCSS query.
    var_stdnames: dictionary
        Dictionary containing the standard names of the variables in the
        query, where the keys are the common names.
    var_units: dictionary
        Dictionary containing the unites of the variables in the query,
        where the keys are the common names.
    variables: list
        Model specific variable names.
    vert_level: float or integer
        Vertical altitude for query data.
    '''

    access_url_key = 'NetcdfSubset'
    catalog_url = 'http://thredds.ucar.edu/thredds/catalog.xml'
    base_tds_url = catalog_url.split('/thredds/')[0]
    data_format = 'netcdf'
    vert_level = 100000
    columns = np.array(['temperature',
                        'wind_speed',
                        'pressure',
                        'total_clouds',
                        'low_clouds',
                        'mid_clouds',
                        'high_clouds',
                        'dni',
                        'dhi',
                        'ghi', ])

    def __init__(self, model_type, model_name, set_type):
        self.model_name = model_name
        self.model_type = model_type
        self.set_type = set_type
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(self.catalog.catalog_refs[model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))
        self.model = TDSCatalog(self.fm_models.catalog_refs[model_name].href)
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
        if self.longitude is list:
            # west, east, south, north
            self.query.lonlat_box(self.latitude[0], self.latitude[1], 
                                    self.longitude[0], self.longitude[1]) 

            self.lbox = True
        else:
            self.query.lonlat_point(self.longitude, self.latitude)
            self.lbox = False

    def set_query_time(self):
        '''
        Sets the NCSS query time range.

        as: single or range

        '''
        if len(self.time) == 1:
            self.query.time(pd.to_datetime(self.time)[0])
        else:
            self.query.time_range(pd.to_datetime(self.time)[0], 
                pd.to_datetime(self.time)[-1])        

    def get_query_data(self, latitude, longitude, time, vert_level=None, 
        variables=None):
        '''
        Submits a query to the UNIDATA servers using siphon NCSS and 
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ----------

        variables: dictionary
            Variables and common names being queried.
        latlon: list
            Latitude and longitude of query location.
        time: pd.datetimeindex
            Time range of interest.
        vert_level: float or integer
            Vertical altitude of interest.

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
        self.time = time
        self.set_query_time()
        # print(pd.to_datetime(self.time)[0], pd.to_datetime(self.time)[-1])
        self.query.vertical_level(self.vert_level)
        self.query.variables(*self.queryvariables)
        self.query.accept(self.data_format)
        netcdf_data = self.ncss.get_data(self.query)

        # print(netcdf_data.variables['Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average'][:])

        # self.netTime = num2date(netcdf_data.variables['time'][:].squeeze(),
        #     netcdf_data.variables['time'].units)

        self.set_time(netcdf_data.variables['time'], self.time.tzinfo)

        self.data = self.netcdf2pandas(netcdf_data)
        self.time = self.time
        self.set_variable_units(netcdf_data)
        self.set_variable_stdnames(netcdf_data)
        self.convert_temperature(netcdf_data)

        self.calc_wind(netcdf_data)
        self.calc_radiation(netcdf_data)

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
                    data[self.variables[var]][:].squeeze(), index=self.time)
            df = pd.DataFrame(data_dict, columns=self.columns, index=self.time)
            return df
        else:
            return pd.DataFrame(columns=self.columns, index=self.time)

    def set_time(self, times, tzinfo):
        '''
        Converts time data into a pandas date object.

        Parameters
        ----------
        times: netcdf
            Contains time information.
        tzinfo: tzinfo
            Timezone information.

        Returns
        -------
        pandas.DatetimeIndex
        '''
        self.time = pd.DatetimeIndex(pd.Series(num2date(times[:].squeeze(),
            times.units)))
        # print(type(tzinfo))
        # self.time = tzinfo.localize(self.time)
        # self.time = self.time.astimezone(pytz.timezone(tzinfo))

        self.time = self.time.tz_localize(tzinfo) # correct, but doesn't plot well

        # self.time = self.time.tz_localize('UTC')
        # self.time = self.time.tz_convert(tzinfo)
        # self.time = self.time.tz_localize(None) # shows correct time but no timezone

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

    def calc_radiation(self, data, cloud_type='total_clouds', rad_funcs=None, 
        args=None, keys=None):
        '''
        Determines shortwave radiation values if they are missing from 
        the model data.

        Parameters
        ----------
        zenith: pd.daraSeries
            Solar zenith angle.

        '''
        self.rad_type = {}
        if not self.lbox and cloud_type in self.data:
            cloud_prct = self.data[cloud_type]
            self.location = Location(self.latitude, self.longitude)#,
                                        #tz=self.time.tzinfo)
            print(self.time)
            solpos = solarposition.get_solarposition(self.time, self.location)
            self.zenith = np.array(solpos.zenith)
            if rad_funcs is None:
                for rad in ['dni','dhi','ghi']:
                    if rad in self.modelvariables:
                        self.data[rad] = pd.Series(
                            data[self.variables[rad]][:].squeeze(), 
                            index=self.time)
                        self.rad_type[rad] = 'forecast'
                    else:
                        self.rad_type[rad] = 'liujordan'
                        if rad == 'dni':
                            self.data['dni'] = \
                                irradiance.liujordan_dni(self.zenith,
                                    cloud_prct)
                        elif rad == 'dhi':
                            self.data['dhi'] = \
                                irradiance.liujordan_dhi(self.zenith, 
                                    cloud_prct)
                        elif rad == 'ghi':
                            self.data['ghi'] = \
                                irradiance.liujordan_ghi(self.zenith, 
                                    cloud_prct)
            elif rad_funcs == 'linear':
                cs = clearsky.ineichen(self.data.index, self.location)
                cs_scaled = cs.mul((100. - self.data['total_clouds']) / 100.,
                                     axis=0)
                self.data['dni'] = cs
                self.data['ghi'] = cs_scaled
                self.rad_type['dni'] = 'linear'
                self.rad_type['ghi'] = 'linear'
                self.rad_type['dhi'] = 'forecast'
            else:
                length = len(rad_funcs)
                if any(len(lst) != length for lst in [args, keys]):
                    import warnings
                    warnings.warn('Radiation functions, arguments, and keys \
                                    should be of equal length')
                    return

                for i in range(len(rad_funcs)):
                    self.data[keys[i]] = rad_funcs[i](*args[i])
                    self.rad_type[keys[i]] = rad_funcs

            for var in ['dni', 'dhi', 'ghi']:
                self.var_units[var] = '$W m^{-2}$'

    def convert_temperature(self, data):
        '''
        Converts Kelvin to celsius.

        '''
        for atemp in ['temperature']:
            if atemp in self.data.columns:
                self.data[atemp] -= 273.15
                self.var_units[atemp] = 'C'

    def calc_wind(self, data):
        '''
        Calculates wind speed quantity.

        '''
        if not self.lbox:
            if 'u-component_of_wind_isobaric' in self.queryvariables and \
                'v-component_of_wind_isobaric' in self.queryvariables:
                wind_data = np.sqrt(\
                    data['u-component_of_wind_isobaric'][:].squeeze()**2 +
                    data['v-component_of_wind_isobaric'][:].squeeze()**2)
                self.wind_type = 'component'
            elif 'Wind_speed_gust_surface' in self.queryvariables:
                wind_data = 0.25 * data['Wind_speed_gust_surface'][:].squeeze()
                self.wind_type = 'gust'

            if 'wind_speed' in self.data:
                self.data['wind_speed'] = pd.Series(wind_data, index=self.time)
                self.var_units['wind_speed'] = 'm/s'


class GFS(ForecastModel):
    '''
    Subclass of the ForecastModel class representing GFS forecast model.

    Model data corresponds to 0.25 degree resolution forecasts.

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: list
        Names of default variables specific to the model.

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
        # 'pressure':'Pressure_surface',
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
        # 'pressure',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds',
        'boundary_clouds',
        'convect_clouds',
        'ghi', ]
        super(GFS, self).__init__(model_type, model, set_type)

        # def map_data(self):


class HRRR_ESRL(ForecastModel):
    '''
    Subclass of the ForecastModel class representing
    NOAA/GSD/ESRL's HRRR forecast model. This is not an operational product.

    Model data corresponds to NOAA/GSD/ESRL HRRR CONUS 3km resolution
    surface forecasts.

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: list
        Names of default variables specific to the model.

    '''

    def __init__(self, set_type='best'):
        import warnings
        warnings.warn('HRRR_ESRL is an experimental model and is not always \
                         available.')       
        model_type = 'Forecast Model Data'
        model = 'GSD HRRR CONUS 3km surface'
        description = ''
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        # 'pressure':'Pressure_surface',
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
        # 'pressure',
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

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    res: string
        Determines which resolution of the GFS to use, as 'Half' or 'Quarter'
    variables: list
        Names of default variables specific to the model.

    '''

    def __init__(self,set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'
        description = ''
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        # 'pressure':'Pressure_surface',
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
        # 'pressure',
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

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: list
        Names of default variables specific to the model.

    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NCEP HRRR CONUS 2.5km'            
        description = ''
        self.variables = {
        'temperature':'Temperature_height_above_ground',
        'wind_speed_gust':'Wind_speed_gust_surface',
        # 'pressure':'Pressure_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere',
        'low_clouds':'Low_cloud_cover_low_cloud',
        'mid_clouds':'Medium_cloud_cover_middle_cloud',
        'high_clouds':'High_cloud_cover_high_cloud', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        # 'pressure',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
                            'high_clouds', ]
        super(HRRR, self).__init__(model_type, model, set_type)


class NDFD(ForecastModel):
    '''
    Subclass of the ForecastModel class representing NDFD forecast model.

    Model data corresponds to NWS CONUS CONDUIT forecasts.

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: list
        Names of default variables specific to the model.

    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        description = ''
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

    Attributes
    ----------
    cols: list
        Common names for variables.
    data_labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
    idx: list
        Indices of the variables corresponding to their common name.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: list
        Names of default variables specific to the model.

    '''

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'Rapid Refresh CONUS 20km'
        description = ''
        self.variables = {
        'temperature':'Temperature_surface',
        'wind_speed_gust':'Wind_speed_gust_surface',
        # 'pressure':'Pressure_surface',
        'total_clouds':'Total_cloud_cover_entire_atmosphere_single_layer',
        'low_clouds':'Low_cloud_cover_low_cloud',
        'mid_clouds':'Medium_cloud_cover_middle_cloud',
        'high_clouds':'High_cloud_cover_high_cloud', }
        self.modelvariables = self.variables.keys()
        self.queryvariables = [self.variables[key] for key in \
                                self.modelvariables]
        self.dataframe_variables = [
        'temperature',
        # 'pressure',
        'total_clouds',
        'low_clouds',
        'mid_clouds',
        'high_clouds', ]
        super(RAP, self).__init__(model_type, model, set_type)
