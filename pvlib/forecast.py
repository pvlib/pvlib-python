"""
The 'forecast' module contains class definitions for 
retreiving forecasted data from UNIDATA Thredd servers.
"""

from netCDF4 import num2date
import pandas as pd
import numpy as np

from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS


class ForecastModel(object):
    '''
    An object for holding forecast model information for use within the 
    pvlib library.

    Simplifies use of siphon library on a THREDDS server.

    Parameters
    ----------
    labels: dictionary
        Dictionary where the common variable name references the model 
        specific variable name.
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
    time: datetime
        Time range specified for the NCSS query.
    timespan: datetime
        A datetime object, that defines the time range for the data.
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
                        'temperature_iso',
                        'wind_speed',
                        'total_clouds',
                        'low_clouds',
                        'mid_clouds',
                        'high_clouds',
                        'boundary_clouds',
                        'convect_clouds',
                        'downward_shortwave_radflux',
                        'downward_shortwave_radflux_avg',])

    def __init__(self,model_type,model_name,labels):
        self.model_name = model_name
        self.model_type = model_type
        self.variables = list(labels.keys())
        self.modelvariables = list(labels.values())
        self.data_labels = labels
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(self.catalog.catalog_refs[self.model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))
        self.model = TDSCatalog(self.fm_models.catalog_refs[self.model_name].href)
        self.datasets_list = list(self.model.datasets.keys())
        self.set_dataset()

    def set_dataset(self,set_type='best'):
        """
        Retreives the designated dataset, creates NCSS object, and 
        initiates a NCSS query.

        Parameters
        ----------
        set_type: string
            as: "best","latest", or "full"

        """
        keys = list(self.model.datasets.keys())
        labels = [item.split()[0].lower() for item in keys]
        if set_type == 'best':
            self.dataset = self.model.datasets[keys[labels.index('best')]]
        elif set_type == 'latest':
            self.dataset = self.model.datasets[keys[labels.index('latest')]]
        elif set_type == 'full':
            self.dataset = self.model.datasets[keys[labels.index('full')]]

        self.access_url = self.dataset.access_urls[self.access_url_key]
        self.ncss = NCSS(self.access_url)
        self.query = self.ncss.query()        

    def set_query_latlon(self):
        """
        Sets the NCSS query location latitude and longitude.

        """
        if len(self.latlon)>2:
            self.query.lonlat_box(self.latlon[0],self.latlon[1],self.latlon[2],self.latlon[3])
        else:
            self.query.lonlat_point(self.latlon[0],self.latlon[1])

    def set_query_time(self):
        """
        Sets the NCSS query time range.

        as: single or range

        """
        if len(self.timespan)>1:
            self.query.time_range(self.timespan[0],self.timespan[1])
        else:
            self.query.time_range(self.timespan)        

    def get_query_data(self,latlon,timespan,vert_level=None,variables=None):
        """
        Submits a query to the UNIDATA servers using siphon NCSS and 
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ==========

        latlon: list
            Latitude and longitude of query location.
        timespan: datetime
            Time range of interest.
        vert_level: float or integer
            Vertical altitude of interest.
        variables: list
            Variables being queried.

        Returns
        =======
        pd.DataFrame
        """
        if vert_level != None:
            self.vert_level = vert_level
        if variables != None:
            self.modelvariables = variables
        
        self.latlon = latlon
        self.set_query_latlon()
        self.timespan = timespan
        self.set_query_time()
        self.query.vertical_level(self.vert_level)
        self.query.variables(*self.modelvariables)

        self.query.accept(self.data_format)
        netcdf_data = self.ncss.get_data(self.query)
        self.set_time(netcdf_data.variables['time'])
        self.data = self.netcdf2pandas(netcdf_data)
        self.set_variable_units(netcdf_data)
        self.set_variable_stdnames(netcdf_data)
        netcdf_data.close()
        return self.data       

    def netcdf2pandas(self,data):
        """
        Transforms data from netcdf  to pandas DataFrame.

        Parameters
        ==========
        data: netcdf
            Data returned from UNIDATA NCSS query.

        Returns
        =======
        pd.DataFrame
        """
        data_dict = {}
        for var in self.variables:
            data_dict[var] = pd.Series(data[self.data_labels[var]][:].squeeze())
        dataframe = pd.DataFrame(data_dict,columns=self.columns)
        return dataframe

    def set_time(self,times):
        """
        Converts time data into a netcdf date format.

        Parameters
        ==========
        times: netcdf
            Contains time information.

        Returns
        =======
        netCDF4.datetime
        """
        self.time = num2date(times[:].squeeze(), times.units)

    def set_variable_units(self,data):
        """
        Extracts variable unit information from netcdf data.

        Parameters
        ==========
        data: netcdf
            Contains queried variable information.

        """
        self.var_units = {}
        for var in self.variables:
            self.var_units[var] = data[self.data_labels[var]].units

    def set_variable_stdnames(self,data):
        """
        Extracts standard names from netcdf data.

        Parameters
        ==========
        data: netcdf
            Contains queried variable information.

        """
        self.var_stdnames = {}
        for var in self.variables:
            try:
                self.var_stdnames[var] = data[self.data_labels[var]].standard_name
            except AttributeError:
                self.var_stdnames[var] = var


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
    def __init__(self,res='Half'):
        model_type = 'Forecast Model Data'
        model = 'GFS '+res+' Degree Forecast'
        variables = ['Temperature_isobaric',
                     'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
                     'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_convective_cloud']
        cols = super(GFS, self).columns
        idx = [1,3,4,5,6,7,8]
        data_labels = dict(zip(cols[idx],variables))
        super(GFS, self).__init__(model_type,model,data_labels)


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
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'
        description = ''
        variables = ['Temperature_surface',
                     'Temperature_isobaric',
                     'Total_cloud_cover_entire_atmosphere_single_layer',
                     'Low_cloud_cover_low_cloud',
                     'Medium_cloud_cover_middle_cloud',
                     'High_cloud_cover_high_cloud',
                     'Downward_Short-Wave_Radiation_Flux_surface',
                     'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average']
        cols = super(NAM, self).columns
        idx = [0,1,3,4,5,6,9,10]
        data_labels = dict(zip(cols[idx],variables))
        super(NAM, self).__init__(model_type,model,data_labels)


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
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'Rapid Refresh CONUS 20km'
        description = ''
        variables = ['Temperature_surface',
                     'Total_cloud_cover_entire_atmosphere_single_layer',
                     'Low_cloud_cover_low_cloud',
                     'Medium_cloud_cover_middle_cloud',
                     'High_cloud_cover_high_cloud']
        cols = super(RAP, self).columns
        idx = [0,3,4,5,6]
        data_labels = dict(zip(cols[idx],variables))
        super(RAP, self).__init__(model_type,model,data_labels)


class NCEP(ForecastModel):
    '''
    Subclass of the ForecastModel class representing NCEP forecast model.

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
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'NCEP HRRR CONUS 2.5km'            
        description = ''
        variables = ['Total_cloud_cover_entire_atmosphere',
                     'Low_cloud_cover_low_cloud',
                     'Medium_cloud_cover_middle_cloud',
                     'High_cloud_cover_high_cloud',]
        cols = super(NCEP, self).columns
        idx = [3,4,5,6]
        data_labels = dict(zip(cols[idx],variables))
        super(NCEP, self).__init__(model_type,model,data_labels)


class GSD(ForecastModel):
    '''
    Subclass of the ForecastModel class representing NCEP forecast model.

    Model data corresponds to NOAA/GSD HRRR CONUS 3km resolution
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
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'GSD HRRR CONUS 3km surface'

        description = ''
        variables = ['Temperature_surface',
                     'Total_cloud_cover_entire_atmosphere',
                     'Low_cloud_cover_UnknownLevelType-214',
                     'Medium_cloud_cover_UnknownLevelType-224',
                     'High_cloud_cover_UnknownLevelType-234',
                     'Downward_short-wave_radiation_flux_surface']
        cols = super(GSD, self).columns
        idx = [0,3,4,5,6,9]
        data_labels = dict(zip(cols[idx],variables))
        super(GSD, self).__init__(model_type,model,data_labels)


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

    def __init__(self):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        description = ''
        variables = ['Temperature_surface',
                     'Wind_speed_surface',
                     'Total_cloud_cover_surface']
        cols = super(NDFD, self).columns
        idx = [0,2,3]
        data_labels = dict(zip(cols[idx],variables))
        super(NDFD, self).__init__(model_type,model,data_labels)
