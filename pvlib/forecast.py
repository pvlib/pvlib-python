
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

# Imports for weather data.
import netCDF4
from netCDF4 import num2date
import pandas as pd

class ForecastModel:#(object):
    r"""
    An object for holding forecast model information for use within the 
    pvlib library.

    Simplifies use of siphon library on a THREDDS server.

    Attributes
    ----------
    catalog_url : string
        The url path of the catalog to parse.
    base_tds_url : string
        The top level server address
    datasets : Dataset
        A dictionary of Dataset object, whose keys are the name of the
        dataset's name
    timespan: datetime
        A datetime object, that defines the time reference info for the
        dataset.
    latlon: list
        The extent of the areal coverage of the dataset.
    model: Dataset
        A dictionary of Dataset object, whose keys are the name of the
        dataset's name.
    model_url: string
        The url path of the dataset to parse.

    """

    def __init__(self,model_type,model_name,variables):
        self.access_url_key = 'NetcdfSubset'
        self.catalog_url = 'http://thredds.ucar.edu/thredds/catalog.xml'
        self.base_tds_url = self.catalog_url.split('/thredds/')[0]
        self.data_format = 'netcdf'
        self.model_name = model_name
        self.model_type = model_type
        self.variables = variables
        self.vert_level = 100000
        self.columns = []
                
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(self.catalog.catalog_refs[self.model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))
        self.model = TDSCatalog(self.fm_models.catalog_refs[self.model_name].href)
        self.datasets_list = list(self.model.datasets.keys())
        self.set_dataset()

    @property
    def columns(self):
        return self._columns
    @columns.setter
    def columns(self, value):
        self._columns = value 

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value 

    @property
    def model_type(self):
        return self._model_type
    @model_type.setter
    def model_type(self, value):
        self._model_type = value 

    @property
    def catalog_url(self):
        return self._catalog_url
    @catalog_url.setter
    def catalog_url(self, value):
        self._catalog_url = value 

    @property
    def base_tds_url(self):
        return self._base_tds_url
    @base_tds_url.setter
    def base_tds_url(self, value):
        self._base_tds_url = value 

    @property
    def data_format(self):
        return self._data_format
    @data_format.setter
    def data_format(self, value):
        self._data_format = value 

    @property
    def access_url_key(self):
        return self._access_url_key
    @access_url_key.setter
    def access_url_key(self, value):
        self._access_url_key = value 

    @property
    def vert_level(self):
        return self._vert_level
    @vert_level.setter
    def vert_level(self, value):
        self._vert_level = value 

    @property
    def variables(self):
        return self._variables
    @variables.setter
    def variables(self, value):
        self._variables = value

    @property
    def timespan(self):
        return self._timespan
    @timespan.setter
    def timespan(self, value):
        self._timespan = value

    @property
    def latlon(self):
        return self._latlon
    @latlon.setter
    def latlon(self, value):
        self._latlon = value

    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, value):
        self._time = value

    @property
    def fm_models(self):
        return self._fm_models
    @fm_models.setter
    def fm_models(self, value):
        self._fm_models = value

    @property
    def fm_models_list(self):
        return self._fm_models_list
    @fm_models_list.setter
    def fm_models_list(self, value):
        self._fm_models_list = value

    @property
    def datasets_list(self):
        return self._datasets_list
    @datasets_list.setter
    def datasets_list(self, value):
        self._datasets_list = value

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, value):
        self._data = value

    @property
    def query(self):
        return self._query
    @query.setter
    def query(self, value):
        self._query = value

    @property
    def ncss(self):
        return self._ncss
    @ncss.setter
    def ncss(self, value):
        self._ncss = value

    @property
    def model_name(self):
        return self._model_name
    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def var_units(self):
        return self._var_units
    @var_units.setter
    def var_units(self, value):
        self._var_units = value

    @property
    def var_stdnames(self):
        return self._var_stdnames
    @var_stdnames.setter
    def var_stdnames(self, value):
        self._var_stdnames = value
        


    def set_dataset(self,set_type='best'):
        r"""
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

        self.ncss = NCSS(self.dataset.access_urls[self.access_url_key])
        self.query = self.ncss.query()        

    def set_query_latlon(self):
        r"""
        as: point or box

        """
        if len(self.latlon)>2:
            self.query.lonlat_box(self.latlon[0],self.latlon[1],self.latlon[2],self.latlon[3])
        else:
            self.query.lonlat_point(self.latlon[0],self.latlon[1])

    def set_query_time(self):
        r"""
        as: single or range

        """
        if len(self.timespan)>1:
            self.query.time_range(self.timespan[0],self.timespan[1])
        else:
            self.query.time_range(self.timespan)        

    def get_query_data(self,latlon,timespan,vert_level=None,variables=None):
        if vert_level != None:
            self.vert_level = vert_level
        if variables != None:
            self.variables = variables
        
        self.latlon = latlon
        self.set_query_latlon()
        self.timespan = timespan
        self.set_query_time()
        self.query.vertical_level(self.vert_level)
        self.query.variables(*self.variables)

        self.query.accept(self.data_format)
        netcdf_data = self.ncss.get_data(self.query)
        self.set_time(netcdf_data.variables['time'])
        self.data = self.netcdf2pandas(netcdf_data)
        self.set_variable_units(netcdf_data)
        self.set_variable_stdnames(netcdf_data)
        netcdf_data.close()
        return self.data       

    def netcdf2pandas(self,data):
        data_dict = {}
        for var in self.variables:
            data_dict[var] = data[var][:].squeeze()

        dataframe = pd.DataFrame(data_dict)
        return dataframe

    def set_time(self,times):
        self.time = num2date(times[:].squeeze(), times.units)

    def set_variable_units(self,data):
        self.var_units = {}
        for var in self.variables:
            self.var_units[var] = data[var].units

    def set_variable_stdnames(self,data):
        self.var_stdnames = {}
        for var in self.variables:
            try:
                self.var_stdnames[var] = data[var].standard_name
            except AttributeError:
                self.var_stdnames[var] = var

class GFS(ForecastModel):
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'GFS Half Degree Forecast'
        variables = ['Temperature_isobaric',
                     'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_convective_cloud',
                     'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
                     'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
                     'Total_cloud_cover_middle_cloud_Mixed_intervals_Average']
        super().__init__(model_type,model,variables)

    ## detailed model description and info ############
    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, value):
        self._description = value

    def get_ghi(self):
        """
            calculate or grab ghi data

        """
        pass


class NAM(ForecastModel):
    def __init__(self):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'
        variables = ['Temperature_isobaric',
                     'Downward_Short-Wave_Radiation_Flux_surface',
                     'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
                     'Temperature_surface',
                     'High_cloud_cover_high_cloud',
                     'Low_cloud_cover_low_cloud',
                     'Medium_cloud_cover_middle_cloud',
                     'Total_cloud_cover_entire_atmosphere_single_layer']
        super().__init__(model_type,model,variables)

    ## detailed model description and info ############
    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, value):
        self._description = value

    def get_ghi(self):
        """
            calculate or grab ghi data

        """
        pass


class NDFD(ForecastModel):
    def __init__(self):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        variables = ['Total_cloud_cover_surface',
                     'Temperature_surface',
                     'Wind_speed_surface']
        super().__init__(model_type,model,variables)

    ## detailed model description and info ############
    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, value):
        self._description = value

    def get_ghi(self):
        """
            calculate or grab ghi data

        """
        pass


class Variable:
    def __init__(self):
        self.value = None
        self.name = None
        self.units = None

