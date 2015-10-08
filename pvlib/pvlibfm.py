
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

# Imports for weather data.
import netCDF4
from netCDF4 import num2date

class PvlibFM(object):
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

    def __init__(self,model_type='Forecast Model Data'):
        self.model_type = model_type
        self.catalog_url = 'http://thredds.ucar.edu/thredds/catalog.xml'
        self.base_tds_url = self.catalog_url.split('/thredds/')[0]
        self.access_url_key = 'NetcdfSubset'
        self.format = 'netcdf'
        self.catalog = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')
        self.set_fm_models()

    def set_fm_models(self):
        self.fm_models = TDSCatalog(self.catalog.catalog_refs[self.model_type].href)

    def set_dataset(self,extent='best'):
        r"""
        as: best,latest, or full

        """
        keys = list(self.model.datasets.keys())
        labels = [item.split()[0].lower() for item in keys]
        if extent == 'best':
            self.dataset = self.model.datasets[keys[labels.index('best')]]
        elif extent == 'latest':
            self.dataset = self.model.datasets[keys[labels.index('latest')]]
        elif extent == 'full':
            self.dataset = self.model.datasets[keys[labels.index('full')]]

        self.set_ncss()

    def set_ncss(self):
        self.ncss = NCSS(self.dataset.access_urls[self.access_url_key])
        self.query = self.ncss.query()

    def set_model(self,model_name):
        self.model = TDSCatalog(self.fm_models.catalog_refs[model_name].href)

    def set_latlon(self,extent):
        r"""
        as: point or box

        """
        if len(extent)>2:
            self.query.lonlat_box(extent[0],extent[1],extent[2],extent[3])
        else:
            self.query.lonlat_point(extent[0],extent[1])

    def set_time(self,extent):
        r"""
        as: single or range

        """
        if len(extent)>1:
            self.query.time_range(extent[0],extent[1])
        else:
            self.query.time_range(extent)

    def set_vertical_level(self,val):
        self.query.vertical_level(val)

    def set_query_vars(self,variables):
        self.query.variables(*variables)

    def get_query_data(self):
        self.submit_query()
        self.data = self.ncss.get_data(self.query)
        return self.data

    def get_fm_models(self):
        return sorted(list(self.fm_models.catalog_refs.keys()))

    def get_model_datasets(self):
        return list(self.model.datasets.keys())

    def get_timespan(self):
        pass

    def get_latlon(self):
        pass

    def submit_query(self):
        self.query.accept(self.format)

    ### data variables ##############################################

    def get_cloud_cover(self):
        return self.data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']

    def get_time(self):
        time = self.data.variables['time']
        return num2date(time[:].squeeze(), time.units)

    def get_ghi(self):
        return self.data['Downward_short-wave_radiation_flux_surface']

    def get_wind(self):
        pass

    def get_temp(self):
        return self.data['Temperature_isobaric']

    def get_model_vars(self):
        return self.ncss.variables

    def get_variables(self):
        return self.ncss.variables

    def close(self):
        self.data.close()




