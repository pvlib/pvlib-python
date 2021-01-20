'''
The 'forecast' module contains class definitions for
retreiving forecasted data from UNIDATA Thredd servers.
'''
from netCDF4 import num2date
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from xml.etree.ElementTree import ParseError

from pvlib.location import Location
from pvlib.irradiance import campbell_norman, get_extra_radiation, disc
from pvlib.irradiance import _liujordan
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

import warnings


warnings.warn(
    'The forecast module algorithms and features are highly experimental. '
    'The API may change, the functionality may be consolidated into an io '
    'module, or the module may be separated into its own package.')


class ForecastModel:
    """
    An object for querying and holding forecast model information for
    use within the pvlib library.

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
        TDSCatalog object containing all available
        forecast models from UNIDATA.
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
        NCSS
    model_name: string
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
    time: DatetimeIndex
        Time range.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    vert_level: float or integer
        Vertical altitude for query data.
    """

    access_url_key = 'NetcdfSubset'
    catalog_url = 'https://thredds.ucar.edu/thredds/catalog.xml'
    base_tds_url = catalog_url.split('/thredds/')[0]
    data_format = 'netcdf'

    units = {
        'temp_air': 'C',
        'wind_speed': 'm/s',
        'ghi': 'W/m^2',
        'ghi_raw': 'W/m^2',
        'dni': 'W/m^2',
        'dhi': 'W/m^2',
        'total_clouds': '%',
        'low_clouds': '%',
        'mid_clouds': '%',
        'high_clouds': '%'}

    def __init__(self, model_type, model_name, set_type, vert_level=None):
        self.model_type = model_type
        self.model_name = model_name
        self.set_type = set_type
        self.connected = False
        self.vert_level = vert_level

    def connect_to_catalog(self):
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(
            self.catalog.catalog_refs[self.model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))

        try:
            model_url = self.fm_models.catalog_refs[self.model_name].href
        except ParseError:
            raise ParseError(self.model_name + ' model may be unavailable.')

        try:
            self.model = TDSCatalog(model_url)
        except HTTPError:
            try:
                self.model = TDSCatalog(model_url)
            except HTTPError:
                raise HTTPError(self.model_name + ' model may be unavailable.')

        self.datasets_list = list(self.model.datasets.keys())
        self.set_dataset()
        self.connected = True

    def __repr__(self):
        return f'{self.model_name}, {self.set_type}'

    def set_dataset(self):
        '''
        Retrieves the designated dataset, creates NCSS object, and
        creates a NCSS query object.
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

    def set_query_time_range(self, start, end):
        """
        Parameters
        ----------
        start : datetime.datetime, pandas.Timestamp
            Must be tz-localized.
        end : datetime.datetime, pandas.Timestamp
            Must be tz-localized.

        Notes
        -----
        Assigns ``self.start``, ``self.end``. Modifies ``self.query``
        """
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        if self.start.tz is None or self.end.tz is None:
            raise TypeError('start and end must be tz-localized')
        self.query.time_range(self.start, self.end)

    def set_query_latlon(self):
        '''
        Sets the NCSS query location latitude and longitude.
        '''

        if (isinstance(self.longitude, list) and
                isinstance(self.latitude, list)):
            self.lbox = True
            # west, east, south, north
            self.query.lonlat_box(self.longitude[0], self.longitude[1],
                                  self.latitude[0], self.latitude[1])
        else:
            self.lbox = False
            self.query.lonlat_point(self.longitude, self.latitude)

    def set_location(self, tz, latitude, longitude):
        '''
        Sets the location for the query.

        Parameters
        ----------
        tz: tzinfo
            Timezone of the query
        latitude: float
            Latitude of the query
        longitude: float
            Longitude of the query

        Notes
        -----
        Assigns ``self.location``.
        '''
        self.location = Location(latitude, longitude, tz=tz)

    def get_data(self, latitude, longitude, start, end,
                 vert_level=None, query_variables=None,
                 close_netcdf_data=True, **kwargs):
        """
        Submits a query to the UNIDATA servers using Siphon NCSS and
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ----------
        latitude: float
            The latitude value.
        longitude: float
            The longitude value.
        start: datetime or timestamp
            The start time.
        end: datetime or timestamp
            The end time.
        vert_level: None, float or integer, default None
            Vertical altitude of interest.
        query_variables: None or list, default None
            If None, uses self.variables.
        close_netcdf_data: bool, default True
            Controls if the temporary netcdf data file should be closed.
            Set to False to access the raw data.
        **kwargs:
            Additional keyword arguments are silently ignored.

        Returns
        -------
        forecast_data : DataFrame
            column names are the weather model's variable names.
        """

        if not self.connected:
            self.connect_to_catalog()

        if vert_level is not None:
            self.vert_level = vert_level

        if query_variables is None:
            self.query_variables = list(self.variables.values())
        else:
            self.query_variables = query_variables

        self.set_query_time_range(start, end)

        self.latitude = latitude
        self.longitude = longitude
        self.set_query_latlon()  # modifies self.query
        self.set_location(self.start.tz, latitude, longitude)

        if self.vert_level is not None:
            self.query.vertical_level(self.vert_level)

        self.query.variables(*self.query_variables)
        self.query.accept(self.data_format)

        self.netcdf_data = self.ncss.get_data(self.query)

        # might be better to go to xarray here so that we can handle
        # higher dimensional data for more advanced applications
        self.data = self._netcdf2pandas(self.netcdf_data, self.query_variables,
                                        self.start, self.end)

        if close_netcdf_data:
            self.netcdf_data.close()

        return self.data

    def process_data(self, data, **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data. Most forecast models implement
        their own version of this method which also call this one.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """
        data = self.rename(data)
        return data

    def get_processed_data(self, *args, **kwargs):
        """
        Get and process forecast data.

        Parameters
        ----------
        *args: positional arguments
            Passed to get_data
        **kwargs: keyword arguments
            Passed to get_data and process_data

        Returns
        -------
        data: DataFrame
            Processed forecast data
        """
        return self.process_data(self.get_data(*args, **kwargs), **kwargs)

    def rename(self, data, variables=None):
        """
        Renames the columns according the variable mapping.

        Parameters
        ----------
        data: DataFrame
        variables: None or dict, default None
            If None, uses self.variables

        Returns
        -------
        data: DataFrame
            Renamed data.
        """
        if variables is None:
            variables = self.variables
        return data.rename(columns={y: x for x, y in variables.items()})

    def _netcdf2pandas(self, netcdf_data, query_variables, start, end):
        """
        Transforms data from netcdf to pandas DataFrame.

        Parameters
        ----------
        data: netcdf
            Data returned from UNIDATA NCSS query.
        query_variables: list
            The variables requested.
        start: Timestamp
            The start time
        end: Timestamp
            The end time

        Returns
        -------
        pd.DataFrame
        """
        # set self.time
        try:
            time_var = 'time'
            self.set_time(netcdf_data.variables[time_var])
        except KeyError:
            # which model does this dumb thing?
            time_var = 'time1'
            self.set_time(netcdf_data.variables[time_var])

        data_dict = {}
        for key, data in netcdf_data.variables.items():
            # if accounts for possibility of extra variable returned
            if key not in query_variables:
                continue
            squeezed = data[:].squeeze()

            # If the data is big endian, swap the byte order to make it
            # little endian
            if squeezed.dtype.byteorder == '>':
                squeezed = squeezed.byteswap().newbyteorder()
            if squeezed.ndim == 1:
                data_dict[key] = squeezed
            elif squeezed.ndim == 2:
                for num, data_level in enumerate(squeezed.T):
                    data_dict[key + '_' + str(num)] = data_level
            else:
                raise ValueError('cannot parse ndim > 2')

        data = pd.DataFrame(data_dict, index=self.time)
        # sometimes data is returned as hours since T0
        # where T0 is before start. Then the hours between
        # T0 and start are added *after* end. So sort and slice
        # to remove the garbage
        data = data.sort_index().loc[start:end]
        return data

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
        times = num2date(time[:].squeeze(), time.units,
                         only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)
        self.time = pd.DatetimeIndex(pd.Series(times), tz=self.location.tz)

    def cloud_cover_to_ghi_linear(self, cloud_cover, ghi_clear, offset=35,
                                  **kwargs):
        """
        Convert cloud cover to GHI using a linear relationship.

        0% cloud cover returns ghi_clear.

        100% cloud cover returns offset*ghi_clear.

        Parameters
        ----------
        cloud_cover: numeric
            Cloud cover in %.
        ghi_clear: numeric
            GHI under clear sky conditions.
        offset: numeric, default 35
            Determines the minimum GHI.
        kwargs
            Not used.

        Returns
        -------
        ghi: numeric
            Estimated GHI.

        References
        ----------
        Larson et. al. "Day-ahead forecasting of solar power output from
        photovoltaic plants in the American Southwest" Renewable Energy
        91, 11-20 (2016).
        """

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
        return ghi

    def cloud_cover_to_irradiance_clearsky_scaling(self, cloud_cover,
                                                   method='linear',
                                                   **kwargs):
        """
        Estimates irradiance from cloud cover in the following steps:

        1. Determine clear sky GHI using Ineichen model and
           climatological turbidity.
        2. Estimate cloudy sky GHI using a function of
           cloud_cover e.g.
           :py:meth:`~ForecastModel.cloud_cover_to_ghi_linear`
        3. Estimate cloudy sky DNI using the DISC model.
        4. Calculate DHI from DNI and GHI.

        Parameters
        ----------
        cloud_cover : Series
            Cloud cover in %.
        method : str, default 'linear'
            Method for converting cloud cover to GHI.
            'linear' is currently the only option.
        **kwargs
            Passed to the method that does the conversion

        Returns
        -------
        irrads : DataFrame
            Estimated GHI, DNI, and DHI.
        """
        solpos = self.location.get_solarposition(cloud_cover.index)
        cs = self.location.get_clearsky(cloud_cover.index, model='ineichen',
                                        solar_position=solpos)

        method = method.lower()
        if method == 'linear':
            ghi = self.cloud_cover_to_ghi_linear(cloud_cover, cs['ghi'],
                                                 **kwargs)
        else:
            raise ValueError('invalid method argument')

        dni = disc(ghi, solpos['zenith'], cloud_cover.index)['dni']
        dhi = ghi - dni * np.cos(np.radians(solpos['zenith']))

        irrads = pd.DataFrame({'ghi': ghi, 'dni': dni, 'dhi': dhi}).fillna(0)
        return irrads

    def cloud_cover_to_transmittance_linear(self, cloud_cover, offset=0.75,
                                            **kwargs):
        """
        Convert cloud cover to atmospheric transmittance using a linear
        model.

        0% cloud cover returns offset.

        100% cloud cover returns 0.

        Parameters
        ----------
        cloud_cover : numeric
            Cloud cover in %.
        offset : numeric, default 0.75
            Determines the maximum transmittance.
        kwargs
            Not used.

        Returns
        -------
        ghi : numeric
            Estimated GHI.
        """
        transmittance = ((100.0 - cloud_cover) / 100.0) * offset

        return transmittance

    def cloud_cover_to_irradiance_campbell_norman(self, cloud_cover, **kwargs):
        """
        Estimates irradiance from cloud cover in the following steps:

        1. Determine transmittance using a function of cloud cover e.g.
           :py:meth:`~ForecastModel.cloud_cover_to_transmittance_linear`
        2. Calculate GHI, DNI, DHI using the
           :py:func:`pvlib.irradiance.campbell_norman` model

        Parameters
        ----------
        cloud_cover : Series

        Returns
        -------
        irradiance : DataFrame
            Columns include ghi, dni, dhi
        """
        # in principle, get_solarposition could use the forecast
        # pressure, temp, etc., but the cloud cover forecast is not
        # accurate enough to justify using these minor corrections
        solar_position = self.location.get_solarposition(cloud_cover.index)
        dni_extra = get_extra_radiation(cloud_cover.index)

        transmittance = self.cloud_cover_to_transmittance_linear(cloud_cover,
                                                                 **kwargs)

        irrads = campbell_norman(solar_position['apparent_zenith'],
                                 transmittance, dni_extra=dni_extra)
        irrads = irrads.fillna(0)

        return irrads

    def cloud_cover_to_irradiance(self, cloud_cover, how='clearsky_scaling',
                                  **kwargs):
        """
        Convert cloud cover to irradiance. A wrapper method.

        Parameters
        ----------
        cloud_cover : Series
        how : str, default 'clearsky_scaling'
            Selects the method for conversion. Can be one of
            clearsky_scaling or campbell_norman. Method liujordan is
            deprecated.
        **kwargs
            Passed to the selected method.

        Returns
        -------
        irradiance : DataFrame
            Columns include ghi, dni, dhi
        """

        how = how.lower()
        if how == 'clearsky_scaling':
            irrads = self.cloud_cover_to_irradiance_clearsky_scaling(
                cloud_cover, **kwargs)
        elif how == 'campbell_norman':
            irrads = self.cloud_cover_to_irradiance_campbell_norman(
                cloud_cover, **kwargs)
        else:
            raise ValueError('invalid how argument')

        return irrads

    def kelvin_to_celsius(self, temperature):
        """
        Converts Kelvin to celsius.

        Parameters
        ----------
        temperature: numeric

        Returns
        -------
        temperature: numeric
        """
        return temperature - 273.15

    def isobaric_to_ambient_temperature(self, data):
        """
        Calculates temperature from isobaric temperature.

        Parameters
        ----------
        data: DataFrame
            Must contain columns pressure, temperature_iso,
            temperature_dew_iso. Input temperature in K.

        Returns
        -------
        temperature : Series
            Temperature in K
        """

        P = data['pressure'] / 100.0                            # noqa: N806
        Tiso = data['temperature_iso']                          # noqa: N806
        Td = data['temperature_dew_iso'] - 273.15               # noqa: N806

        # saturation water vapor pressure
        e = 6.11 * 10**((7.5 * Td) / (Td + 273.3))

        # saturation water vapor mixing ratio
        w = 0.622 * (e / (P - e))

        temperature = Tiso - ((2.501 * 10.**6) / 1005.7) * w

        return temperature

    def uv_to_speed(self, data):
        """
        Computes wind speed from wind components.

        Parameters
        ----------
        data : DataFrame
            Must contain the columns 'wind_speed_u' and 'wind_speed_v'.

        Returns
        -------
        wind_speed : Series
        """
        wind_speed = np.sqrt(data['wind_speed_u']**2 + data['wind_speed_v']**2)

        return wind_speed

    def gust_to_speed(self, data, scaling=1/1.4):
        """
        Computes standard wind speed from gust.
        Very approximate and location dependent.

        Parameters
        ----------
        data : DataFrame
            Must contain the column 'wind_speed_gust'.

        Returns
        -------
        wind_speed : Series
        """
        wind_speed = data['wind_speed_gust'] * scaling

        return wind_speed


class GFS(ForecastModel):
    """
    Subclass of the ForecastModel class representing GFS
    forecast model.

    Model data corresponds to 0.25 degree resolution forecasts.

    Parameters
    ----------
    resolution: string, default 'half'
        Resolution of the model, either 'half' or 'quarter' degree.
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    _resolutions = ['Half', 'Quarter']

    def __init__(self, resolution='half', set_type='best'):
        model_type = 'Forecast Model Data'

        resolution = resolution.title()
        if resolution not in self._resolutions:
            raise ValueError(f'resolution must in {self._resolutions}')

        model = f'GFS {resolution} Degree Forecast'

        # isobaric variables will require a vert_level to prevent
        # excessive data downloads
        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'wind_speed_u': 'u-component_of_wind_isobaric',
            'wind_speed_v': 'v-component_of_wind_isobaric',
            'total_clouds':
                'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
            'low_clouds':
                'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
            'mid_clouds':
                'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
            'high_clouds':
                'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
            'boundary_clouds': ('Total_cloud_cover_boundary_layer_cloud_'
                                'Mixed_intervals_Average'),
            'convect_clouds': 'Total_cloud_cover_convective_cloud',
            'ghi_raw': ('Downward_Short-Wave_Radiation_Flux_'
                        'surface_Mixed_intervals_Average')}

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super().__init__(model_type, model, set_type,
                                  vert_level=100000)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """
        data = super().process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.uv_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class HRRR_ESRL(ForecastModel):                                 # noqa: N801
    """
    Subclass of the ForecastModel class representing
    NOAA/GSD/ESRL's HRRR forecast model.
    This is not an operational product.

    Model data corresponds to NOAA/GSD/ESRL HRRR CONUS 3km resolution
    surface forecasts.

    Parameters
    ----------
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    def __init__(self, set_type='best'):
        warnings.warn('HRRR_ESRL is an experimental model and is not '
                      'always available.')

        model_type = 'Forecast Model Data'
        model = 'GSD HRRR CONUS 3km surface'

        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            # 'temp_air': 'Temperature_height_above_ground',  # GH 702
            # 'wind_speed_u': 'u-component_of_wind_height_above_ground',
            # 'wind_speed_v': 'v-component_of_wind_height_above_ground',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_UnknownLevelType-214',
            'mid_clouds': 'Medium_cloud_cover_UnknownLevelType-224',
            'high_clouds': 'High_cloud_cover_UnknownLevelType-234',
            'ghi_raw': 'Downward_short-wave_radiation_flux_surface', }

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi_raw',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super().__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """

        data = super().process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        # data['wind_speed'] = self.uv_to_speed(data)  # GH 702
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class NAM(ForecastModel):
    """
    Subclass of the ForecastModel class representing NAM
    forecast model.

    Model data corresponds to NAM CONUS 12km resolution forecasts
    from CONDUIT.

    Parameters
    ----------
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'

        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere_single_layer',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud',
            'ghi_raw': 'Downward_Short-Wave_Radiation_Flux_surface', }

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super().__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """

        data = super().process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class HRRR(ForecastModel):
    """
    Subclass of the ForecastModel class representing HRRR
    forecast model.

    Model data corresponds to NCEP HRRR CONUS 2.5km resolution
    forecasts.

    Parameters
    ----------
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NCEP HRRR CONUS 2.5km'

        self.variables = {
            'temp_air': 'Temperature_height_above_ground',
            'pressure': 'Pressure_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'wind_speed_u': 'u-component_of_wind_height_above_ground',
            'wind_speed_v': 'v-component_of_wind_height_above_ground',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud'}

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds', ]

        super().__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """
        data = super().process_data(data, **kwargs)
        wind_mapping = {
            'wind_speed_u': 'u-component_of_wind_height_above_ground_0',
            'wind_speed_v': 'v-component_of_wind_height_above_ground_0',
        }
        data = self.rename(data, variables=wind_mapping)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.uv_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        data = data.iloc[:-1, :]  # issue with last point
        return data[self.output_variables]


class NDFD(ForecastModel):
    """
    Subclass of the ForecastModel class representing NDFD forecast
    model.

    Model data corresponds to NWS CONUS CONDUIT forecasts.

    Parameters
    ----------
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    def __init__(self, set_type='best'):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        self.variables = {
            'temp_air': 'Temperature_height_above_ground',
            'wind_speed': 'Wind_speed_height_above_ground',
            'total_clouds': 'Total_cloud_cover_surface', }
        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds', ]
        super().__init__(model_type, model, set_type)

    def process_data(self, data, **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """

        cloud_cover = 'total_clouds'
        data = super().process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class RAP(ForecastModel):
    """
    Subclass of the ForecastModel class representing RAP forecast model.

    Model data corresponds to Rapid Refresh CONUS 20km resolution
    forecasts.

    Parameters
    ----------
    resolution: string or int, default '20'
        The model resolution, either '20' or '40' (km)
    set_type: string, default 'best'
        Type of model to pull data from.

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    _resolutions = ['20', '40']

    def __init__(self, resolution='20', set_type='best'):

        resolution = str(resolution)
        if resolution not in self._resolutions:
            raise ValueError(f'resolution must in {self._resolutions}')

        model_type = 'Forecast Model Data'
        model = f'Rapid Refresh CONUS {resolution}km'
        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud', }
        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds', ]
        super().__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """

        data = super().process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]
