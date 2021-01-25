"""
The ``pvsystem`` module contains functions for modeling the output and
performance of PV modules and inverters.
"""

from collections import OrderedDict
import functools
import io
import os
from urllib.request import urlopen
import warnings
import numpy as np
import pandas as pd

from pvlib._deprecation import deprecated

from pvlib import (atmosphere, iam, inverter, irradiance,
                   singlediode as _singlediode, temperature)
from pvlib.tools import _build_kwargs
from pvlib._deprecation import pvlibDeprecationWarning


# a dict of required parameter names for each DC power model
_DC_MODEL_PARAMS = {
    'sapm': {
        'A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3',
        'B4', 'B5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'C7', 'Isco', 'Impo', 'Voco', 'Vmpo', 'Aisc', 'Aimp', 'Bvoco',
        'Mbvoc', 'Bvmpo', 'Mbvmp', 'N', 'Cells_in_Series',
        'IXO', 'IXXO', 'FD'},
    'desoto': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s'},
    'cec': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s', 'Adjust'},
    'pvsyst': {
        'gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_sh_0', 'R_s', 'alpha_sc', 'EgRef',
        'cells_in_series'},
    'singlediode': {
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s'},
    'pvwatts': {'pdc0', 'gamma_pdc'}
}


def _unwrap_single_value(func):
    """Decorator for functions that return iterables.

    If the length of the iterable returned by `func` is 1, then
    the single member of the iterable is returned. If the length is
    greater than 1, then entire iterable is returned.

    Adds 'unwrap' as a keyword argument that can be set to False
    to force the return value to be a tuple, regardless of its length.
    """
    @functools.wraps(func)
    def f(*args, **kwargs):
        unwrap = kwargs.pop('unwrap', True)
        x = func(*args, **kwargs)
        if unwrap and len(x) == 1:
            return x[0]
        return x
    return f


# not sure if this belongs in the pvsystem module.
# maybe something more like core.py? It may eventually grow to
# import a lot more functionality from other modules.
class PVSystem:
    """
    The PVSystem class defines a standard set of PV system attributes
    and modeling functions. This class describes the collection and
    interactions of PV system components rather than an installed system
    on the ground. It is typically used in combination with
    :py:class:`~pvlib.location.Location` and
    :py:class:`~pvlib.modelchain.ModelChain`
    objects.

    The class supports basic system topologies consisting of:

        * `N` total modules arranged in series
          (`modules_per_string=N`, `strings_per_inverter=1`).
        * `M` total modules arranged in parallel
          (`modules_per_string=1`, `strings_per_inverter=M`).
        * `NxM` total modules arranged in `M` strings of `N` modules each
          (`modules_per_string=N`, `strings_per_inverter=M`).

    The class is complementary to the module-level functions.

    The attributes should generally be things that don't change about
    the system, such the type of module and the inverter. The instance
    methods accept arguments for things that do change, such as
    irradiance and temperature.

    Parameters
    ----------
    arrays : iterable of Array, optional
        List of arrays that are part of the system. If not specified
        a single array is created from the other parameters (e.g.
        `surface_tilt`, `surface_azimuth`). If `arrays` is specified
        the following parameters are ignored:

        - `surface_tilt`
        - `surface_azimuth`
        - `albedo`
        - `surface_type`
        - `module`
        - `module_type`
        - `module_parameters`
        - `temperature_model_parameters`
        - `modules_per_string`
        - `strings_per_inverter`

    surface_tilt: float or array-like, default 0
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth: float or array-like, default 180
        Azimuth angle of the module surface.
        North=0, East=90, South=180, West=270.

    albedo : None or float, default None
        The ground albedo. If ``None``, will attempt to use
        ``surface_type`` and ``irradiance.SURFACE_ALBEDOS``
        to lookup albedo.

    surface_type : None or string, default None
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS``
        for valid values.

    module : None or string, default None
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_type : None or string, default 'glass_polymer'
         Describes the module's construction. Valid strings are 'glass_polymer'
         and 'glass_glass'. Used for cell and module temperature calculations.

    module_parameters : None, dict or Series, default None
        Module parameters as defined by the SAPM, CEC, or other.

    temperature_model_parameters : None, dict or Series, default None.
        Temperature model parameters as defined by the SAPM, Pvsyst, or other.

    modules_per_string: int or float, default 1
        See system topology discussion above.

    strings_per_inverter: int or float, default 1
        See system topology discussion above.

    inverter : None or string, default None
        The model name of the inverters.
        May be used to look up the inverter_parameters dictionary
        via some other method.

    inverter_parameters : None, dict or Series, default None
        Inverter parameters as defined by the SAPM, CEC, or other.

    racking_model : None or string, default 'open_rack'
        Valid strings are 'open_rack', 'close_mount', and 'insulated_back'.
        Used to identify a parameter set for the SAPM cell temperature model.

    losses_parameters : None, dict or Series, default None
        Losses parameters as defined by PVWatts or other.

    name : None or string, default None

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.

    See also
    --------
    pvlib.location.Location
    pvlib.tracking.SingleAxisTracker
    """

    def __init__(self,
                 arrays=None,
                 surface_tilt=0, surface_azimuth=180,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 modules_per_string=1, strings_per_inverter=1,
                 inverter=None, inverter_parameters=None,
                 racking_model=None, losses_parameters=None, name=None):

        if arrays is None:
            self.arrays = (Array(
                surface_tilt,
                surface_azimuth,
                albedo,
                surface_type,
                module,
                module_type,
                module_parameters,
                temperature_model_parameters,
                modules_per_string,
                strings_per_inverter,
                racking_model
            ),)
        else:
            self.arrays = tuple(arrays)

        self.inverter = inverter
        if inverter_parameters is None:
            self.inverter_parameters = {}
        else:
            self.inverter_parameters = inverter_parameters

        if losses_parameters is None:
            self.losses_parameters = {}
        else:
            self.losses_parameters = losses_parameters

        self.name = name

    def __repr__(self):
        repr = f'PVSystem:\n  name: {self.name}\n  '
        for array in self.arrays:
            repr += '\n  '.join(array.__repr__().split('\n'))
            repr += '\n  '
        repr += f'inverter: {self.inverter}'
        return repr

    def _validate_per_array(self, values, system_wide=False):
        """Check that `values` is a tuple of the same length as
        `self.arrays`.

        If `values` is not a tuple it is packed in to a length-1 tuple before
        the check. If the lengths are not the same a ValueError is raised,
        otherwise the tuple `values` is returned.

        When `system_wide` is True and `values` is not a tuple, `values`
        is replicated to a tuple of the same length as `self.arrays` and that
        tuple is returned.
        """
        if system_wide and not isinstance(values, tuple):
            return (values,) * self.num_arrays
        if not isinstance(values, tuple):
            values = (values,)
        if len(values) != len(self.arrays):
            raise ValueError("Length mismatch for per-array parameter")
        return values

    @_unwrap_single_value
    def _infer_cell_type(self):

        """
        Examines module_parameters and maps the Technology key for the CEC
        database and the Material key for the Sandia database to a common
        list of strings for cell type.

        Returns
        -------
        cell_type: str
        """
        return tuple(array._infer_cell_type() for array in self.arrays)

    @_unwrap_single_value
    def get_aoi(self, solar_zenith, solar_azimuth):
        """Get the angle of incidence on the Array(s) in the system.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.

        Returns
        -------
        aoi : Series or tuple of Series
            The angle of incidence
        """

        return tuple(array.get_aoi(solar_zenith, solar_azimuth)
                     for array in self.arrays)

    @_unwrap_single_value
    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, model='haydavies',
                       **kwargs):
        """
        Uses the :py:func:`irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components on a tilted
        surface defined by ``self.surface_tilt``,
        ``self.surface_azimuth``, and ``self.albedo``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        dni : float or Series or tuple of float or Series
            Direct Normal Irradiance
        ghi : float or Series or tuple of float or Series
            Global horizontal irradiance
        dhi : float or Series or tuple of float or Series
            Diffuse horizontal irradiance
        dni_extra : None, float or Series, default None
            Extraterrestrial direct normal irradiance
        airmass : None, float or Series, default None
            Airmass
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to :func:`irradiance.get_total_irradiance`.

        Notes
        -----
        Each of `dni`, `ghi`, and `dni` parameters may be passed as a tuple
        to provide different irradiance for each array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.

        Returns
        -------
        poa_irradiance : DataFrame or tuple of DataFrame
            Column names are: ``total, beam, sky, ground``.
        """
        dni = self._validate_per_array(dni, system_wide=True)
        ghi = self._validate_per_array(ghi, system_wide=True)
        dhi = self._validate_per_array(dhi, system_wide=True)
        return tuple(
            array.get_irradiance(solar_zenith, solar_azimuth,
                                 dni, ghi, dhi,
                                 dni_extra, airmass, model,
                                 **kwargs)
            for array, dni, ghi, dhi in zip(
                self.arrays, dni, ghi, dhi
            )
        )

    @_unwrap_single_value
    def get_iam(self, aoi, iam_model='physical'):
        """
        Determine the incidence angle modifier using the method specified by
        ``iam_model``.

        Parameters for the selected IAM model are expected to be in
        ``PVSystem.module_parameters``. Default parameters are available for
        the 'physical', 'ashrae' and 'martin_ruiz' models.

        Parameters
        ----------
        aoi : numeric or tuple of numeric
            The angle of incidence in degrees.

        aoi_model : string, default 'physical'
            The IAM model to be used. Valid strings are 'physical', 'ashrae',
            'martin_ruiz' and 'sapm'.
        Returns
        -------
        iam : numeric or tuple of numeric
            The AOI modifier.

        Raises
        ------
        ValueError
            if `iam_model` is not a valid model name.
        """
        aoi = self._validate_per_array(aoi)
        return tuple(array.get_iam(aoi, iam_model)
                     for array, aoi in zip(self.arrays, aoi))

    @_unwrap_single_value
    def calcparams_desoto(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`calcparams_desoto` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        **kwargs
            See pvsystem.calcparams_desoto for details

        Returns
        -------
        See pvsystem.calcparams_desoto for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
             'R_s', 'alpha_sc', 'EgRef', 'dEgdT',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_desoto(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def calcparams_cec(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`calcparams_cec` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        **kwargs
            See pvsystem.calcparams_cec for details

        Returns
        -------
        See pvsystem.calcparams_cec for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
             'R_s', 'alpha_sc', 'Adjust', 'EgRef', 'dEgdT',
             'irrad_ref', 'temp_ref']
        )

        return tuple(
            calcparams_cec(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def calcparams_pvsyst(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`calcparams_pvsyst` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.calcparams_pvsyst for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        build_kwargs = functools.partial(
            _build_kwargs,
            ['gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
             'R_sh_ref', 'R_sh_0', 'R_sh_exp',
             'R_s', 'alpha_sc', 'EgRef',
             'irrad_ref', 'temp_ref',
             'cells_in_series']
        )

        return tuple(
            calcparams_pvsyst(
                effective_irradiance, temp_cell,
                **build_kwargs(array.module_parameters)
            )
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def sapm(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`sapm` function, the input parameters,
        and ``self.module_parameters`` to calculate
        Voc, Isc, Ix, Ixx, Vmp, and Imp.

        Parameters
        ----------
        effective_irradiance : numeric or tuple of numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series or tuple of float or Series
            The average cell temperature of cells within a module in C.

        kwargs
            See pvsystem.sapm for details

        Returns
        -------
        See pvsystem.sapm for details
        """
        effective_irradiance = self._validate_per_array(effective_irradiance)
        temp_cell = self._validate_per_array(temp_cell)

        return tuple(
            sapm(effective_irradiance, temp_cell, array.module_parameters)
            for array, effective_irradiance, temp_cell
            in zip(self.arrays, effective_irradiance, temp_cell)
        )

    @_unwrap_single_value
    def sapm_celltemp(self, poa_global, temp_air, wind_speed):
        """Uses :py:func:`temperature.sapm_cell` to calculate cell
        temperatures.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric
            Wind speed in m/s at a height of 10 meters.

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)

        build_kwargs = functools.partial(_build_kwargs, ['a', 'b', 'deltaT'])
        return tuple(
            temperature.sapm_cell(
                poa_global, temp_air, wind_speed,
                **build_kwargs(array.temperature_model_parameters)
            )
            for array, poa_global, temp_air, wind_speed in zip(
                self.arrays, poa_global, temp_air, wind_speed
            )
        )

    @_unwrap_single_value
    def sapm_spectral_loss(self, airmass_absolute):
        """
        Use the :py:func:`sapm_spectral_loss` function, the input
        parameters, and ``self.module_parameters`` to calculate F1.

        Parameters
        ----------
        airmass_absolute : numeric
            Absolute airmass.

        Returns
        -------
        F1 : numeric or tuple of numeric
            The SAPM spectral loss coefficient.
        """
        return tuple(
            sapm_spectral_loss(airmass_absolute, array.module_parameters)
            for array in self.arrays
        )

    @_unwrap_single_value
    def sapm_effective_irradiance(self, poa_direct, poa_diffuse,
                                  airmass_absolute, aoi,
                                  reference_irradiance=1000):
        """
        Use the :py:func:`sapm_effective_irradiance` function, the input
        parameters, and ``self.module_parameters`` to calculate
        effective irradiance.

        Parameters
        ----------
        poa_direct : numeric or tuple of numeric
            The direct irradiance incident upon the module.  [W/m2]

        poa_diffuse : numeric or tuple of numeric
            The diffuse irradiance incident on module.  [W/m2]

        airmass_absolute : numeric
            Absolute airmass. [unitless]

        aoi : numeric or tuple of numeric
            Angle of incidence. [degrees]

        Returns
        -------
        effective_irradiance : numeric or tuple of numeric
            The SAPM effective irradiance. [W/m2]
        """
        poa_direct = self._validate_per_array(poa_direct)
        poa_diffuse = self._validate_per_array(poa_diffuse)
        aoi = self._validate_per_array(aoi)
        return tuple(
            sapm_effective_irradiance(
                poa_direct, poa_diffuse, airmass_absolute, aoi,
                array.module_parameters)
            for array, poa_direct, poa_diffuse, aoi
            in zip(self.arrays, poa_direct, poa_diffuse, aoi)
        )

    @_unwrap_single_value
    def pvsyst_celltemp(self, poa_global, temp_air, wind_speed=1.0):
        """Uses :py:func:`temperature.pvsyst_cell` to calculate cell
        temperature.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric, default 1.0
            Wind speed in m/s measured at the same height for which the wind
            loss factor was determined.  The default value is 1.0, which is
            the wind speed at module height used to determine NOCT.

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)

        def build_celltemp_kwargs(array):
            return {**_build_kwargs(['eta_m', 'alpha_absorption'],
                                    array.module_parameters),
                    **_build_kwargs(['u_c', 'u_v'],
                                    array.temperature_model_parameters)}
        return tuple(
            temperature.pvsyst_cell(poa_global, temp_air, wind_speed,
                                    **build_celltemp_kwargs(array))
            for array, poa_global, temp_air, wind_speed in zip(
                self.arrays, poa_global, temp_air, wind_speed
            )
        )

    @_unwrap_single_value
    def faiman_celltemp(self, poa_global, temp_air, wind_speed=1.0):
        """
        Use :py:func:`temperature.faiman` to calculate cell temperature.

        Parameters
        ----------
        poa_global : numeric or tuple of numeric
            Total incident irradiance [W/m^2].

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature [C].

        wind_speed : numeric or tuple of numeric, default 1.0
            Wind speed in m/s measured at the same height for which the wind
            loss factor was determined.  The default value 1.0 m/s is the wind
            speed at module height used to determine NOCT. [m/s]

        Returns
        -------
        numeric or tuple of numeric
            values in degrees C.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)
        return tuple(
            temperature.faiman(
                poa_global, temp_air, wind_speed,
                **_build_kwargs(
                    ['u0', 'u1'], array.temperature_model_parameters))
            for array, poa_global, temp_air, wind_speed in zip(
                self.arrays, poa_global, temp_air, wind_speed
            )
        )

    @_unwrap_single_value
    def fuentes_celltemp(self, poa_global, temp_air, wind_speed):
        """
        Use :py:func:`temperature.fuentes` to calculate cell temperature.

        Parameters
        ----------
        poa_global : pandas Series or tuple of Series
            Total incident irradiance [W/m^2]

        temp_air : pandas Series or tuple of Series
            Ambient dry bulb temperature [C]

        wind_speed : pandas Series or tuple of Series
            Wind speed [m/s]

        Returns
        -------
        temperature_cell : Series or tuple of Series
            The modeled cell temperature [C]

        Notes
        -----
        The Fuentes thermal model uses the module surface tilt for convection
        modeling. The SAM implementation of PVWatts hardcodes the surface tilt
        value at 30 degrees, ignoring whatever value is used for irradiance
        transposition. This method defaults to using ``self.surface_tilt``, but
        if you want to match the PVWatts behavior, you can override it by
        including a ``surface_tilt`` value in ``temperature_model_parameters``.

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If not
        passed as a tuple then the same value is used for input to each Array.
        If passed as a tuple the length must be the same as the number of
        Arrays.
        """
        # default to using the Array attribute, but allow user to
        # override with a custom surface_tilt value
        poa_global = self._validate_per_array(poa_global)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)

        def _build_kwargs_fuentes(array):
            kwargs = {'surface_tilt': array.surface_tilt}
            temp_model_kwargs = _build_kwargs([
                'noct_installed', 'module_height', 'wind_height', 'emissivity',
                'absorption', 'surface_tilt', 'module_width', 'module_length'],
                array.temperature_model_parameters)
            kwargs.update(temp_model_kwargs)
            return kwargs
        return tuple(
            temperature.fuentes(
                poa_global, temp_air, wind_speed,
                **_build_kwargs_fuentes(array))
            for array, poa_global, temp_air, wind_speed in zip(
                self.arrays, poa_global, temp_air, wind_speed
            )
        )

    @_unwrap_single_value
    def first_solar_spectral_loss(self, pw, airmass_absolute):

        """
        Use the :py:func:`first_solar_spectral_correction` function to
        calculate the spectral loss modifier. The model coefficients are
        specific to the module's cell type, and are determined by searching
        for one of the following keys in self.module_parameters (in order):

        - 'first_solar_spectral_coefficients' (user-supplied coefficients)
        - 'Technology' - a string describing the cell type, can be read from
          the CEC module parameter database
        - 'Material' - a string describing the cell type, can be read from
          the Sandia module database.

        Parameters
        ----------
        pw : array-like
            atmospheric precipitable water (cm).

        airmass_absolute : array-like
            absolute (pressure corrected) airmass.

        Returns
        -------
        modifier: array-like or tuple of array-like
            spectral mismatch factor (unitless) which can be multiplied
            with broadband irradiance reaching a module's cells to estimate
            effective irradiance, i.e., the irradiance that is converted to
            electrical current.
        """

        def _spectral_correction(array):
            if 'first_solar_spectral_coefficients' in \
                    array.module_parameters.keys():
                coefficients = \
                    array.module_parameters[
                        'first_solar_spectral_coefficients'
                    ]
                module_type = None
            else:
                module_type = array._infer_cell_type()
                coefficients = None

            return atmosphere.first_solar_spectral_correction(
                pw, airmass_absolute,
                module_type, coefficients
            )
        return tuple(_spectral_correction(array) for array in self.arrays)

    def singlediode(self, photocurrent, saturation_current,
                    resistance_series, resistance_shunt, nNsVth,
                    ivcurve_pnts=None):
        """Wrapper around the :py:func:`pvlib.pvsystem.singlediode` function.

        See :py:func:`pvsystem.singlediode` for details
        """
        return singlediode(photocurrent, saturation_current,
                           resistance_series, resistance_shunt, nNsVth,
                           ivcurve_pnts=ivcurve_pnts)

    def i_from_v(self, resistance_shunt, resistance_series, nNsVth, voltage,
                 saturation_current, photocurrent):
        """Wrapper around the :py:func:`pvlib.pvsystem.i_from_v` function.

        See :py:func:`pvsystem.i_from_v` for details
        """
        return i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
                        saturation_current, photocurrent)

    # inverter now specified by self.inverter_parameters
    def snlinverter(self, v_dc, p_dc):
        """Uses :py:func:`pvlib.inverter.sandia` to calculate AC power based on
        ``self.inverter_parameters`` and the input voltage and power.

        See :py:func:`pvlib.inverter.sandia` for details
        """
        return inverter.sandia(v_dc, p_dc, self.inverter_parameters)

    def sandia_multi(self, v_dc, p_dc):
        """Uses :py:func:`pvlib.inverter.sandia_multi` to calculate AC power
        based on ``self.inverter_parameters`` and the input voltage and power.

        The parameters `v_dc` and `p_dc` must be tuples with length equal to
        ``self.num_arrays`` if the system has more than one array.

        See :py:func:`pvlib.inverter.sandia_multi` for details.
        """
        v_dc = self._validate_per_array(v_dc)
        p_dc = self._validate_per_array(p_dc)
        return inverter.sandia_multi(v_dc, p_dc, self.inverter_parameters)

    def adrinverter(self, v_dc, p_dc):
        """Uses :py:func:`pvlib.inverter.adr` to calculate AC power based on
        ``self.inverter_parameters`` and the input voltage and power.

        See :py:func:`pvlib.inverter.adr` for details
        """
        return inverter.adr(v_dc, p_dc, self.inverter_parameters)

    @_unwrap_single_value
    def scale_voltage_current_power(self, data):
        """
        Scales the voltage, current, and power of the `data` DataFrame
        by `self.modules_per_string` and `self.strings_per_inverter`.

        Parameters
        ----------
        data: DataFrame or tuple of DataFrame
            May contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
            'i_sc', 'p_mp'`.

        Returns
        -------
        scaled_data: DataFrame or tuple of DataFrame
            A scaled copy of the input data.
        """
        data = self._validate_per_array(data)
        return tuple(
            scale_voltage_current_power(data,
                                        voltage=array.modules_per_string,
                                        current=array.strings)
            for array, data in zip(self.arrays, data)
        )

    @_unwrap_single_value
    def pvwatts_dc(self, g_poa_effective, temp_cell):
        """
        Calcuates DC power according to the PVWatts model using
        :py:func:`pvlib.pvsystem.pvwatts_dc`, `self.module_parameters['pdc0']`,
        and `self.module_parameters['gamma_pdc']`.

        See :py:func:`pvlib.pvsystem.pvwatts_dc` for details.
        """
        g_poa_effective = self._validate_per_array(g_poa_effective)
        temp_cell = self._validate_per_array(temp_cell)
        return tuple(
            pvwatts_dc(g_poa_effective, temp_cell,
                       array.module_parameters['pdc0'],
                       array.module_parameters['gamma_pdc'],
                       **_build_kwargs(['temp_ref'], array.module_parameters))
            for array, g_poa_effective, temp_cell
            in zip(self.arrays, g_poa_effective, temp_cell)
        )

    def pvwatts_losses(self):
        """
        Calculates DC power losses according the PVwatts model using
        :py:func:`pvlib.pvsystem.pvwatts_losses` and
        ``self.losses_parameters``.

        See :py:func:`pvlib.pvsystem.pvwatts_losses` for details.
        """
        kwargs = _build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
                                'wiring', 'connections', 'lid',
                                'nameplate_rating', 'age', 'availability'],
                               self.losses_parameters)
        return pvwatts_losses(**kwargs)

    def pvwatts_ac(self, pdc):
        """
        Calculates AC power according to the PVWatts model using
        :py:func:`pvlib.inverter.pvwatts`, `self.module_parameters["pdc0"]`,
        and `eta_inv_nom=self.inverter_parameters["eta_inv_nom"]`.

        See :py:func:`pvlib.inverter.pvwatts` for details.
        """
        kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                               self.inverter_parameters)

        return inverter.pvwatts(pdc, self.inverter_parameters['pdc0'],
                                **kwargs)

    def pvwatts_multi(self, p_dc):
        """Uses :py:func:`pvlib.inverter.pvwatts_multi` to calculate AC power
        based on ``self.inverter_parameters`` and the input voltage and power.

        The parameter `p_dc` must be a tuple with length equal to
        ``self.num_arrays`` if the system has more than one array.

        See :py:func:`pvlib.inverter.pvwatts_multi` for details.
        """
        p_dc = self._validate_per_array(p_dc)
        kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                               self.inverter_parameters)
        return inverter.pvwatts_multi(p_dc, self.inverter_parameters['pdc0'],
                                      **kwargs)

    @property
    @_unwrap_single_value
    def module_parameters(self):
        return tuple(array.module_parameters for array in self.arrays)

    @property
    @_unwrap_single_value
    def module(self):
        return tuple(array.module for array in self.arrays)

    @property
    @_unwrap_single_value
    def module_type(self):
        return tuple(array.module_type for array in self.arrays)

    @property
    @_unwrap_single_value
    def temperature_model_parameters(self):
        return tuple(array.temperature_model_parameters
                     for array in self.arrays)

    @temperature_model_parameters.setter
    def temperature_model_parameters(self, value):
        for array in self.arrays:
            array.temperature_model_parameters = value

    @property
    @_unwrap_single_value
    def surface_tilt(self):
        return tuple(array.surface_tilt for array in self.arrays)

    @surface_tilt.setter
    def surface_tilt(self, value):
        for array in self.arrays:
            array.surface_tilt = value

    @property
    @_unwrap_single_value
    def surface_azimuth(self):
        return tuple(array.surface_azimuth for array in self.arrays)

    @surface_azimuth.setter
    def surface_azimuth(self, value):
        for array in self.arrays:
            array.surface_azimuth = value

    @property
    @_unwrap_single_value
    def albedo(self):
        return tuple(array.albedo for array in self.arrays)

    @property
    @_unwrap_single_value
    def racking_model(self):
        return tuple(array.racking_model for array in self.arrays)

    @racking_model.setter
    def racking_model(self, value):
        for array in self.arrays:
            array.racking_model = value

    @property
    @_unwrap_single_value
    def modules_per_string(self):
        return tuple(array.modules_per_string for array in self.arrays)

    @property
    @_unwrap_single_value
    def strings_per_inverter(self):
        return tuple(array.strings for array in self.arrays)

    @property
    def num_arrays(self):
        """The number of Arrays in the system."""
        return len(self.arrays)


class Array:
    """
    An Array is a set of of modules at the same orientation.

    Specifically, an array is defined by tilt, azimuth, the
    module parameters, the number of parallel strings of modules
    and the number of modules on each string.

    Parameters
    ----------
    surface_tilt: float or array-like, default 0
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth: float or array-like, default 180
        Azimuth angle of the module surface.
        North=0, East=90, South=180, West=270.

    albedo : None or float, default None
        The ground albedo. If ``None``, will attempt to use
        ``surface_type`` to look up an albedo value in
        ``irradiance.SURFACE_ALBEDOS``. If a surface albedo
        cannot be found then 0.25 is used.

    surface_type : None or string, default None
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS``
        for valid values.

    module : None or string, default None
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_type : None or string, default None
         Describes the module's construction. Valid strings are 'glass_polymer'
         and 'glass_glass'. Used for cell and module temperature calculations.

    module_parameters : None, dict or Series, default None
        Parameters for the module model, e.g., SAPM, CEC, or other.

    temperature_model_parameters : None, dict or Series, default None.
        Parameters for the module temperature model, e.g., SAPM, Pvsyst, or
        other.

    modules_per_string: int, default 1
        Number of modules per string in the array.

    strings: int, default 1
        Number of parallel strings in the array.

    racking_model : None or string, default None
        Valid strings are 'open_rack', 'close_mount', and 'insulated_back'.
        Used to identify a parameter set for the SAPM cell temperature model.

    """

    def __init__(self,
                 surface_tilt=0, surface_azimuth=180,
                 albedo=None, surface_type=None,
                 module=None, module_type=None,
                 module_parameters=None,
                 temperature_model_parameters=None,
                 modules_per_string=1, strings=1,
                 racking_model=None, name=None):
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

        self.surface_type = surface_type
        if albedo is None:
            self.albedo = irradiance.SURFACE_ALBEDOS.get(surface_type, 0.25)
        else:
            self.albedo = albedo

        self.module = module
        if module_parameters is None:
            self.module_parameters = {}
        else:
            self.module_parameters = module_parameters

        self.module_type = module_type
        self.racking_model = racking_model

        self.strings = strings
        self.modules_per_string = modules_per_string

        if temperature_model_parameters is None:
            self.temperature_model_parameters = \
                self._infer_temperature_model_params()
        else:
            self.temperature_model_parameters = temperature_model_parameters

        self.name = name

    def __repr__(self):
        attrs = ['name', 'surface_tilt', 'surface_azimuth', 'module',
                 'albedo', 'racking_model', 'module_type',
                 'temperature_model_parameters',
                 'strings', 'modules_per_string']
        return 'Array:\n  ' + '\n  '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs
        )

    def _infer_temperature_model_params(self):
        # try to infer temperature model parameters from from racking_model
        # and module_type
        param_set = f'{self.racking_model}_{self.module_type}'
        if param_set in temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']:
            return temperature._temperature_model_params('sapm', param_set)
        elif 'freestanding' in param_set:
            return temperature._temperature_model_params('pvsyst',
                                                         'freestanding')
        elif 'insulated' in param_set:  # after SAPM to avoid confusing keys
            return temperature._temperature_model_params('pvsyst',
                                                         'insulated')
        else:
            return {}

    def _infer_cell_type(self):

        """
        Examines module_parameters and maps the Technology key for the CEC
        database and the Material key for the Sandia database to a common
        list of strings for cell type.

        Returns
        -------
        cell_type: str

        """

        _cell_type_dict = {'Multi-c-Si': 'multisi',
                           'Mono-c-Si': 'monosi',
                           'Thin Film': 'cigs',
                           'a-Si/nc': 'asi',
                           'CIS': 'cigs',
                           'CIGS': 'cigs',
                           '1-a-Si': 'asi',
                           'CdTe': 'cdte',
                           'a-Si': 'asi',
                           '2-a-Si': None,
                           '3-a-Si': None,
                           'HIT-Si': 'monosi',
                           'mc-Si': 'multisi',
                           'c-Si': 'multisi',
                           'Si-Film': 'asi',
                           'EFG mc-Si': 'multisi',
                           'GaAs': None,
                           'a-Si / mono-Si': 'monosi'}

        if 'Technology' in self.module_parameters.keys():
            # CEC module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Technology']]
        elif 'Material' in self.module_parameters.keys():
            # Sandia module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Material']]
        else:
            cell_type = None

        return cell_type

    def get_aoi(self, solar_zenith, solar_azimuth):
        """
        Get the angle of incidence on the array.

        Parameters
        ----------
        solar_zenith : float or Series
            Solar zenith angle.
        solar_azimuth : float or Series
            Solar azimuth angle

        Returns
        -------
        aoi : Series
            Then angle of incidence.
        """
        return irradiance.aoi(self.surface_tilt, self.surface_azimuth,
                              solar_zenith, solar_azimuth)

    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, model='haydavies',
                       **kwargs):
        """
        Get plane of array irradiance components.

        Uses the :py:func:`pvlib.irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components for a surface
        defined by ``self.surface_tilt`` and ``self.surface_azimuth`` with
        albedo ``self.albedo``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        dni : float or Series
            Direct Normal Irradiance
        ghi : float or Series
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance
        dni_extra : None, float or Series, default None
            Extraterrestrial direct normal irradiance
        airmass : None, float or Series, default None
            Airmass
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to
            :py:func:`pvlib.irradiance.get_total_irradiance`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``total, beam, sky, ground``.
        """
        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        return irradiance.get_total_irradiance(self.surface_tilt,
                                               self.surface_azimuth,
                                               solar_zenith, solar_azimuth,
                                               dni, ghi, dhi,
                                               dni_extra=dni_extra,
                                               airmass=airmass,
                                               model=model,
                                               albedo=self.albedo,
                                               **kwargs)

    def get_iam(self, aoi, iam_model='physical'):
        """
        Determine the incidence angle modifier using the method specified by
        ``iam_model``.

        Parameters for the selected IAM model are expected to be in
        ``Array.module_parameters``. Default parameters are available for
        the 'physical', 'ashrae' and 'martin_ruiz' models.

        Parameters
        ----------
        aoi : numeric
            The angle of incidence in degrees.

        aoi_model : string, default 'physical'
            The IAM model to be used. Valid strings are 'physical', 'ashrae',
            'martin_ruiz' and 'sapm'.

        Returns
        -------
        iam : numeric
            The AOI modifier.

        Raises
        ------
        ValueError
            if `iam_model` is not a valid model name.
        """
        model = iam_model.lower()
        if model in ['ashrae', 'physical', 'martin_ruiz']:
            param_names = iam._IAM_MODEL_PARAMS[model]
            kwargs = _build_kwargs(param_names, self.module_parameters)
            func = getattr(iam, model)
            return func(aoi, **kwargs)
        elif model == 'sapm':
            return iam.sapm(aoi, self.module_parameters)
        elif model == 'interp':
            raise ValueError(model + ' is not implemented as an IAM model '
                             'option for Array')
        else:
            raise ValueError(model + ' is not a valid IAM model')


def calcparams_desoto(effective_irradiance, temp_cell,
                      alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                      EgRef=1.121, dEgdT=-0.0002677,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the De Soto et al.
    model described in [1]_. The five values returned by calcparams_desoto
    can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [1]_) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    .. [3] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [4] O. Madelung, "Semiconductors: Data Handbook, 3rd ed." ISBN
       3-540-40488-0

    See Also
    --------
    singlediode
    retrieve_sam

    Notes
    -----
    If the reference parameters in the ModuleParameters struct are read
    from a database or library of parameters (e.g. System Advisor
    Model), it is important to use the same EgRef and dEgdT values that
    were used to generate the reference parameters, regardless of the
    actual bandgap characteristics of the semiconductor. For example, in
    the case of the System Advisor Model library, created as described
    in [3], EgRef and dEgdT for all modules were 1.121 and -0.0002677,
    respectively.

    This table of reference bandgap energies (EgRef), bandgap energy
    temperature dependence (dEgdT), and "typical" airmass response (M)
    is provided purely as reference to those who may generate their own
    reference module parameters (a_ref, IL_ref, I0_ref, etc.) based upon
    the various PV semiconductors. Again, we stress the importance of
    using identical EgRef and dEgdT when generation reference parameters
    and modifying the reference parameters (for irradiance, temperature,
    and airmass) per DeSoto's equations.

     Crystalline Silicon (Si):
         * EgRef = 1.121
         * dEgdT = -0.0002677

         >>> M = np.polyval([-1.26E-4, 2.816E-3, -0.024459, 0.086257, 0.9181],
         ...                AMa) # doctest: +SKIP

         Source: [1]

     Cadmium Telluride (CdTe):
         * EgRef = 1.475
         * dEgdT = -0.0003

         >>> M = np.polyval([-2.46E-5, 9.607E-4, -0.0134, 0.0716, 0.9196],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium diSelenide (CIS):
         * EgRef = 1.010
         * dEgdT = -0.00011

         >>> M = np.polyval([-3.74E-5, 0.00125, -0.01462, 0.0718, 0.9210],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium Gallium diSelenide (CIGS):
         * EgRef = 1.15
         * dEgdT = ????

         >>> M = np.polyval([-9.07E-5, 0.0022, -0.0202, 0.0652, 0.9417],
         ...                AMa) # doctest: +SKIP

         Source: Wikipedia

     Gallium Arsenide (GaAs):
         * EgRef = 1.424
         * dEgdT = -0.000433
         * M = unknown

         Source: [4]
    '''

    # Boltzmann constant in eV/K
    k = 8.617332478e-05

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    E_g = EgRef * (1 + dEgdT*(Tcell_K - Tref_K))

    nNsVth = a_ref * (Tcell_K / Tref_K)

    # In the equation for IL, the single factor effective_irradiance is
    # used, in place of the product S*M in [1]. effective_irradiance is
    # equivalent to the product of S (irradiance reaching a module's cells) *
    # M (spectral adjustment factor) as described in [1].
    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))
    I0 = (I_o_ref * ((Tcell_K / Tref_K) ** 3) *
          (np.exp(EgRef / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
    # Note that the equation for Rsh differs from [1]. In [1] Rsh is given as
    # Rsh = Rsh_ref * (S_ref / S) where S is broadband irradiance reaching
    # the module's cells. If desired this model behavior can be duplicated
    # by applying reflection and soiling losses to broadband plane of array
    # irradiance and not applying a spectral loss modifier, i.e.,
    # spectral_modifier = 1.0.
    # use errstate to silence divide by warning
    with np.errstate(divide='ignore'):
        Rsh = R_sh_ref * (irrad_ref / effective_irradiance)
    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def calcparams_cec(effective_irradiance, temp_cell,
                   alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                   Adjust, EgRef=1.121, dEgdT=-0.0002677,
                   irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the CEC
    model. The CEC model [1]_ differs from the De soto et al.
    model [3]_ by the parameter Adjust. The five values returned by
    calcparams_cec can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [3]) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    .. [2] System Advisor Model web page. https://sam.nrel.gov.

    .. [3] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    See Also
    --------
    calcparams_desoto
    singlediode
    retrieve_sam

    '''

    # pass adjusted temperature coefficient to desoto
    return calcparams_desoto(effective_irradiance, temp_cell,
                             alpha_sc*(1.0 - Adjust/100),
                             a_ref, I_L_ref, I_o_ref,
                             R_sh_ref, R_s,
                             EgRef=1.121, dEgdT=-0.0002677,
                             irrad_ref=1000, temp_ref=25)


def calcparams_pvsyst(effective_irradiance, temp_cell,
                      alpha_sc, gamma_ref, mu_gamma,
                      I_L_ref, I_o_ref,
                      R_sh_ref, R_sh_0, R_s,
                      cells_in_series,
                      R_sh_exp=5.5,
                      EgRef=1.121,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the PVsyst v6
    model.  The PVsyst v6 model is described in [1]_, [2]_, [3]_.
    The five values returned by calcparams_pvsyst can be used by singlediode
    to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    gamma_ref : float
        The diode ideality factor

    mu_gamma : float
        The temperature coefficient for the diode ideality factor, 1/K

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_sh_0 : float
        The shunt resistance at zero irradiance conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    cells_in_series : integer
        The number of cells connected in series.

    R_sh_exp : float
        The exponent in the equation for shunt resistance, unitless. Defaults
        to 5.5.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation current in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    .. [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
       Temperature Dependence of Photovoltaic Modules in PVsyst,
       IEEE Journal of Photovoltaics v5(1), January 2015.

    .. [2] A. Mermoud, PV modules modelling, Presentation at the 2nd PV
       Performance Modeling Workshop, Santa Clara, CA, May 2013

    .. [3] A. Mermoud, T. Lejeune, Performance Assessment of a Simulation Model
       for PV modules of any available technology, 25th European Photovoltaic
       Solar Energy Conference, Valencia, Spain, Sept. 2010

    See Also
    --------
    calcparams_desoto
    singlediode

    '''

    # Boltzmann constant in J/K
    k = 1.38064852e-23

    # elementary charge in coulomb
    q = 1.6021766e-19

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    gamma = gamma_ref + mu_gamma * (Tcell_K - Tref_K)
    nNsVth = gamma * k / q * cells_in_series * Tcell_K

    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))

    I0 = I_o_ref * ((Tcell_K / Tref_K) ** 3) * \
        (np.exp((q * EgRef) / (k * gamma) * (1 / Tref_K - 1 / Tcell_K)))

    Rsh_tmp = \
        (R_sh_ref - R_sh_0 * np.exp(-R_sh_exp)) / (1.0 - np.exp(-R_sh_exp))
    Rsh_base = np.maximum(0.0, Rsh_tmp)

    Rsh = Rsh_base + (R_sh_0 - Rsh_base) * \
        np.exp(-R_sh_exp * effective_irradiance / irrad_ref)

    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def retrieve_sam(name=None, path=None):
    '''
    Retrieve latest module and inverter info from a local file or the
    SAM website.

    This function will retrieve either:

        * CEC module database
        * Sandia Module database
        * CEC Inverter database
        * Anton Driesse Inverter database

    and return it as a pandas DataFrame.

    Parameters
    ----------
    name : None or string, default None
        Name can be one of:

        * 'CECMod' - returns the CEC module database
        * 'CECInverter' - returns the CEC Inverter database
        * 'SandiaInverter' - returns the CEC Inverter database
          (CEC is only current inverter db available; tag kept for
          backwards compatibility)
        * 'SandiaMod' - returns the Sandia Module database
        * 'ADRInverter' - returns the ADR Inverter database

    path : None or string, default None
        Path to the SAM file. May also be a URL.

    Returns
    -------
    samfile : DataFrame
        A DataFrame containing all the elements of the desired database.
        Each column represents a module or inverter, and a specific
        dataset can be retrieved by the command

    Raises
    ------
    ValueError
        If no name or path is provided.

    Notes
    -----
    Files available at
        https://github.com/NREL/SAM/tree/develop/deploy/libraries
    Documentation for module and inverter data sets:
        https://sam.nrel.gov/photovoltaic/pv-sub-page-2.html

    Examples
    --------

    >>> from pvlib import pvsystem
    >>> invdb = pvsystem.retrieve_sam('CECInverter')
    >>> inverter = invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
    >>> inverter
    Vac           277.000000
    Paco         6000.000000
    Pdco         6165.670000
    Vdco          361.123000
    Pso            36.792300
    C0             -0.000002
    C1             -0.000047
    C2             -0.001861
    C3              0.000721
    Pnt             0.070000
    Vdcmax        600.000000
    Idcmax         32.000000
    Mppt_low      200.000000
    Mppt_high     500.000000
    Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64
    '''

    if name is not None:
        name = name.lower()
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        if name == 'cecmod':
            csvdata = os.path.join(
                data_path, 'sam-library-cec-modules-2019-03-05.csv')
        elif name == 'sandiamod':
            csvdata = os.path.join(
                data_path, 'sam-library-sandia-modules-2015-6-30.csv')
        elif name == 'adrinverter':
            csvdata = os.path.join(data_path, 'adr-library-2013-10-01.csv')
        elif name in ['cecinverter', 'sandiainverter']:
            # Allowing either, to provide for old code,
            # while aligning with current expectations
            csvdata = os.path.join(
                data_path, 'sam-library-cec-inverters-2019-03-05.csv')
        else:
            raise ValueError(f'invalid name {name}')
    elif path is not None:
        if path.startswith('http'):
            response = urlopen(path)
            csvdata = io.StringIO(response.read().decode(errors='ignore'))
        else:
            csvdata = path
    elif name is None and path is None:
        raise ValueError("A name or path must be provided!")

    return _parse_raw_sam_df(csvdata)


def _normalize_sam_product_names(names):
    '''
    Replace special characters within the product names to make them more
    suitable for use as Dataframe column names.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    import warnings

    BAD_CHARS = ' -.()[]:+/",'
    GOOD_CHARS = '____________'

    mapping = str.maketrans(BAD_CHARS, GOOD_CHARS)
    names = pd.Series(data=names)
    norm_names = names.str.translate(mapping)

    n_duplicates = names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn('Original names contain %d duplicate(s).' % n_duplicates)

    n_duplicates = norm_names.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn(
            'Normalized names contain %d duplicate(s).' % n_duplicates)

    return norm_names.values


def _parse_raw_sam_df(csvdata):

    df = pd.read_csv(csvdata, index_col=0, skiprows=[1, 2])

    df.columns = df.columns.str.replace(' ', '_')
    df.index = _normalize_sam_product_names(df.index)
    df = df.transpose()

    if 'ADRCoefficients' in df.index:
        ad_ce = 'ADRCoefficients'
        # for each inverter, parses a string of coefficients like
        # ' 1.33, 2.11, 3.12' into a list containing floats:
        # [1.33, 2.11, 3.12]
        df.loc[ad_ce] = df.loc[ad_ce].map(lambda x: list(
            map(float, x.strip(' []').split())))

    return df


def sapm(effective_irradiance, temp_cell, module):
    '''
    The Sandia PV Array Performance Model (SAPM) generates 5 points on a
    PV module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
    SAND2004-3535. Assumes a reference cell temperature of 25 C.

    Parameters
    ----------
    effective_irradiance : numeric
        Irradiance reaching the module's cells, after reflections and
        adjustment for spectrum. [W/m2]

    temp_cell : numeric
        Cell temperature [C].

    module : dict-like
        A dict or Series defining the SAPM parameters. See the notes section
        for more details.

    Returns
    -------
    A DataFrame with the columns:

        * i_sc : Short-circuit current (A)
        * i_mp : Current at the maximum-power point (A)
        * v_oc : Open-circuit voltage (V)
        * v_mp : Voltage at maximum-power point (V)
        * p_mp : Power at maximum-power point (W)
        * i_x : Current at module V = 0.5Voc, defines 4th point on I-V
          curve for modeling curve shape
        * i_xx : Current at module V = 0.5(Voc+Vmp), defines 5th point on
          I-V curve for modeling curve shape

    Notes
    -----
    The SAPM parameters which are required in ``module`` are
    listed in the following table.

    The Sandia module database contains parameter values for a limited set
    of modules. The CEC module database does not contain these parameters.
    Both databases can be accessed using :py:func:`retrieve_sam`.

    ================   ========================================================
    Key                Description
    ================   ========================================================
    A0-A4              The airmass coefficients used in calculating
                       effective irradiance
    B0-B5              The angle of incidence coefficients used in calculating
                       effective irradiance
    C0-C7              The empirically determined coefficients relating
                       Imp, Vmp, Ix, and Ixx to effective irradiance
    Isco               Short circuit current at reference condition (amps)
    Impo               Maximum power current at reference condition (amps)
    Voco               Open circuit voltage at reference condition (amps)
    Vmpo               Maximum power voltage at reference condition (amps)
    Aisc               Short circuit current temperature coefficient at
                       reference condition (1/C)
    Aimp               Maximum power current temperature coefficient at
                       reference condition (1/C)
    Bvoco              Open circuit voltage temperature coefficient at
                       reference condition (V/C)
    Mbvoc              Coefficient providing the irradiance dependence for the
                       BetaVoc temperature coefficient at reference irradiance
                       (V/C)
    Bvmpo              Maximum power voltage temperature coefficient at
                       reference condition
    Mbvmp              Coefficient providing the irradiance dependence for the
                       BetaVmp temperature coefficient at reference irradiance
                       (V/C)
    N                  Empirically determined "diode factor" (dimensionless)
    Cells_in_Series    Number of cells in series in a module's cell string(s)
    IXO                Ix at reference conditions
    IXXO               Ixx at reference conditions
    FD                 Fraction of diffuse irradiance used by module
    ================   ========================================================

    References
    ----------
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    See Also
    --------
    retrieve_sam
    pvlib.temperature.sapm_cell
    pvlib.temperature.sapm_module
    '''

    # TODO: someday, change temp_ref and irrad_ref to reference_temperature and
    # reference_irradiance and expose
    temp_ref = 25
    irrad_ref = 1000

    q = 1.60218e-19  # Elementary charge in units of coulombs
    kb = 1.38066e-23  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    Ee = np.array(effective_irradiance, dtype='float64') / irrad_ref

    # set up masking for 0, positive, and nan inputs
    Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
    Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
    notnan = ~np.isnan(Ee)
    np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
    np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

    Bvmpo = module['Bvmpo'] + module['Mbvmp']*(1 - Ee)
    Bvoco = module['Bvoco'] + module['Mbvoc']*(1 - Ee)
    delta = module['N'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)
    # avoid repeated __getitem__
    cells_in_series = module['Cells_in_Series']

    out = OrderedDict()

    out['i_sc'] = (
        module['Isco'] * Ee * (1 + module['Aisc']*(temp_cell - temp_ref)))

    out['i_mp'] = (
        module['Impo'] * (module['C0']*Ee + module['C1']*(Ee**2)) *
        (1 + module['Aimp']*(temp_cell - temp_ref)))

    out['v_oc'] = np.maximum(0, (
        module['Voco'] + cells_in_series * delta * logEe +
        Bvoco*(temp_cell - temp_ref)))

    out['v_mp'] = np.maximum(0, (
        module['Vmpo'] +
        module['C2'] * cells_in_series * delta * logEe +
        module['C3'] * cells_in_series * ((delta * logEe) ** 2) +
        Bvmpo*(temp_cell - temp_ref)))

    out['p_mp'] = out['i_mp'] * out['v_mp']

    out['i_x'] = (
        module['IXO'] * (module['C4']*Ee + module['C5']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    # the Ixx calculation in King 2004 has a typo (mixes up Aisc and Aimp)
    out['i_xx'] = (
        module['IXXO'] * (module['C6']*Ee + module['C7']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - temp_ref)))

    if isinstance(out['i_sc'], pd.Series):
        out = pd.DataFrame(out)

    return out


def sapm_spectral_loss(airmass_absolute, module):
    """
    Calculates the SAPM spectral loss coefficient, F1.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    F1 : numeric
        The SAPM spectral loss coefficient.

    Notes
    -----
    nan airmass values will result in 0 output.
    """

    am_coeff = [module['A4'], module['A3'], module['A2'], module['A1'],
                module['A0']]

    spectral_loss = np.polyval(am_coeff, airmass_absolute)

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

    spectral_loss = np.maximum(0, spectral_loss)

    if isinstance(airmass_absolute, pd.Series):
        spectral_loss = pd.Series(spectral_loss, airmass_absolute.index)

    return spectral_loss


def sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi,
                              module):
    r"""
    Calculates the SAPM effective irradiance using the SAPM spectral
    loss and SAPM angle of incidence loss functions.

    Parameters
    ----------
    poa_direct : numeric
        The direct irradiance incident upon the module. [W/m2]

    poa_diffuse : numeric
        The diffuse irradiance incident on module.  [W/m2]

    airmass_absolute : numeric
        Absolute airmass. [unitless]

    aoi : numeric
        Angle of incidence. [degrees]

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    effective_irradiance : numeric
        Effective irradiance accounting for reflections and spectral content.
        [W/m2]

    Notes
    -----
    The SAPM model for effective irradiance [1]_ translates broadband direct
    and diffuse irradiance on the plane of array to the irradiance absorbed by
    a module's cells.

    The model is
    .. math::

        `Ee = f_1(AM_a) (E_b f_2(AOI) + f_d E_d)`

    where :math:`Ee` is effective irradiance (W/m2), :math:`f_1` is a fourth
    degree polynomial in air mass :math:`AM_a`, :math:`E_b` is beam (direct)
    irradiance on the plane of array, :math:`E_d` is diffuse irradiance on the
    plane of array, :math:`f_2` is a fifth degree polynomial in the angle of
    incidence :math:`AOI`, and :math:`f_d` is the fraction of diffuse
    irradiance on the plane of array that is not reflected away.

    References
    ----------
    .. [1] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    See also
    --------
    pvlib.iam.sapm
    pvlib.pvsystem.sapm_spectral_loss
    pvlib.pvsystem.sapm
    """

    F1 = sapm_spectral_loss(airmass_absolute, module)
    F2 = iam.sapm(aoi, module)

    Ee = F1 * (poa_direct * F2 + module['FD'] * poa_diffuse)

    return Ee


def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None,
                method='lambertw'):
    r"""
    Solve the single-diode equation to obtain a photovoltaic IV curve.

    Solves the single diode equation [1]_

    .. math::

        I = I_L -
            I_0 \left[
                \exp \left(\frac{V+I R_s}{n N_s V_{th}} \right)-1
            \right] -
            \frac{V + I R_s}{R_{sh}}

    for :math:`I` and :math:`V` when given :math:`I_L, I_0, R_s, R_{sh},` and
    :math:`n N_s V_{th}` which are described later. Returns a DataFrame
    which contains the 5 points on the I-V curve specified in
    [3]_. If all :math:`I_L, I_0, R_s, R_{sh},` and
    :math:`n N_s V_{th}` are scalar, a single curve is returned, if any
    are Series (of the same length), multiple IV curves are calculated.

    The input parameters can be calculated from meteorological data using a
    function for a single diode model, e.g.,
    :py:func:`~pvlib.pvsystem.calcparams_desoto`.

    Parameters
    ----------
    photocurrent : numeric
        Light-generated current :math:`I_L` (photocurrent)
        ``0 <= photocurrent``. [A]

    saturation_current : numeric
        Diode saturation :math:`I_0` current under desired IV curve
        conditions. ``0 < saturation_current``. [A]

    resistance_series : numeric
        Series resistance :math:`R_s` under desired IV curve conditions.
        ``0 <= resistance_series < numpy.inf``.  [ohm]

    resistance_shunt : numeric
        Shunt resistance :math:`R_{sh}` under desired IV curve conditions.
        ``0 < resistance_shunt <= numpy.inf``.  [ohm]

    nNsVth : numeric
        The product of three components: 1) the usual diode ideality factor
        :math:`n`, 2) the number of cells in series :math:`N_s`, and 3)
        the cell thermal voltage
        :math:`V_{th}`. The thermal voltage of the cell (in volts) may be
        calculated as :math:`k_B T_c / q`, where :math:`k_B` is
        Boltzmann's constant (J/K), :math:`T_c` is the temperature of the p-n
        junction in Kelvin, and :math:`q` is the charge of an electron
        (coulombs). ``0 < nNsVth``.  [V]

    ivcurve_pnts : None or int, default None
        Number of points in the desired IV curve. If None or 0, no points on
        the IV curves will be produced.

    method : str, default 'lambertw'
        Determines the method used to calculate points on the IV curve. The
        options are ``'lambertw'``, ``'newton'``, or ``'brentq'``.

    Returns
    -------
    OrderedDict or DataFrame

    The returned dict-like object always contains the keys/columns:

        * i_sc - short circuit current in amperes.
        * v_oc - open circuit voltage in volts.
        * i_mp - current at maximum power point in amperes.
        * v_mp - voltage at maximum power point in volts.
        * p_mp - power at maximum power point in watts.
        * i_x - current, in amperes, at ``v = 0.5*v_oc``.
        * i_xx - current, in amperes, at ``V = 0.5*(v_oc+v_mp)``.

    If ivcurve_pnts is greater than 0, the output dictionary will also
    include the keys:

        * i - IV curve current in amperes.
        * v - IV curve voltage in volts.

    The output will be an OrderedDict if photocurrent is a scalar,
    array, or ivcurve_pnts is not None.

    The output will be a DataFrame if photocurrent is a Series and
    ivcurve_pnts is None.

    See also
    --------
    calcparams_desoto
    calcparams_cec
    calcparams_pvsyst
    sapm
    pvlib.singlediode.bishop88

    Notes
    -----
    If the method is ``'lambertw'`` then the solution employed to solve the
    implicit diode equation utilizes the Lambert W function to obtain an
    explicit function of :math:`V=f(I)` and :math:`I=f(V)` as shown in [2]_.

    If the method is ``'newton'`` then the root-finding Newton-Raphson method
    is used. It should be safe for well behaved IV-curves, but the ``'brentq'``
    method is recommended for reliability.

    If the method is ``'brentq'`` then Brent's bisection search method is used
    that guarantees convergence by bounding the voltage between zero and
    open-circuit.

    If the method is either ``'newton'`` or ``'brentq'`` and ``ivcurve_pnts``
    are indicated, then :func:`pvlib.singlediode.bishop88` [4]_ is used to
    calculate the points on the IV curve points at diode voltages from zero to
    open-circuit voltage with a log spacing that gets closer as voltage
    increases. If the method is ``'lambertw'`` then the calculated points on
    the IV curve are linearly spaced.

    References
    ----------
    .. [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
       0 86758 909 4

    .. [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.

    .. [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
       SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    .. [4] "Computer simulation of the effects of electrical mismatches in
       photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
       https://doi.org/10.1016/0379-6787(88)90059-2
    """
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _singlediode._lambertw(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, ivcurve_pnts
        )
        i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx = out[:7]
        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)  # collect args
        v_oc = _singlediode.bishop88_v_from_i(
            0.0, *args, method=method.lower()
        )
        i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
            *args, method=method.lower()
        )
        i_sc = _singlediode.bishop88_i_from_v(
            0.0, *args, method=method.lower()
        )
        i_x = _singlediode.bishop88_i_from_v(
            v_oc / 2.0, *args, method=method.lower()
        )
        i_xx = _singlediode.bishop88_i_from_v(
            (v_oc + v_mp) / 2.0, *args, method=method.lower()
        )

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                (11.0 - np.logspace(np.log10(11.0), 0.0, ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = _singlediode.bishop88(vd, *args)

    out = OrderedDict()
    out['i_sc'] = i_sc
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = i_x
    out['i_xx'] = i_xx

    if ivcurve_pnts:

        out['v'] = ivcurve_v
        out['i'] = ivcurve_i

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)

    return out


def max_power_point(photocurrent, saturation_current, resistance_series,
                    resistance_shunt, nNsVth, d2mutau=0, NsVbi=np.Inf,
                    method='brentq'):
    """
    Given the single diode equation coefficients, calculates the maximum power
    point (MPP).

    Parameters
    ----------
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode reverse saturation current [A]
    resistance_series : numeric
        series resitance [ohms]
    resistance_shunt : numeric
        shunt resitance [ohms]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of serices cells ``Ns``
    d2mutau : numeric, default 0
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that accounts for recombination current in the
        intrinsic layer. The value is the ratio of intrinsic layer thickness
        squared :math:`d^2` to the diffusion length of charge carriers
        :math:`\\mu \\tau`. [V]
    NsVbi : numeric, default np.inf
        PVsyst parameter for cadmium-telluride (CdTe) and amorphous-silicon
        (a-Si) modules that is the product of the PV module number of series
        cells ``Ns`` and the builtin voltage ``Vbi`` of the intrinsic layer.
        [V].
    method : str
        either ``'newton'`` or ``'brentq'``

    Returns
    -------
    OrderedDict or pandas.Datafrane
        ``(i_mp, v_mp, p_mp)``

    Notes
    -----
    Use this function when you only want to find the maximum power point. Use
    :func:`singlediode` when you need to find additional points on the IV
    curve. This function uses Brent's method by default because it is
    guaranteed to converge.
    """
    i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
        photocurrent, saturation_current, resistance_series,
        resistance_shunt, nNsVth, d2mutau=0, NsVbi=np.Inf,
        method=method.lower()
    )
    if isinstance(photocurrent, pd.Series):
        ivp = {'i_mp': i_mp, 'v_mp': v_mp, 'p_mp': p_mp}
        out = pd.DataFrame(ivp, index=photocurrent.index)
    else:
        out = OrderedDict()
        out['i_mp'] = i_mp
        out['v_mp'] = v_mp
        out['p_mp'] = p_mp
    return out


def v_from_i(resistance_shunt, resistance_series, nNsVth, current,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device voltage at the given device current for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
    Jain and Kapoor 2004 [1]_.
    The solution is per Eq 3 of [1]_ except when resistance_shunt=numpy.inf,
    in which case the explict solution for voltage is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
    resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
    the caller's responsibility to ensure that the arguments are all float64
    and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    current : numeric
        The current in amperes under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    .. [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_v_from_i(
            resistance_shunt, resistance_series, nNsVth, current,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (current, photocurrent, saturation_current,
                resistance_series, resistance_shunt, nNsVth)
        V = _singlediode.bishop88_v_from_i(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                V = np.tile(V, shape)
        if np.isnan(V).any() and size <= 1:
            V = np.repeat(V, size)
            if shape is not None:
                V = V.reshape(shape)
        return V


def i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device current at the given device voltage for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
     Jain and Kapoor 2004 [1]_.
    The solution is per Eq 2 of [1] except when resistance_series=0,
     in which case the explict solution for current is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
     resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
     the caller's responsibility to ensure that the arguments are all float64
     and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    voltage : numeric
        The voltage in Volts under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    .. [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
       parameters of real solar cells using Lambert W-function", Solar
       Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_i_from_v(
            resistance_shunt, resistance_series, nNsVth, voltage,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (voltage, photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)
        current = _singlediode.bishop88_i_from_v(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                current = np.tile(current, shape)
        if np.isnan(current).any() and size <= 1:
            current = np.repeat(current, size)
            if shape is not None:
                current = current.reshape(shape)
        return current


def scale_voltage_current_power(data, voltage=1, current=1):
    """
    Scales the voltage, current, and power in data by the voltage
    and current factors.

    Parameters
    ----------
    data: DataFrame
        May contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
        'i_sc', 'p_mp'`.
    voltage: numeric, default 1
        The amount by which to multiply the voltages.
    current: numeric, default 1
        The amount by which to multiply the currents.

    Returns
    -------
    scaled_data: DataFrame
        A scaled copy of the input data.
        `'p_mp'` is scaled by `voltage * current`.
    """

    # as written, only works with a DataFrame
    # could make it work with a dict, but it would be more verbose
    voltage_keys = ['v_mp', 'v_oc']
    current_keys = ['i_mp', 'i_x', 'i_xx', 'i_sc']
    power_keys = ['p_mp']
    voltage_df = data.filter(voltage_keys, axis=1) * voltage
    current_df = data.filter(current_keys, axis=1) * current
    power_df = data.filter(power_keys, axis=1) * voltage * current
    df = pd.concat([voltage_df, current_df, power_df], axis=1)
    df_sorted = df[data.columns]  # retain original column order
    return df_sorted


def pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.):
    r"""
    Implements NREL's PVWatts DC power model. The PVWatts DC model [1]_ is:

    .. math::

        P_{dc} = \frac{G_{poa eff}}{1000} P_{dc0} ( 1 + \gamma_{pdc} (T_{cell} - T_{ref}))

    Note that the pdc0 is also used as a symbol in
    :py:func:`pvlib.inverter.pvwatts`. pdc0 in this function refers to the DC
    power of the modules at reference conditions. pdc0 in
    :py:func:`pvlib.inverter.pvwatts` refers to the DC power input limit of
    the inverter.

    Parameters
    ----------
    g_poa_effective: numeric
        Irradiance transmitted to the PV cells. To be
        fully consistent with PVWatts, the user must have already
        applied angle of incidence losses, but not soiling, spectral,
        etc. [W/m^2]
    temp_cell: numeric
        Cell temperature [C].
    pdc0: numeric
        Power of the modules at 1000 W/m^2 and cell reference temperature. [W]
    gamma_pdc: numeric
        The temperature coefficient of power. Typically -0.002 to
        -0.005 per degree C. [1/C]
    temp_ref: numeric, default 25.0
        Cell reference temperature. PVWatts defines it to be 25 C and
        is included here for flexibility. [C]

    Returns
    -------
    pdc: numeric
        DC power.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """  # noqa: E501

    pdc = (g_poa_effective * 0.001 * pdc0 *
           (1 + gamma_pdc * (temp_cell - temp_ref)))

    return pdc


def pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2,
                   connections=0.5, lid=1.5, nameplate_rating=1, age=0,
                   availability=3):
    r"""
    Implements NREL's PVWatts system loss model.
    The PVWatts loss model [1]_ is:

    .. math::

        L_{total}(\%) = 100 [ 1 - \Pi_i ( 1 - \frac{L_i}{100} ) ]

    All parameters must be in units of %. Parameters may be
    array-like, though all array sizes must match.

    Parameters
    ----------
    soiling: numeric, default 2
    shading: numeric, default 3
    snow: numeric, default 0
    mismatch: numeric, default 2
    wiring: numeric, default 2
    connections: numeric, default 0.5
    lid: numeric, default 1.5
        Light induced degradation
    nameplate_rating: numeric, default 1
    age: numeric, default 0
    availability: numeric, default 3

    Returns
    -------
    losses: numeric
        System losses in units of %.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    params = [soiling, shading, snow, mismatch, wiring, connections, lid,
              nameplate_rating, age, availability]

    # manually looping over params allows for numpy/pandas to handle any
    # array-like broadcasting that might be necessary.
    perf = 1
    for param in params:
        perf *= 1 - param/100

    losses = (1 - perf) * 100.

    return losses


def combine_loss_factors(index, *losses, fill_method='ffill'):
    r"""
    Combines Series loss fractions while setting a common index.

    The separate losses are compounded using the following equation:

    .. math::

        L_{total} = 1 - [ 1 - \Pi_i ( 1 - L_i ) ]

    :math:`L_{total}` is the total loss returned
    :math:`L_i` is each individual loss factor input

    Note the losses must each be a series with a DatetimeIndex.
    All losses will be resampled to match the index parameter using
    the fill method specified (defaults to "fill forward").

    Parameters
    ----------
    index : DatetimeIndex
        The index of the returned loss factors

    *losses : Series
        One or more Series of fractions to be compounded

    fill_method : {'ffill', 'bfill', 'nearest'}, default 'ffill'
        Method to use for filling holes in reindexed DataFrame

    Returns
    -------
    Series
        Fractions resulting from the combination of each loss factor
    """
    combined_factor = 1

    for loss in losses:
        loss = loss.reindex(index, method=fill_method)
        combined_factor *= (1 - loss)

    return 1 - combined_factor
