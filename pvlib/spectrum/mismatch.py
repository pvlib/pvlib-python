"""
The ``mismatch`` module provides functions for spectral mismatch calculations.
"""

import pvlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from warnings import warn


def get_example_spectral_response(wavelength=None):
    '''
    Generate a generic smooth spectral response (SR) for tests and experiments.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which spectral response values are generated.
        By default ``wavelength`` is from 280 to 1200 in 5 nm intervals. [nm]

    Returns
    -------
    spectral_response : pandas.Series
        The relative spectral response indexed by ``wavelength`` in nm. [-]

    Notes
    -----
    This spectral response is based on measurements taken on a c-Si cell.
    A small number of points near the measured curve are used to define
    a cubic spline having no undue oscillations, as shown in [1]_.  The spline
    can be interpolated at arbitrary wavelengths to produce a continuous,
    smooth curve , which makes it suitable for experimenting with spectral
    data of different resolutions.

    References
    ----------
    .. [1] Driesse, Anton, and Stein, Joshua. "Global Normal Spectral
       Irradiance in Albuquerque: a One-Year Open Dataset for PV Research".
       United States 2020. :doi:`10.2172/1814068`.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    SR_DATA = np.array([[ 290, 0.00],
                        [ 350, 0.27],
                        [ 400, 0.37],
                        [ 500, 0.52],
                        [ 650, 0.71],
                        [ 800, 0.88],
                        [ 900, 0.97],
                        [ 950, 1.00],
                        [1000, 0.93],
                        [1050, 0.58],
                        [1100, 0.21],
                        [1150, 0.05],
                        [1190, 0.00]]).transpose()

    if wavelength is None:
        resolution = 5.0
        wavelength = np.arange(280, 1200 + resolution, resolution)

    interpolator = interp1d(SR_DATA[0], SR_DATA[1],
                            kind='cubic',
                            bounds_error=False,
                            fill_value=0.0,
                            copy=False,
                            assume_sorted=True)

    sr = pd.Series(data=interpolator(wavelength), index=wavelength)

    sr.index.name = 'wavelength'
    sr.name = 'spectral_response'

    return sr


def get_am15g(wavelength=None):
    '''
    Read the ASTM G173-03 AM1.5 global spectrum on a 37-degree tilted surface,
    optionally interpolated to the specified wavelength(s).

    Global (tilted) irradiance includes direct and diffuse irradiance from sky
    and ground reflections, and is more formally called hemispherical
    irradiance (on a tilted surface).  In the context of photovoltaic systems
    the irradiance on a flat receiver is frequently called plane-of-array (POA)
    irradiance.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which the spectrum is interpolated.
        By default the 2002 wavelengths of the standard are returned. [nm]

    Returns
    -------
    am15g: pandas.Series
        The AM1.5g standard spectrum indexed by ``wavelength``. [(W/m^2)/nm]

    Notes
    -----
    If ``wavelength`` is specified this function uses linear interpolation.

    If the values in ``wavelength`` are too widely spaced, the integral of the
    spectrum may deviate from the standard value of 1000.37 W/m^2.

    The values in the data file provided with pvlib-python are copied from an
    Excel file distributed by NREL, which is found here:
    https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

    More information about reference spectra is found here:
    https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html

    References
    ----------
    .. [1] ASTM "G173-03 Standard Tables for Reference Solar Spectral
        Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface."
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    pvlib_path = pvlib.__path__[0]
    filepath = os.path.join(pvlib_path, 'data', 'astm_g173_am15g.csv')

    am15g = pd.read_csv(filepath, index_col=0).squeeze()

    if wavelength is not None:
        interpolator = interp1d(am15g.index, am15g,
                                kind='linear',
                                bounds_error=False,
                                fill_value=0.0,
                                copy=False,
                                assume_sorted=True)

        am15g = pd.Series(data=interpolator(wavelength), index=wavelength)

    am15g.index.name = 'wavelength'
    am15g.name = 'am15g'

    return am15g


def calc_spectral_mismatch_field(sr, e_sun, e_ref=None):
    """
    Calculate spectral mismatch between a test device and broadband reference
    device under specified solar spectral irradiance conditions.

    Parameters
    ----------
    sr: pandas.Series
        The relative spectral response of one (photovoltaic) test device.
        The index of the Series must contain wavelength values in nm. [-]

    e_sun: pandas.DataFrame or pandas.Series
        One or more measured solar irradiance spectra in a pandas.DataFrame
        having wavelength in nm as column index.  A single spectrum may be
        be given as a pandas.Series having wavelength in nm as index.
        [(W/m^2)/nm]

    e_ref: pandas.Series, optional
        The reference spectrum to use for the mismatch calculation.
        The index of the Series must contain wavelength values in nm.
        The default is the ASTM G173-03 global tilted spectrum. [(W/m^2)/nm]

    Returns
    -------
    smm: pandas.Series or float if a single measured spectrum is provided. [-]

    Notes
    -----
    Measured solar spectral irradiance usually covers a wavelength range
    that is smaller than the range considered as broadband irradiance.
    The infrared limit for the former typically lies around 1100 or 1600 nm,
    whereas the latter extends to around 2800 or 4000 nm.  To avoid imbalance
    between the magnitudes of the integrated spectra (the broadband values)
    this function truncates the reference spectrum to the same range as the
    measured (or simulated) field spectra. The assumption implicit in this
    truncation is that the energy in the unmeasured wavelength range
    is the same fraction of the broadband energy for both the measured
    spectra and the reference spectrum.

    If the default reference spectrum is used it is linearly interpolated
    to the wavelengths of the measured spectrum, but if a reference spectrum
    is provided via the parameter ``e_ref`` it is used without change. This
    makes it possible to avoid interpolation, or to use a different method of
    interpolation, or to avoid truncation.

    The spectral response is linearly interpolated to the wavelengths of each
    spectrum with which is it multiplied internally (``e_sun`` and ``e_ref``).
    If the wavelengths of the spectral response already match one or both
    of these spectra interpolation has no effect; therefore, another type of
    interpolation could be used to process ``sr`` before calling this function.

    The standards describing mismatch calculations focus on indoor laboratory
    applications, but are applicable to outdoor performance as well.
    The 2016 version of ASTM E973 [1]_ is somewhat more difficult to
    read than the 2010 version [2]_ because it includes adjustments for
    the temperature dependency of spectral response, which led to a
    formulation using quantum efficiency (QE).
    IEC 60904-7 is clearer and also discusses the use of a broadband
    reference device. [3]_

    References
    ----------
    .. [1] ASTM "E973-16 Standard Test Method for Determination of the
       Spectral Mismatch Parameter Between a Photovoltaic Device and a
       Photovoltaic Reference Cell" :doi:`10.1520/E0973-16R20`
    .. [2] ASTM "E973-10 Standard Test Method for Determination of the
       Spectral Mismatch Parameter Between a Photovoltaic Device and a
       Photovoltaic Reference Cell" :doi:`10.1520/E0973-10`
    .. [3] IEC 60904-7 "Computation of the spectral mismatch correction
       for measurements of photovoltaic devices"
    """
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    # get the reference spectrum at wavelengths matching the measured spectra
    if e_ref is None:
        e_ref = get_am15g(wavelength=e_sun.T.index)

    # interpolate the sr at the wavelengths of the spectra
    # reference spectrum wavelengths may differ if e_ref is from caller
    sr_sun = np.interp(e_sun.T.index, sr.index, sr, left=0.0, right=0.0)
    sr_ref = np.interp(e_ref.T.index, sr.index, sr, left=0.0, right=0.0)

    # a helper function to make usable fraction calculations more readable
    def integrate(e):
        return np.trapz(e, x=e.T.index, axis=-1)

    # calculate usable fractions
    uf_sun = integrate(e_sun * sr_sun) / integrate(e_sun)
    uf_ref = integrate(e_ref * sr_ref) / integrate(e_ref)

    # mismatch is the ratio or quotient of the usable fractions
    smm = uf_sun / uf_ref

    if isinstance(e_sun, pd.DataFrame):
        smm = pd.Series(smm, index=e_sun.index)

    return smm


def martin_ruiz_spectral_modifier(clearness_index, airmass_absolute,
                                  cell_type=None, model_parameters=None):
    r"""
    Calculate mismatch modifiers for POA direct, sky diffuse and ground diffuse
    irradiances due to material's spectral response to spectrum characterised
    by the clearness index and the absolute airmass, with respect to the
    standard spectrum.

    .. warning::
        Included model parameters for ``monosi``, ``polysi`` and ``asi`` were
        estimated using the airmass model ``kasten1966`` [1]_. It is heavily
        recommended to use the same model in order to not introduce errors.
        See :py:func:`~pvlib.atmosphere.get_relative_airmass`.

    Parameters
    ----------
    clearness_index : numeric
        Clearness index of the sky.

    airmass_absolute : numeric
        Absolute airmass. Give attention to algorithm used (``kasten1966`` is
        recommended for default parameters of ``monosi``, ``polysi`` and
        ``asi``, see reference [1]_).

    cell_type : string, optional
        Specifies material of the cell in order to infer model parameters.
        Allowed types are ``monosi``, ``polysi`` and ``asi``, either lower or
        upper case. If not specified, ``model_parameters`` must be provided.

    model_parameters : dict-like, optional
        In case you computed the model parameters. In case any component is not
        specified, result will have a ``None`` value in its corresponding key.
        Provide either a dict or a ``pd.DataFrame`` as follows:

        .. code-block:: python

            # Using a dict
            # Return keys are the same as specifying 'cell_type'
            model_parameters = {
                'direct': {'c': c1, 'a': a1, 'b': b1},
                'sky_diffuse': {'c': c2, 'a': a2, 'b': b2},
                'ground_diffuse': {'c': c3, 'a': a3, 'b': b3}
            }
            # Using a pd.DataFrame
            model_parameters = pd.DataFrame({
                'direct': [c1, a1, b1],
                'sky_diffuse':  [c2, a2, b2],
                'ground_diffuse': [c3, a3, b3]},
                index=('c', 'a', 'b'))

        ``c``, ``a`` and ``b`` must be scalar.

    Returns
    -------
    Mismatch modifiers : pd.Series of numeric or None
        Modifiers for direct, sky diffuse and ground diffuse irradiances, with
        indexes ``direct``, ``sky_diffuse``, ``ground_diffuse``.
        Each mismatch modifier should be multiplied by its corresponding
        POA component.
        Returns None for a component if provided ``model_parameters`` does not
        include its coefficients.

    Raises
    ------
    ValueError
        If ``model_parameters`` is not suitable. See examples given above.
    TypeError
        If neither ``cell_type`` nor ``model_parameters`` are given.
    NotImplementedError
        If ``cell_type`` is not found in internal table of parameters.

    Notes
    -----
    The mismatch modifier is defined as

    .. math:: M = c \cdot \exp( a \cdot (K_t - 0.74) + b \cdot (AM - 1.5) )

    where ``c``, ``a`` and ``b`` are the model parameters, different for each
    irradiance component.

    References
    ----------
    .. [1] Martín, N. and Ruiz, J.M. (1999), A new method for the spectral
       characterisation of PV modules. Prog. Photovolt: Res. Appl., 7: 299-310.
       :doi:10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0

    See Also
    --------
    pvlib.irradiance.clearness_index
    pvlib.atmosphere.get_relative_airmass
    pvlib.atmosphere.get_absolute_airmass
    pvlib.atmosphere.first_solar_spectral_correction
    """
    # Note tests for this function are prefixed with test_martin_ruiz_mm_*

    IRRAD_COMPONENTS = ('direct', 'sky_diffuse', 'ground_diffuse')
    # Fitting parameters directly from [1]_
    MARTIN_RUIZ_PARAMS = pd.DataFrame(
        index=('monosi', 'polysi', 'asi'),
        columns=pd.MultiIndex.from_product([IRRAD_COMPONENTS,
                                           ('c', 'a', 'b')]),
        data=[  # Direct(c,a,b)  | Sky diffuse(c,a,b) | Ground diffuse(c,a,b)
            [1.029, -.313, 524e-5, .764, -.882, -.0204, .970, -.244, .0129],
            [1.029, -.311, 626e-5, .764, -.929, -.0192, .970, -.270, .0158],
            [1.024, -.222, 920e-5, .840, -.728, -.0183, .989, -.219, .0179],
        ])

    # Argument validation and choose components and model parameters
    if cell_type is not None and model_parameters is None:
        # Infer parameters from cell material
        cell_type_lower = cell_type.lower()
        if cell_type_lower in MARTIN_RUIZ_PARAMS.index:
            _params = MARTIN_RUIZ_PARAMS.loc[cell_type_lower]
        else:
            raise NotImplementedError('Cell type parameters not defined in '
                                      'algorithm! Allowed types are '
                                      f'{tuple(MARTIN_RUIZ_PARAMS.index)}')
    elif cell_type is None and model_parameters is None:
        raise TypeError('You must pass at least "cell_type" '
                        'or "model_parameters" as arguments!')
    elif model_parameters is not None:  # Use user-defined model parameters
        # Validate 'model_parameters' sub-dicts keys
        if any([{'a', 'b', 'c'} != set(model_parameters[component].keys())
                for component in model_parameters.keys()]):
            raise ValueError("You must specify model parameters with keys "
                             "'a','b','c' for each irradiation component.")

        _params = model_parameters
        if cell_type is not None:
            warn('Both "cell_type" and "model_parameters" given! '
                 'Using provided "model_parameters".')

    # Compute difference to avoid recalculating inside loop
    kt_delta = clearness_index - 0.74
    am_delta = airmass_absolute - 1.5

    modifiers = pd.Series(index=IRRAD_COMPONENTS, data=[None, None, None])

    # Calculate mismatch modifier for each irradiation
    for irrad_type in IRRAD_COMPONENTS:
        # Skip irradiations not specified in 'model_params'
        if irrad_type not in _params.keys():
            continue
        # Else, calculate the mismatch modifier
        _coeffs = _params[irrad_type]
        modifier = _coeffs['c'] * np.exp(_coeffs['a'] * kt_delta
                                         + _coeffs['b'] * am_delta)
        modifiers[irrad_type] = modifier

    return modifiers
