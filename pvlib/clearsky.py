"""
The ``clearsky`` module contains several methods
to calculate clear sky GHI, DNI, and DHI.
"""

from __future__ import division

import logging
logger = logging.getLogger('pvlib')

import os

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition


def ineichen(time, latitude, longitude, altitude=0, linke_turbidity=None,
             solarposition_method='nrel_numpy', zenith_data=None,
             airmass_model='young1994', airmass_data=None,
             interp_turbidity=True):
    '''
    Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model

    Implements the Ineichen and Perez clear sky model for global horizontal
    irradiance (GHI), direct normal irradiance (DNI), and calculates
    the clear-sky diffuse horizontal (DHI) component as the difference
    between GHI and DNI*cos(zenith) as presented in [1, 2]. A report on clear
    sky models found the Ineichen/Perez model to have excellent performance
    with a minimal input data set [3].

    Default values for montly Linke turbidity provided by SoDa [4, 5].

    Parameters
    -----------
    time : pandas.DatetimeIndex

    latitude : float

    longitude : float

    altitude : float

    linke_turbidity : None or float
        If None, uses ``LinkeTurbidities.mat`` lookup table.

    solarposition_method : string
        Sets the solar position algorithm.
        See solarposition.get_solarposition()

    zenith_data : None or Series
        If None, ephemeris data will be calculated using ``solarposition_method``.

    airmass_model : string
        See pvlib.airmass.relativeairmass().

    airmass_data : None or Series
        If None, absolute air mass data will be calculated using
        ``airmass_model`` and location.alitude.

    interp_turbidity : bool
        If ``True``, interpolates the monthly Linke turbidity values
        found in ``LinkeTurbidities.mat`` to daily values.

    Returns
    --------
    DataFrame with the following columns: ``ghi, dni, dhi``.

    Notes
    -----
    If you are using this function
    in a loop, it may be faster to load LinkeTurbidities.mat outside of
    the loop and feed it in as a keyword argument, rather than
    having the function open and process the file each time it is called.

    References
    ----------

    [1] P. Ineichen and R. Perez, "A New airmass independent formulation for
        the Linke turbidity coefficient", Solar Energy, vol 73, pp. 151-157, 2002.

    [2] R. Perez et. al., "A New Operational Model for Satellite-Derived
        Irradiances: Description and Validation", Solar Energy, vol 73, pp.
        307-317, 2002.

    [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
        Sky Models: Implementation and Analysis", Sandia National
        Laboratories, SAND2012-2389, 2012.

    [4] http://www.soda-is.com/eng/services/climat_free_eng.php#c5 (obtained
        July 17, 2012).

    [5] J. Remund, et. al., "Worldwide Linke Turbidity Information", Proc.
        ISES Solar World Congress, June 2003. Goteborg, Sweden.
    '''
    # Initial implementation of this algorithm by Matthew Reno.
    # Ported to python by Rob Andrews
    # Added functionality by Will Holmgren (@wholmgren)

    I0 = irradiance.extraradiation(time.dayofyear)

    if zenith_data is None:
        ephem_data = solarposition.get_solarposition(time,
                                                     latitude=latitude,
                                                     longitude=longitude,
                                                     altitude=altitude,
                                                     method=solarposition_method)
        time = ephem_data.index # fixes issue with time possibly not being tz-aware
        try:
            ApparentZenith = ephem_data['apparent_zenith']
        except KeyError:
            ApparentZenith = ephem_data['zenith']
            logger.warning('could not find apparent_zenith. using zenith')
    else:
        ApparentZenith = zenith_data
    #ApparentZenith[ApparentZenith >= 90] = 90 # can cause problems in edge cases


    if linke_turbidity is None:
        TL = lookup_linke_turbidity(time, latitude, longitude,
                                    interp_turbidity=interp_turbidity)
    else:
        TL = linke_turbidity

    # Get the absolute airmass assuming standard local pressure (per
    # alt2pres) using Kasten and Young's 1989 formula for airmass.

    if airmass_data is None:
        AMabsolute = atmosphere.absoluteairmass(airmass_relative=atmosphere.relativeairmass(ApparentZenith, airmass_model),
                                                pressure=atmosphere.alt2pres(altitude))
    else:
        AMabsolute = airmass_data

    fh1 = np.exp(-altitude/8000.)
    fh2 = np.exp(-altitude/1250.)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387
    logger.debug('fh1=%s, fh2=%s, cg1=%s, cg2=%s', fh1, fh2, cg1, cg2)

    #  Dan's note on the TL correction: By my reading of the publication on
    #  pages 151-157, Ineichen and Perez introduce (among other things) three
    #  things. 1) Beam model in eqn. 8, 2) new turbidity factor in eqn 9 and
    #  appendix A, and 3) Global horizontal model in eqn. 11. They do NOT appear
    #  to use the new turbidity factor (item 2 above) in either the beam or GHI
    #  models. The phrasing of appendix A seems as if there are two separate
    #  corrections, the first correction is used to correct the beam/GHI models,
    #  and the second correction is used to correct the revised turibidity
    #  factor. In my estimation, there is no need to correct the turbidity
    #  factor used in the beam/GHI models.

    #  Create the corrected TL for TL < 2
    #  TLcorr = TL;
    #  TLcorr(TL < 2) = TLcorr(TL < 2) - 0.25 .* (2-TLcorr(TL < 2)) .^ (0.5);

    #  This equation is found in Solar Energy 73, pg 311.
    #  Full ref: Perez et. al., Vol. 73, pp. 307-317 (2002).
    #  It is slightly different than the equation given in Solar Energy 73, pg 156.
    #  We used the equation from pg 311 because of the existence of known typos
    #  in the pg 156 publication (notably the fh2-(TL-1) should be fh2 * (TL-1)).

    cos_zenith = tools.cosd(ApparentZenith)

    clearsky_GHI = ( cg1 * I0 * cos_zenith *
                     np.exp(-cg2*AMabsolute*(fh1 + fh2*(TL - 1))) *
                     np.exp(0.01*AMabsolute**1.8) )
    clearsky_GHI[clearsky_GHI < 0] = 0

    # BncI == "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    BncI = b * I0 * np.exp( -0.09 * AMabsolute * (TL - 1) )
    logger.debug('b=%s', b)

    # "empirical correction" SE 73, 157 & SE 73, 312.
    BncI_2 = ( clearsky_GHI *
               ( 1 - (0.1 - 0.2*np.exp(-TL))/(0.1 + 0.882/fh1) ) /
               cos_zenith )

    clearsky_DNI = np.minimum(BncI, BncI_2)

    clearsky_DHI = clearsky_GHI - clearsky_DNI*cos_zenith

    df_out = pd.DataFrame({'ghi':clearsky_GHI, 'dni':clearsky_DNI,
                           'dhi':clearsky_DHI})
    df_out.fillna(0, inplace=True)

    return df_out


def lookup_linke_turbidity(time, latitude, longitude, filepath=None,
                           interp_turbidity=True):
    """
    Look up the Linke Turibidity from the ``LinkeTurbidities.mat``
    data file supplied with pvlib.

    Parameters
    ----------
    time : pandas.DatetimeIndex

    latitude : float

    longitude : float

    filepath : string
        The path to the ``.mat`` file.

    interp_turbidity : bool
        If ``True``, interpolates the monthly Linke turbidity values
        found in ``LinkeTurbidities.mat`` to daily values.

    Returns
    -------
    turbidity : Series
    """

    # The .mat file 'LinkeTurbidities.mat' contains a single 2160 x 4320 x 12
    # matrix of type uint8 called 'LinkeTurbidity'. The rows represent global
    # latitudes from 90 to -90 degrees; the columns represent global longitudes
    # from -180 to 180; and the depth (third dimension) represents months of
    # the year from January (1) to December (12). To determine the Linke
    # turbidity for a position on the Earth's surface for a given month do the
    # following: LT = LinkeTurbidity(LatitudeIndex, LongitudeIndex, month).
    # Note that the numbers within the matrix are 20 * Linke Turbidity,
    # so divide the number from the file by 20 to get the
    # turbidity.

    try:
        import scipy.io
    except ImportError:
        raise ImportError('The Linke turbidity lookup table requires scipy. ' +
                          'You can still use clearsky.ineichen if you ' +
                          'supply your own turbidities.')

    if filepath is None:
        pvlib_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(pvlib_path, 'data', 'LinkeTurbidities.mat')

    mat = scipy.io.loadmat(filepath)
    linke_turbidity_table = mat['LinkeTurbidity']

    latitude_index = np.around(_linearly_scale(latitude, 90, -90, 1, 2160)).astype(np.int64)
    longitude_index = np.around(_linearly_scale(longitude, -180, 180, 1, 4320)).astype(np.int64)

    g = linke_turbidity_table[latitude_index][longitude_index]

    if interp_turbidity:
        logger.info('interpolating turbidity to the day')
        # Cata covers 1 year.
        # Assume that data corresponds to the value at
        # the middle of each month.
        # This means that we need to add previous Dec and next Jan
        # to the array so that the interpolation will work for
        # Jan 1 - Jan 15 and Dec 16 - Dec 31.
        # Then we map the month value to the day of year value.
        # This is approximate and could be made more accurate.
        g2 = np.concatenate([[g[-1]], g, [g[0]]])
        days = np.linspace(-15, 380, num=14)
        linke_turbidity = pd.Series(np.interp(time.dayofyear, days, g2),
                                    index=time)
    else:
        logger.info('using monthly turbidity')
        apply_month = lambda x: g[x[0]-1]
        linke_turbidity = pd.DataFrame(time.month, index=time)
        linke_turbidity = linke_turbidity.apply(apply_month, axis=1)

    linke_turbidity /= 20.

    return linke_turbidity


def haurwitz(apparent_zenith):
    '''
    Determine clear sky GHI from Haurwitz model.

    Implements the Haurwitz clear sky model for global horizontal
    irradiance (GHI) as presented in [1, 2]. A report on clear
    sky models found the Haurwitz model to have the best performance of
    models which require only zenith angle [3]. Extreme care should
    be taken in the interpretation of this result!

    Parameters
    ----------
    apparent_zenith : Series
        The apparent (refraction corrected) sun zenith angle
        in degrees.

    Returns
    -------
    pd.Series
    The modeled global horizonal irradiance in W/m^2 provided
    by the Haurwitz clear-sky model.

    Initial implementation of this algorithm by Matthew Reno.

    References
    ----------

    [1] B. Haurwitz, "Insolation in Relation to Cloudiness and Cloud
     Density," Journal of Meteorology, vol. 2, pp. 154-166, 1945.

    [2] B. Haurwitz, "Insolation in Relation to Cloud Type," Journal of
     Meteorology, vol. 3, pp. 123-124, 1946.

    [3] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
     Sky Models: Implementation and Analysis", Sandia National
     Laboratories, SAND2012-2389, 2012.
    '''

    cos_zenith = tools.cosd(apparent_zenith)

    clearsky_GHI = 1098.0 * cos_zenith * np.exp(-0.059/cos_zenith)

    clearsky_GHI[clearsky_GHI < 0] = 0

    df_out = pd.DataFrame({'ghi':clearsky_GHI})

    return df_out


def _linearly_scale(inputmatrix, inputmin, inputmax, outputmin, outputmax):
    """ used by linke turbidity lookup function """

    inputrange = inputmax - inputmin
    outputrange = outputmax - outputmin
    OutputMatrix = (inputmatrix-inputmin) * outputrange/inputrange + outputmin
    return OutputMatrix


def simplified_solis(apparent_elevation, aod700=0.1, precipitable_water=1.,
                     pressure=101325., dni_extra=1364., return_raw=False):
    """
    Calculate the clear sky GHI, DNI, and DHI according to the
    simplified Solis model [1]_.

    Reference [1]_ describes the accuracy of the model as being 15, 20,
    and 18 W/m^2 for the beam, global, and diffuse components. Reference
    [2]_ provides comparisons with other clear sky models.

    Parameters
    ----------
    apparent_elevation: numeric
        The apparent elevation of the sun above the horizon (deg).

    aod700: numeric
        The aerosol optical depth at 700 nm (unitless).
        Algorithm derived for values between 0 and 0.45.

    precipitable_water: numeric
        The precipitable water of the atmosphere (cm).
        Algorithm derived for values between 0.2 and 10 cm.
        Values less than 0.2 will be assumed to be equal to 0.2.

    pressure: numeric
        The atmospheric pressure (Pascals).
        Algorithm derived for altitudes between sea level and 7000 m,
        or 101325 and 41000 Pascals.

    dni_extra: numeric
        Extraterrestrial irradiance.

    return_raw: bool
        Controls the return type. If False, function returns a DataFrame,
        if True, function returns an array.

    Returns
    --------
    clearsky : pd.DataFrame or np.array (determined by ``return_raw``)
        DataFrame contains the columns ``'dhi', 'dni', 'ghi'`` with the
        units of the ``dni_extra`` input. If ``return_raw=True``,
        returns the array [dhi, dni, ghi] with shape determined by the
        input arrays.

    References
    ----------
    .. [1] P. Ineichen, "A broadband simplified version of the
       Solis clear sky model," Solar Energy, 82, 758-762 (2008).

    .. [2] P. Ineichen, "Validation of models that estimate the clear
       sky global and beam solar irradiance," Solar Energy, 132,
       332-344 (2016).
    """

    p = pressure

    w = precipitable_water

    # algorithm fails for pw < 0.2
    if np.isscalar(w):
        w = 0.2 if w < 0.2 else w
    else:
        w = w.copy()
        w[w < 0.2] = 0.2

    # this algorithm is reasonably fast already, but it could be made
    # faster by precalculating the powers of aod700, the log(p/p0), and
    # the log(w) instead of repeating the calculations as needed in each
    # function

    i0p = _calc_i0p(dni_extra, w, aod700, p)

    taub = _calc_taub(w, aod700, p)
    b = _calc_b(w, aod700)

    taug = _calc_taug(w, aod700, p)
    g = _calc_g(w, aod700)

    taud = _calc_taud(w, aod700, p)
    d = _calc_d(w, aod700, p)

    sin_elev = np.sin(np.radians(apparent_elevation))

    dni = i0p * np.exp(-taub/sin_elev**b)
    ghi = i0p * np.exp(-taug/sin_elev**g) * sin_elev
    dhi = i0p * np.exp(-taud/sin_elev**d)

    irrads = np.array([dhi, dni, ghi])

    if not return_raw:
        if isinstance(dni, pd.Series):
            index = dni.index
        else:
            index = None

        try:
            irrads = pd.DataFrame(irrads.T, columns=['dhi', 'dni', 'ghi'],
                                  index=index)
        except ValueError:
            # probably all scalar input, so we
            # need to increase the dimensionality
            irrads = pd.DataFrame(np.array([irrads]),
                                  columns=['dhi', 'dni', 'ghi'])
        finally:
            irrads = irrads.fillna(0)

    return irrads


def _calc_i0p(i0, w, aod700, p):
    """Calculate the "enhanced extraterrestrial irradiance"."""
    p0 = 101325.
    io0 = 1.08 * w**0.0051
    i01 = 0.97 * w**0.032
    i02 = 0.12 * w**0.56
    i0p = i0 * (i02*aod700**2 + i01*aod700 + io0 + 0.071*np.log(p/p0))

    return i0p


def _calc_taub(w, aod700, p):
    """Calculate the taub coefficient"""
    p0 = 101325.
    tb1 = 1.82 + 0.056*np.log(w) + 0.0071*np.log(w)**2
    tb0 = 0.33 + 0.045*np.log(w) + 0.0096*np.log(w)**2
    tbp = 0.0089*w + 0.13

    taub = tb1*aod700 + tb0 + tbp*np.log(p/p0)

    return taub


def _calc_b(w, aod700):
    """Calculate the b coefficient."""

    b1 = 0.00925*aod700**2 + 0.0148*aod700 - 0.0172
    b0 = -0.7565*aod700**2 + 0.5057*aod700 + 0.4557

    b = b1 * np.log(w) + b0

    return b


def _calc_taug(w, aod700, p):
    """Calculate the taug coefficient"""
    p0 = 101325.
    tg1 = 1.24 + 0.047*np.log(w) + 0.0061*np.log(w)**2
    tg0 = 0.27 + 0.043*np.log(w) + 0.0090*np.log(w)**2
    tgp = 0.0079*w + 0.1
    taug = tg1*aod700 + tg0 + tgp*np.log(p/p0)

    return taug


def _calc_g(w, aod700):
    """Calculate the g coefficient."""

    g = -0.0147*np.log(w) - 0.3079*aod700**2 + 0.2846*aod700 + 0.3798

    return g


def _calc_taud(w, aod700, p):
    """Calculate the taud coefficient."""

    # isscalar tests needed to ensure that the arrays will have the
    # right shape in the tds calculation.
    # there's probably a better way to do this.

    if np.isscalar(w) and np.isscalar(aod700):
        w = np.array([w])
        aod700 = np.array([aod700])
    elif np.isscalar(w):
        w = np.full_like(aod700, w)
    elif np.isscalar(aod700):
        aod700 = np.full_like(w, aod700)

    aod700_mask = aod700 < 0.05
    aod700_mask = np.array([aod700_mask, ~aod700_mask], dtype=np.int)

    # create tuples of coefficients for
    # aod700 < 0.05, aod700 >= 0.05
    td4 = 86*w - 13800, -0.21*w + 11.6
    td3 = -3.11*w + 79.4, 0.27*w - 20.7
    td2 = -0.23*w + 74.8, -0.134*w + 15.5
    td1 = 0.092*w - 8.86, 0.0554*w - 5.71
    td0 = 0.0042*w + 3.12, 0.0057*w + 2.94
    tdp = -0.83*(1+aod700)**(-17.2), -0.71*(1+aod700)**(-15.0)

    tds = (np.array([td0, td1, td2, td3, td4, tdp]) * aod700_mask).sum(axis=1)

    p0 = 101325.
    taud = (tds[4]*aod700**4 + tds[3]*aod700**3 + tds[2]*aod700**2 +
            tds[1]*aod700 + tds[0] + tds[5]*np.log(p/p0))

    # be polite about matching the output type to the input type(s)
    if len(taud) == 1:
        taud = taud[0]

    return taud


def _calc_d(w, aod700, p):
    """Calculate the d coefficient."""

    p0 = 101325.
    dp = 1/(18 + 152*aod700)
    d = -0.337*aod700**2 + 0.63*aod700 + 0.116 + dp*np.log(p/p0)

    return d
