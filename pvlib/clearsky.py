"""
The ``clearsky`` module contains several methods
to calculate clear sky GHI, DNI, and DHI.
"""

from __future__ import division

import os
from collections import OrderedDict
import calendar

import numpy as np
import pandas as pd

from pvlib import tools


def ineichen(apparent_zenith, airmass_absolute, linke_turbidity,
             altitude=0, dni_extra=1364.):
    '''
    Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model.

    Implements the Ineichen and Perez clear sky model for global
    horizontal irradiance (GHI), direct normal irradiance (DNI), and
    calculates the clear-sky diffuse horizontal (DHI) component as the
    difference between GHI and DNI*cos(zenith) as presented in [1, 2]. A
    report on clear sky models found the Ineichen/Perez model to have
    excellent performance with a minimal input data set [3].

    Default values for monthly Linke turbidity provided by SoDa [4, 5].

    Parameters
    -----------
    apparent_zenith: numeric
        Refraction corrected solar zenith angle in degrees.

    airmass_absolute: numeric
        Pressure corrected airmass.

    linke_turbidity: numeric
        Linke Turbidity.

    altitude: numeric
        Altitude above sea level in meters.

    dni_extra: numeric
        Extraterrestrial irradiance. The units of ``dni_extra``
        determine the units of the output.

    Returns
    -------
    clearsky : DataFrame (if Series input) or OrderedDict of arrays
        DataFrame/OrderedDict contains the columns/keys
        ``'dhi', 'dni', 'ghi'``.

    See also
    --------
    lookup_linke_turbidity
    pvlib.location.Location.get_clearsky

    References
    ----------
    [1] P. Ineichen and R. Perez, "A New airmass independent formulation for
        the Linke turbidity coefficient", Solar Energy, vol 73, pp. 151-157,
        2002.

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

    # Dan's note on the TL correction: By my reading of the publication
    # on pages 151-157, Ineichen and Perez introduce (among other
    # things) three things. 1) Beam model in eqn. 8, 2) new turbidity
    # factor in eqn 9 and appendix A, and 3) Global horizontal model in
    # eqn. 11. They do NOT appear to use the new turbidity factor (item
    # 2 above) in either the beam or GHI models. The phrasing of
    # appendix A seems as if there are two separate corrections, the
    # first correction is used to correct the beam/GHI models, and the
    # second correction is used to correct the revised turibidity
    # factor. In my estimation, there is no need to correct the
    # turbidity factor used in the beam/GHI models.

    # Create the corrected TL for TL < 2
    # TLcorr = TL;
    # TLcorr(TL < 2) = TLcorr(TL < 2) - 0.25 .* (2-TLcorr(TL < 2)) .^ (0.5);

    # This equation is found in Solar Energy 73, pg 311. Full ref: Perez
    # et. al., Vol. 73, pp. 307-317 (2002). It is slightly different
    # than the equation given in Solar Energy 73, pg 156. We used the
    # equation from pg 311 because of the existence of known typos in
    # the pg 156 publication (notably the fh2-(TL-1) should be fh2 *
    # (TL-1)).

    # The NaN handling is a little subtle. The AM input is likely to
    # have NaNs that we'll want to map to 0s in the output. However, we
    # want NaNs in other inputs to propagate through to the output. This
    # is accomplished by judicious use and placement of np.maximum,
    # np.minimum, and np.fmax

    # use max so that nighttime values will result in 0s instead of
    # negatives. propagates nans.
    cos_zenith = np.maximum(tools.cosd(apparent_zenith), 0)

    tl = linke_turbidity

    fh1 = np.exp(-altitude/8000.)
    fh2 = np.exp(-altitude/1250.)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387

    ghi = (np.exp(-cg2*airmass_absolute*(fh1 + fh2*(tl - 1))) *
           np.exp(0.01*airmass_absolute**1.8))
    # use fmax to map airmass nans to 0s. multiply and divide by tl to
    # reinsert tl nans
    ghi = cg1 * dni_extra * cos_zenith * tl / tl * np.fmax(ghi, 0)

    # BncI = "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    bnci = b * np.exp(-0.09 * airmass_absolute * (tl - 1))
    bnci = dni_extra * np.fmax(bnci, 0)

    # "empirical correction" SE 73, 157 & SE 73, 312.
    bnci_2 = ((1 - (0.1 - 0.2*np.exp(-tl))/(0.1 + 0.882/fh1)) /
              cos_zenith)
    bnci_2 = ghi * np.fmin(np.fmax(bnci_2, 0), 1e20)

    dni = np.minimum(bnci, bnci_2)

    dhi = ghi - dni*cos_zenith

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(dni, pd.Series):
        irrads = pd.DataFrame.from_dict(irrads)

    return irrads


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

    # The nodes of the grid are 5' (1/12=0.0833[arcdeg]) apart.
    # From Section 8 of Aerosol optical depth and Linke turbidity climatology
    # http://www.meteonorm.com/images/uploads/downloads/ieashc36_report_TL_AOD_climatologies.pdf
    # 1st row: 89.9583 S, 2nd row: 89.875 S
    # 1st column: 179.9583 W, 2nd column: 179.875 W

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

    latitude_index = (
        np.around(_linearly_scale(latitude, 90, -90, 0, 2160))
        .astype(np.int64))
    longitude_index = (
        np.around(_linearly_scale(longitude, -180, 180, 0, 4320))
        .astype(np.int64))

    g = linke_turbidity_table[latitude_index][longitude_index]

    if interp_turbidity:
        # Data covers 1 year. Assume that data corresponds to the value at the
        # middle of each month. This means that we need to add previous Dec and
        # next Jan to the array so that the interpolation will work for
        # Jan 1 - Jan 15 and Dec 16 - Dec 31.
        g2 = np.concatenate([[g[-1]], g, [g[0]]])
        # Then we map the month value to the day of year value.
        isleap = [calendar.isleap(t.year) for t in time]
        if all(isleap):
            days = _calendar_month_middles(2016)  # all years are leap
        elif not any(isleap):
            days = _calendar_month_middles(2015)  # none of the years are leap
        else:
            days = None  # some of the years are leap years and some are not
        if days is None:
            # Loop over different years, might be slow for large timeserires
            linke_turbidity = pd.Series([
                np.interp(t.dayofyear, _calendar_month_middles(t.year), g2)
                for t in time
            ], index=time)
        else:
            linke_turbidity = pd.Series(np.interp(time.dayofyear, days, g2),
                                        index=time)
    else:
        linke_turbidity = pd.DataFrame(time.month, index=time)
        # apply monthly data
        linke_turbidity = linke_turbidity.apply(lambda x: g[x[0]-1], axis=1)

    linke_turbidity /= 20.

    return linke_turbidity


def _calendar_month_middles(year):
    """list of middle day of each month, used by Linke turbidity lookup"""
    # remove mdays[0] since January starts at mdays[1]
    # make local copy of mdays since we need to change February for leap years
    mdays = np.array(calendar.mdays[1:])
    ydays = 365
    # handle leap years
    if calendar.isleap(year):
        mdays[1] = mdays[1] + 1
        ydays = 366
    return np.concatenate([[-calendar.mdays[-1] / 2.0],  # Dec last year
                           np.cumsum(mdays) - np.array(mdays) / 2.,  # this year
                           [ydays + calendar.mdays[1] / 2.0]])  # Jan next year


def _linearly_scale(inputmatrix, inputmin, inputmax, outputmin, outputmax):
    """linearly scale input to output, used by Linke turbidity lookup"""
    inputrange = inputmax - inputmin
    outputrange = outputmax - outputmin
    delta = outputrange/inputrange  # number of indices per input unit
    inputmin = inputmin + 1.0 / delta / 2.0  # shift to center of index
    outputmax = outputmax - 1  # shift index to zero indexing
    outputmatrix = (inputmatrix - inputmin) * delta + outputmin
    err = IndexError('Input, %g, is out of range (%g, %g).' %
                     (inputmatrix, inputmax - inputrange, inputmax))
    # round down if input is within half an index or else raise index error
    if outputmatrix > outputmax:
        if np.around(outputmatrix - outputmax, 1) <= 0.5:
            outputmatrix = outputmax
        else:
            raise err
    elif outputmatrix < outputmin:
        if np.around(outputmin - outputmatrix, 1) <= 0.5:
            outputmatrix = outputmin
        else:
            raise err
    return outputmatrix


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

    clearsky_ghi = 1098.0 * cos_zenith * np.exp(-0.059/cos_zenith)

    clearsky_ghi[clearsky_ghi < 0] = 0

    df_out = pd.DataFrame({'ghi': clearsky_ghi})

    return df_out


def simplified_solis(apparent_elevation, aod700=0.1, precipitable_water=1.,
                     pressure=101325., dni_extra=1364.):
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
        Extraterrestrial irradiance. The units of ``dni_extra``
        determine the units of the output.

    Returns
    -------
    clearsky : DataFrame (if Series input) or OrderedDict of arrays
        DataFrame/OrderedDict contains the columns/keys
        ``'dhi', 'dni', 'ghi'``.

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

    # this prevents the creation of nans at night instead of 0s
    # it's also friendly to scalar and series inputs
    sin_elev = np.maximum(1.e-30, np.sin(np.radians(apparent_elevation)))

    dni = i0p * np.exp(-taub/sin_elev**b)
    ghi = i0p * np.exp(-taug/sin_elev**g) * sin_elev
    dhi = i0p * np.exp(-taud/sin_elev**d)

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(dni, pd.Series):
        irrads = pd.DataFrame.from_dict(irrads)

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


def detect_clearsky(measured, clearsky, times, window_length,
                    mean_diff=75, max_diff=75,
                    lower_line_length=-5, upper_line_length=10,
                    var_diff=0.005, slope_dev=8, max_iterations=20,
                    return_components=False):
    """
    Detects clear sky times according to the algorithm developed by Reno
    and Hansen for GHI measurements [1]. The algorithm was designed and
    validated for analyzing GHI time series only. Users may attempt to
    apply it to other types of time series data using different filter
    settings, but should be skeptical of the results.

    The algorithm detects clear sky times by comparing statistics for a
    measured time series and an expected clearsky time series.
    Statistics are calculated using a sliding time window (e.g., 10
    minutes). An iterative algorithm identifies clear periods, uses the
    identified periods to estimate bias in the clearsky data, scales the
    clearsky data and repeats.

    Clear times are identified by meeting 5 criteria. Default values for
    these thresholds are appropriate for 10 minute windows of 1 minute
    GHI data.

    Parameters
    ----------
    measured : array or Series
        Time series of measured values.
    clearsky : array or Series
        Time series of the expected clearsky values.
    times : DatetimeIndex
        Times of measured and clearsky values.
    window_length : int
        Length of sliding time window in minutes. Must be greater than 2
        periods.
    mean_diff : float
        Threshold value for agreement between mean values of measured
        and clearsky in each interval, see Eq. 6 in [1].
    max_diff : float
        Threshold value for agreement between maxima of measured and
        clearsky values in each interval, see Eq. 7 in [1].
    lower_line_length : float
        Lower limit of line length criterion from Eq. 8 in [1].
        Criterion satisfied when
        lower_line_length < line length difference < upper_line_length
    upper_line_length : float
        Upper limit of line length criterion from Eq. 8 in [1].
    var_diff : float
        Threshold value in Hz for the agreement between normalized
        standard deviations of rate of change in irradiance, see Eqs. 9
        through 11 in [1].
    slope_dev : float
        Threshold value for agreement between the largest magnitude of
        change in successive values, see Eqs. 12 through 14 in [1].
    max_iterations : int
        Maximum number of times to apply a different scaling factor to
        the clearsky and redetermine clear_samples. Must be 1 or larger.
    return_components : bool
        Controls if additional output should be returned. See below.

    Returns
    -------
    clear_samples : array or Series
        Boolean array or Series of whether or not the given time is
        clear. Return type is the same as the input type.

    components : OrderedDict, optional
        Dict of arrays of whether or not the given time window is clear
        for each condition. Only provided if return_components is True.

    alpha : scalar, optional
        Scaling factor applied to the clearsky_ghi to obtain the
        detected clear_samples. Only provided if return_components is
        True.

    References
    ----------
    [1] Reno, M.J. and C.W. Hansen, "Identification of periods of clear
    sky irradiance in time series of GHI measurements" Renewable Energy,
    v90, p. 520-531, 2016.

    Notes
    -----
    Initial implementation in MATLAB by Matthew Reno. Modifications for
    computational efficiency by Joshua Patrick and Curtis Martin. Ported
    to Python by Will Holmgren, Tony Lorenzo, and Cliff Hansen.

    Differences from MATLAB version:

        * no support for unequal times
        * automatically determines sample_interval
        * requires a reference clear sky series instead calculating one
          from a user supplied location and UTCoffset
        * parameters are controllable via keyword arguments
        * option to return individual test components and clearsky scaling
          parameter
    """

    # calculate deltas in units of minutes (matches input window_length units)
    deltas = np.diff(times) / np.timedelta64(1, '60s')

    # determine the unique deltas and if we can proceed
    unique_deltas = np.unique(deltas)
    if len(unique_deltas) == 1:
        sample_interval = unique_deltas[0]
    else:
        raise NotImplementedError('algorithm does not yet support unequal ' \
                                  'times. consider resampling your data.')

    samples_per_window = int(window_length / sample_interval)

    # generate matrix of integers for creating windows with indexing
    from scipy.linalg import hankel
    H = hankel(np.arange(samples_per_window),
               np.arange(samples_per_window-1, len(times)))

    # calculate measurement statistics
    meas_mean = np.mean(measured[H], axis=0)
    meas_max = np.max(measured[H], axis=0)
    meas_slope = np.diff(measured[H], n=1, axis=0)
    # matlab std function normalizes by N-1, so set ddof=1 here
    meas_slope_nstd = np.std(meas_slope, axis=0, ddof=1) / meas_mean
    meas_slope_max = np.max(np.abs(meas_slope), axis=0)
    meas_line_length = np.sum(np.sqrt(
        meas_slope*meas_slope + sample_interval*sample_interval), axis=0)

    # calculate clear sky statistics
    clear_mean = np.mean(clearsky[H], axis=0)
    clear_max = np.max(clearsky[H], axis=0)
    clear_slope = np.diff(clearsky[H], n=1, axis=0)
    clear_slope_max = np.max(np.abs(clear_slope), axis=0)

    from scipy.optimize import minimize_scalar

    alpha = 1
    for iteration in range(max_iterations):
        clear_line_length = np.sum(np.sqrt(
            alpha*alpha*clear_slope*clear_slope +
            sample_interval*sample_interval), axis=0)

        line_diff = meas_line_length - clear_line_length

        # evaluate comparison criteria
        c1 = np.abs(meas_mean - alpha*clear_mean) < mean_diff
        c2 = np.abs(meas_max - alpha*clear_max) < max_diff
        c3 = (line_diff > lower_line_length) & (line_diff < upper_line_length)
        c4 = meas_slope_nstd < var_diff
        c5 = (meas_slope_max - alpha*clear_slope_max) < slope_dev
        c6 = (clear_mean != 0) & ~np.isnan(clear_mean)
        clear_windows = c1 & c2 & c3 & c4 & c5 & c6

        # create array to return
        clear_samples = np.full_like(measured, False, dtype='bool')
        # find the samples contained in any window classified as clear
        clear_samples[np.unique(H[:, clear_windows])] = True

        # find a new alpha
        previous_alpha = alpha
        clear_meas = measured[clear_samples]
        clear_clear = clearsky[clear_samples]
        def rmse(alpha):
            return np.sqrt(np.mean((clear_meas - alpha*clear_clear)**2))
        alpha = minimize_scalar(rmse).x
        if round(alpha*10000) == round(previous_alpha*10000):
            break
    else:
        import warnings
        warnings.warn('failed to converge after %s iterations' \
                      % max_iterations, RuntimeWarning)

    # be polite about returning the same type as was input
    if isinstance(measured, pd.Series):
        clear_samples = pd.Series(clear_samples, index=times)

    if return_components:
        components = OrderedDict()
        components['mean_diff'] = c1
        components['max_diff'] = c2
        components['line_length'] = c3
        components['slope_nstd'] = c4
        components['slope_max'] = c5
        components['mean_nan'] = c6
        components['windows'] = clear_windows
        return clear_samples, components, alpha
    else:
        return clear_samples
