# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:04:01 2019

@author: cwhanse
"""
import numpy as np
import pandas as pd
from pvlib.tools import cosd, sind, tand, asind


# a dict of required parameter names for each IAM model
# keys are the function names for the IAM models
IAM_MODEL_PARAMS = {
    'ashrae': set(['b']),
    'physical': set(['n', 'K', 'L']),
    'martin_ruiz': set(['a_r']),
    'sapm': set(['B0', 'B1', 'B2', 'B3', 'B4', 'B5']),
    'interp': set([])
}


def ashrae(aoi, b=0.05):
    r"""
    Determine the incidence angle modifier using the ASHRAE transmission
    model.

    The ASHRAE (American Society of Heating, Refrigeration, and Air
    Conditioning Engineers) transmission model is developed in
    [1], and in [2]. The model has been used in software such as PVSyst [3].

    Parameters
    ----------
    aoi : numeric
        The angle of incidence (AOI) between the module normal vector and the
        sun-beam vector in degrees. Angles of nan will result in nan.

    b : float, default 0.05
        A parameter to adjust the incidence angle modifier as a function of
        angle of incidence. Typical values are on the order of 0.05 [3].

    Returns
    -------
    iam : numeric
        The incident angle modifier (IAM). Returns zero for all abs(aoi) >= 90
        and for all ``iam`` values that would be less than 0.

    Notes
    -----
    The incidence angle modifier is calculated as

    .. math::

        IAM = 1 - b (\sec(aoi) - 1)

    As AOI approaches 90 degrees, the model yields negative values for IAM;
    negative IAM values are set to zero in this implementation.

    References
    ----------
    [1] Souka A.F., Safwat H.H., "Determination of the optimum
    orientations for the double exposure flat-plate collector and its
    reflections". Solar Energy vol .10, pp 170-174. 1966.

    [2] ASHRAE standard 93-77

    [3] PVsyst Contextual Help.
    http://files.pvsyst.com/help/index.html?iam_loss.htm retrieved on
    September 10, 2012

    See Also
    --------
    iam.physical
    iam.martin_ruiz
    iam.interp
    """

    iam = 1 - b * ((1 / np.cos(np.radians(aoi)) - 1))
    aoi_gte_90 = np.full_like(aoi, False, dtype='bool')
    np.greater_equal(np.abs(aoi), 90, where=~np.isnan(aoi), out=aoi_gte_90)
    iam = np.where(aoi_gte_90, 0, iam)
    iam = np.maximum(0, iam)

    if isinstance(aoi, pd.Series):
        iam = pd.Series(iam, index=aoi.index)

    return iam


def physical(aoi, n=1.526, K=4., L=0.002):
    r"""
    Determine the incidence angle modifier using refractive index ``n``,
    extinction coefficient ``K``, and glazing thickness ``L``.

    ``iam.physical`` calculates the incidence angle modifier as described in
    [1], Section 3. The calculation is based on a physical model of absorbtion
    and transmission through a transparent cover.

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees. Angles of 0 are replaced with 1e-06
        to ensure non-nan results. Angles of nan will result in nan.

    n : numeric, default 1.526
        The effective index of refraction (unitless). Reference [1]
        indicates that a value of 1.526 is acceptable for glass.

    K : numeric, default 4.0
        The glazing extinction coefficient in units of 1/meters.
        Reference [1] indicates that a value of 4 is reasonable for
        "water white" glass.

    L : numeric, default 0.002
        The glazing thickness in units of meters. Reference [1]
        indicates that 0.002 meters (2 mm) is reasonable for most
        glass-covered PV panels.

    Returns
    -------
    iam : numeric
        The incident angle modifier

    Notes
    -----
    The pvlib python authors believe that Eqn. 14 in [1] is
    incorrect, which presents :math:`\theta_{r} = \arcsin(n \sin(AOI))`.
    Here, :math:`\theta_{r} = \arcsin(1/n \times \sin(AOI))`

    References
    ----------
    [1] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
    2006.

    [2] Duffie, John A. & Beckman, William A.. (2006). Solar Engineering
    of Thermal Processes, third edition. [Books24x7 version] Available
    from http://common.books24x7.com/toc.aspx?bookid=17160.

    See Also
    --------
    iam.martin_ruiz
    iam.ashrae
    iam.interp
    """
    zeroang = 1e-06

    # hold a new reference to the input aoi object since we're going to
    # overwrite the aoi reference below, but we'll need it for the
    # series check at the end of the function
    aoi_input = aoi

    aoi = np.where(aoi == 0, zeroang, aoi)

    # angle of reflection
    thetar_deg = asind(1.0 / n * (sind(aoi)))

    # reflectance and transmittance for normal incidence light
    rho_zero = ((1-n) / (1+n)) ** 2
    tau_zero = np.exp(-K*L)

    # reflectance for parallel and perpendicular polarized light
    rho_para = (tand(thetar_deg - aoi) / tand(thetar_deg + aoi)) ** 2
    rho_perp = (sind(thetar_deg - aoi) / sind(thetar_deg + aoi)) ** 2

    # transmittance for non-normal light
    tau = np.exp(-K * L / cosd(thetar_deg))

    # iam is ratio of non-normal to normal incidence transmitted light
    # after deducting the reflected portion of each
    iam = ((1 - (rho_para + rho_perp) / 2) / (1 - rho_zero) * tau / tau_zero)

    with np.errstate(invalid='ignore'):
        # angles near zero produce nan, but iam is defined as one
        small_angle = 1e-06
        iam = np.where(np.abs(aoi) < small_angle, 1.0, iam)

        # angles at 90 degrees can produce tiny negative values,
        # which should be zero. this is a result of calculation precision
        # rather than the physical model
        iam = np.where(iam < 0, 0, iam)

        # for light coming from behind the plane, none can enter the module
        iam = np.where(aoi > 90, 0, iam)

    if isinstance(aoi_input, pd.Series):
        iam = pd.Series(iam, index=aoi_input.index)

    return iam


def martin_ruiz(aoi, a_r=0.16):
    r'''
    Determine the incidence angle modifier (IAM) using the Martin
    and Ruiz incident angle model.

    Parameters
    ----------
    aoi : numeric, degrees
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees.

    a_r : numeric
        The angular losses coefficient described in equation 3 of [1].
        This is an empirical dimensionless parameter. Values of ``a_r`` are
        generally on the order of 0.08 to 0.25 for flat-plate PV modules.

    Returns
    -------
    iam : numeric
        The incident angle modifier(s)

    Notes
    -----
    `martin_ruiz` calculates the incidence angle modifier (IAM) as described in
    [1]. The information required is the incident angle (AOI) and the angular
    losses coefficient (a_r). Note that [1] has a corrigendum [2] which makes
    the document much simpler to understand.

    The incident angle modifier is defined as

    .. math::

       IAM = \frac{1 - \exp(-\cos(\frac{aoi}{a_r}))}
       {1 - \exp(\frac{-1}{a_r}}

    which is presented as :math:`AL(\alpha) = 1 - IAM` in equation 4 of [1],
    with :math:`\alpha` representing the angle of incidence AOI. Thus IAM = 1
    at AOI = 0, and IAM = 0 at AOI = 90.  This equation is only valid for 
    -90 <= aoi <= 90, therefore `iam` is constrained to 0.0 outside this
    interval.

    References
    ----------
    [1] N. Martin and J. M. Ruiz, "Calculation of the PV modules angular
    losses under field conditions by means of an analytical model", Solar
    Energy Materials & Solar Cells, vol. 70, pp. 25-38, 2001.

    [2] N. Martin and J. M. Ruiz, "Corrigendum to 'Calculation of the PV
    modules angular losses under field conditions by means of an
    analytical model'", Solar Energy Materials & Solar Cells, vol. 110,
    pp. 154, 2013.

    See Also
    --------
    iam.physical
    iam.ashrae
    iam.interp
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    aoi_input = aoi

    aoi = np.asanyarray(aoi)
    a_r = np.asanyarray(a_r)

    if np.any(np.less_equal(a_r, 0)):
        raise RuntimeError("The parameter 'a_r' cannot be zero or negative.")

    with np.errstate(invalid='ignore'):
        iam = (1 - np.exp(-cosd(aoi) / a_r)) / (1 - np.exp(-1 / a_r))
        iam = np.where(np.abs(aoi) >= 90.0, 0.0, iam)

    if isinstance(aoi_input, pd.Series):
        iam = pd.Series(iam, index=aoi_input.index)

    return iam


def interp(aoi, theta_ref, iam_ref, method='linear', normalize=True):
    r'''
    Determine the incidence angle modifier (IAM) by interpolating a set of
    reference values, which are usually measured values.

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector [degrees].

    theta_ref : numeric
        Vector of angles at which the IAM is known [degrees].

    iam_ref : numeric
        IAM values for each angle in ``theta_ref`` [unitless].

    method : str, default 'linear'
        Specifies the interpolation method.
        Useful options are: 'linear', 'quadratic','cubic'.
        See scipy.interpolate.interp1d for more options.

    normalize : boolean, default True
        When true, the interpolated values are divided by the interpolated
        value at zero degrees.  This ensures that ``iam``=1.0 at normal
        incidence.

    Returns
    -------
    iam : numeric
        The incident angle modifier(s) [unitless]

    Notes
    -----
    ``theta_ref`` must have two or more points and may span any range of
    angles. Typically there will be a dozen or more points in the range 0-90
    degrees. Beyond the range of ``theta_ref``, IAM values are extrapolated,
    but constrained to be non-negative.

    The sign of ``aoi`` is ignored; only the magnitude is used.

    See Also
    --------
    iam.physical
    iam.ashrae
    iam.martin_ruiz
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    from scipy.interpolate import interp1d

    # Scipy doesn't give the clearest feedback, so check number of points here.
    MIN_REF_VALS = {'linear': 2, 'quadratic': 3, 'cubic': 4, 1: 2, 2: 3, 3: 4}

    if len(theta_ref) < MIN_REF_VALS.get(method, 2):
        raise ValueError("Too few reference points defined "
                         "for interpolation method '%s'." % method)

    if np.any(np.less(iam_ref, 0)):
        raise ValueError("Negative value(s) found in 'iam_ref'. "
                         "This is not physically possible.")

    interpolator = interp1d(theta_ref, iam_ref, kind=method,
                            fill_value='extrapolate')
    aoi_input = aoi

    aoi = np.asanyarray(aoi)
    aoi = np.abs(aoi)
    iam = interpolator(aoi)
    iam = np.clip(iam, 0, None)

    if normalize:
        iam /= interpolator(0)

    if isinstance(aoi_input, pd.Series):
        iam = pd.Series(iam, index=aoi_input.index)

    return iam


def sapm(aoi, module, upper=None):
    r"""
    Determine the incidence angle modifier (IAM) using the SAPM model.

    Parameters
    ----------
    aoi : numeric
        Angle of incidence in degrees. Negative input angles will return
        zeros.

    module : dict-like
        A dict or Series with the SAPM IAM model parameters.
        See the :py:func:`sapm` notes section for more details.

    upper : None or float, default None
        Upper limit on the results.

    Returns
    -------
    iam : numeric
        The SAPM angle of incidence loss coefficient, termed F2 in [1].

    Notes
    -----
    The SAPM [1] traditionally does not define an upper limit on the AOI
    loss function and values slightly exceeding 1 may exist for moderate
    angles of incidence (15-40 degrees). However, users may consider
    imposing an upper limit of 1.

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    [2] B.H. King et al, "Procedure to Determine Coefficients for the
    Sandia Array Performance Model (SAPM)," SAND2016-5284, Sandia
    National Laboratories (2016).

    [3] B.H. King et al, "Recent Advancements in Outdoor Measurement
    Techniques for Angle of Incidence Effects," 42nd IEEE PVSC (2015).
    DOI: 10.1109/PVSC.2015.7355849
    """

    aoi_coeff = [module['B5'], module['B4'], module['B3'], module['B2'],
                 module['B1'], module['B0']]

    iam = np.polyval(aoi_coeff, aoi)
    iam = np.clip(iam, 0, upper)
    # nan tolerant masking
    aoi_lt_0 = np.full_like(aoi, False, dtype='bool')
    np.less(aoi, 0, where=~np.isnan(aoi), out=aoi_lt_0)
    iam = np.where(aoi_lt_0, 0, iam)

    if isinstance(aoi, pd.Series):
        iam = pd.Series(iam, aoi.index)

    return iam
