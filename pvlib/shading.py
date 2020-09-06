"""
The ``shading`` module contains functions that model module shading and the
associated effects on PV module output
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind, cosd


def masking_angle(surface_tilt, gcr, slant_height):
    """
    The elevation angle below which diffuse irradiance is blocked.

    The ``height`` parameter determines how far up the module's surface to
    evaluate the masking angle.  The lower the point, the steeper the masking
    angle [1]_.  SAM uses a "worst-case" approach where the masking angle
    is calculated for the bottom of the array (i.e. ``slant_height=0``) [2]_.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal [degrees].

    gcr : float
        The ground coverage ratio of the array [unitless].

    slant_height : numeric
        The distance up the module's slant height to evaluate the masking
        angle, as a fraction [0-1] of the module slant height [unitless].

    Returns
    -------
    mask_angle : numeric
        Angle from horizontal where diffuse light is blocked by the
        preceding row [degrees].

    See Also
    --------
    masking_angle_passias
    sky_diffuse_passias

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    # The original equation (8 in [1]) requires pitch and collector width,
    # but it's easy to non-dimensionalize it to make it a function of GCR
    # by factoring out B from the argument to arctan.
    numerator = (1 - slant_height) * sind(surface_tilt)
    denominator = 1/gcr - (1 - slant_height) * cosd(surface_tilt)
    phi = np.arctan(numerator / denominator)
    return np.degrees(phi)


def masking_angle_passias(surface_tilt, gcr):
    r"""
    The average masking angle over the slant height of a row.

    The masking angle is the angle from horizontal where the sky dome is
    blocked by the row in front. The masking angle is larger near the lower
    edge of a row than near the upper edge. This function calculates the
    average masking angle as described in [1]_.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal [degrees].

    gcr : float
        The ground coverage ratio of the array [unitless].

    Returns
    ----------
    mask_angle : numeric
        Average angle from horizontal where diffuse light is blocked by the
        preceding row [degrees].

    See Also
    --------
    masking_angle
    sky_diffuse_passias

    Notes
    -----
    The pvlib-python authors believe that Eqn. 9 in [1]_ is incorrect.
    Here we use an independent equation.  First, Eqn. 8 is non-dimensionalized
    (recasting in terms of GCR):

    .. math::

        \psi(z') = \arctan \left [
            \frac{(1 - z') \sin \beta}
                 {\mathrm{GCR}^{-1} + (z' - 1) \cos \beta}
        \right ]

    Where :math:`GCR = B/C` and :math:`z' = z/B`. The average masking angle
    :math:`\overline{\psi} = \int_0^1 \psi(z') \mathrm{d}z'` is then
    evaluated symbolically using Maxima (using :math:`X = 1/\mathrm{GCR}`):

    .. code-block:: none

        load(scifac)    /* for the gcfac function */
        assume(X>0, cos(beta)>0, cos(beta)-X<0);   /* X is 1/GCR */
        gcfac(integrate(atan((1-z)*sin(beta)/(X+(z-1)*cos(beta))), z, 0, 1))

    This yields the equation implemented by this function:

    .. math::

        \overline{\psi} = \
            &-\frac{X}{2} \sin\beta \log | 2 X \cos\beta - (X^2 + 1)| \\
            &+ (X \cos\beta - 1) \arctan \frac{X \cos\beta - 1}{X \sin\beta} \\
            &+ (1 - X \cos\beta) \arctan \frac{\cos\beta}{\sin\beta} \\
            &+ X \log X \sin\beta

    The pvlib-python authors have validated this equation against numerical
    integration of :math:`\overline{\psi} = \int_0^1 \psi(z') \mathrm{d}z'`.

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    """
    # wrap it in an array so that division by zero is handled well
    beta = np.radians(np.array(surface_tilt))
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)
    X = 1/gcr

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore beta=0
        term1 = -X * sin_b * np.log(np.abs(2 * X * cos_b - (X**2 + 1))) / 2
        term2 = (X * cos_b - 1) * np.arctan((X * cos_b - 1) / (X * sin_b))
        term3 = (1 - X * cos_b) * np.arctan(cos_b / sin_b)
        term4 = X * np.log(X) * sin_b

    psi_avg = term1 + term2 + term3 + term4
    # when beta=0, divide by zero makes psi_avg NaN.  replace with 0:
    psi_avg = np.where(np.isfinite(psi_avg), psi_avg, 0)

    if isinstance(surface_tilt, pd.Series):
        psi_avg = pd.Series(psi_avg, index=surface_tilt.index)

    return np.degrees(psi_avg)


def sky_diffuse_passias(masking_angle):
    r"""
    The diffuse irradiance loss caused by row-to-row sky diffuse shading.

    Even when the sun is high in the sky, a row's view of the sky dome will
    be partially blocked by the row in front. This causes a reduction in the
    diffuse irradiance incident on the module. The reduction depends on the
    masking angle, the elevation angle from a point on the shaded module to
    the top of the shading row. In [1]_ the masking angle is calculated as
    the average across the module height. SAM assumes the "worst-case" loss
    where the masking angle is calculated for the bottom of the array [2]_.

    This function, as in [1]_, makes the assumption that sky diffuse
    irradiance is isotropic.

    Parameters
    ----------
    masking_angle : numeric
        The elevation angle below which diffuse irradiance is blocked
        [degrees].

    Returns
    -------
    derate : numeric
        The fraction [0-1] of blocked sky diffuse irradiance.

    See Also
    --------
    masking_angle
    masking_angle_passias

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    return 1 - cosd(masking_angle/2)**2
