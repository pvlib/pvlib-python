"""
The ``shading`` module contains functions that model module shading and the
associated effects on PV module output
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind, cosd


def ground_angle(surface_tilt, gcr, slant_height):
    """
    Angle from horizontal of the line from a point on the row slant length
    to the bottom of the facing row.

    The angles are clockwise from horizontal, rather than the usual
    counterclockwise direction.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : float
        ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    slant_height : numeric
        The distance up the module's slant height to evaluate the ground
        angle, as a fraction [0-1] of the module slant height [unitless].

    Returns
    -------
    psi : numeric
        Angle [degree].
    """
    #  : \\            \
    #  :  \\            \
    #  :   \\            \
    #  :    \\            \  facing row
    #  :     \\.___________\
    #  :       \  ^*-.  psi \
    #  :        \  x   *-.   \
    #  :         \  v      *-.\
    #  :          \<-----P---->\

    x1 = gcr * slant_height * sind(surface_tilt)
    x2 = gcr * slant_height * cosd(surface_tilt) + 1
    psi = np.arctan2(x1, x2)  # do this before rad2deg because it handles 0 / 0
    return np.rad2deg(psi)


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
       :doi:`10.1016/0379-6787(84)90017-6`
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    # The original equation (8 in [1]) requires pitch and collector width,
    # but it's easy to non-dimensionalize it to make it a function of GCR
    # by factoring out B from the argument to arctan.
    numerator = gcr * (1 - slant_height) * sind(surface_tilt)
    denominator = 1 - gcr * (1 - slant_height) * cosd(surface_tilt)
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
       :doi:`10.1016/0379-6787(84)90017-6`
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
       :doi:`10.1016/0379-6787(84)90017-6`
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    return 1 - cosd(masking_angle/2)**2


def projected_solar_zenith_angle(solar_zenith, solar_azimuth,
                                 axis_tilt, axis_azimuth):
    r"""
    Calculate projected solar zenith angle in degrees.

    This solar zenith angle is projected onto the plane whose normal vector is
    defined by ``axis_tilt`` and ``axis_azimuth``. The normal vector is in the
    direction of ``axis_azimuth`` (clockwise from north) and tilted from
    horizontal by ``axis_tilt``. See Figure 5 in [1]_:

    .. figure:: ../../_images/Anderson_Mikofski_2020_Fig5.jpg
       :alt: Wire diagram of coordinates systems to obtain the projected angle.
       :align: center
       :scale: 50 %

       Fig. 5, [1]_: Solar coordinates projection onto tracker rotation plane.

    Parameters
    ----------
    solar_zenith : numeric
        Sun's apparent zenith in degrees.
    solar_azimuth : numeric
        Sun's azimuth in degrees.
    axis_tilt : numeric
        Axis tilt angle in degrees. From horizontal plane to array plane.
    axis_azimuth : numeric
        Axis azimuth angle in degrees.
        North = 0°; East = 90°; South = 180°; West = 270°

    Returns
    -------
    Projected_solar_zenith : numeric
        In degrees.

    Notes
    -----
    This projection has a variety of applications in PV. For example:

    - Projecting the sun's position onto the plane perpendicular to
      the axis of a single-axis tracker (i.e. the plane
      whose normal vector coincides with the tracker torque tube)
      yields the tracker rotation angle that maximizes direct irradiance
      capture. This tracking strategy is called *true-tracking*. Learn more
      about tracking in
      :ref:`sphx_glr_gallery_solar-tracking_plot_single_axis_tracking.py`.

    - Self-shading in large PV arrays is often modeled by assuming
      a simplified 2-D array geometry where the sun's position is
      projected onto the plane perpendicular to the PV rows.
      The projected zenith angle is then used for calculations
      regarding row-to-row shading.

    Examples
    --------
    Calculate the ideal true-tracking angle for a horizontal north-south
    single-axis tracker:

    >>> rotation = projected_solar_zenith_angle(solar_zenith, solar_azimuth,
    >>>                                         axis_tilt=0, axis_azimuth=180)

    Calculate the projected zenith angle in a south-facing fixed tilt array
    (note: the ``axis_azimuth`` of a fixed-tilt row points along the length
    of the row):

    >>> psza = projected_solar_zenith_angle(solar_zenith, solar_azimuth,
    >>>                                     axis_tilt=0, axis_azimuth=90)

    References
    ----------
    .. [1] K. Anderson and M. Mikofski, 'Slope-Aware Backtracking for
       Single-Axis Trackers', National Renewable Energy Lab. (NREL), Golden,
       CO (United States);
       NREL/TP-5K00-76626, Jul. 2020. :doi:`10.2172/1660126`.

    See Also
    --------
    pvlib.solarposition.get_solarposition
    """
    # Assume the tracker reference frame is right-handed. Positive y-axis is
    # oriented along tracking axis; from north, the y-axis is rotated clockwise
    # by the axis azimuth and tilted from horizontal by the axis tilt. The
    # positive x-axis is 90 deg clockwise from the y-axis and parallel to
    # horizontal (e.g., if the y-axis is south, the x-axis is west); the
    # positive z-axis is normal to the x and y axes, pointed upward.

    # Since elevation = 90 - zenith, sin(90-x) = cos(x) & cos(90-x) = sin(x):
    # Notation from [1], modified to use zenith instead of elevation
    # cos(elevation) = sin(zenith) and sin(elevation) = cos(zenith)
    # Avoid recalculating these values
    sind_solar_zenith = sind(solar_zenith)
    cosd_axis_azimuth = cosd(axis_azimuth)
    sind_axis_azimuth = sind(axis_azimuth)
    sind_axis_tilt = sind(axis_tilt)

    # Sun's x, y, z coords
    sx = sind_solar_zenith * sind(solar_azimuth)
    sy = sind_solar_zenith * cosd(solar_azimuth)
    sz = cosd(solar_zenith)
    # Eq. (4); sx', sz' values from sun coordinates projected onto surface
    sx_prime = sx * cosd_axis_azimuth - sy * sind_axis_azimuth
    sz_prime = (
        sx * sind_axis_azimuth * sind_axis_tilt
        + sy * sind_axis_tilt * cosd_axis_azimuth
        + sz * cosd(axis_tilt)
    )
    # Eq. (5); angle between sun's beam and surface
    theta_T = np.degrees(np.arctan2(sx_prime, sz_prime))
    return theta_T


def shaded_fraction1d(solar_zenith, solar_azimuth, surface_tilt,
                      surface_azimuth, gcr, cross_axis_slope=0):
    r"""
    Shaded fraction in the vertical dimension of the rows.

    Assumes both the shaded row and the one blocking the direct beam share
    the same tilt and azimuth values.

    Parameters
    ----------
    solar_zenith : numeric
        Solar position zenith, in degrees.
    solar_azimuth : numeric
        Solar position azimuth, in degrees.
    surface_tilt : numeric
        In degrees.
    surface_azimuth : numeric
        In degrees. North=0º, South=180º, East=90º, West=270º.
    gcr : numeric
        The ground coverage ratio as a fraction equal to the collector width
        over the horizontal row-to-row pitch.
    cross_axis_slope : numeric, default 0
        Angle of the plane containing the rows' axes in degrees from
        horizontal. A row axis is defined by the vector product of
        ``surface_tilt`` and ``surface_azimuth``.

    Returns
    -------
    shaded_fraction : numeric
        The fraction of the collector width shaded by an adjacent row. A
        value of 1 is completely shaded and zero is no shade.

    See also
    --------
    pvlib.shading.linear_shade_loss


    The shaded fraction is derived using trigonometry and similar triangles
    from the tracker rotation :math:`\beta`, the ground slope :math:`\theta_g`,
    the projected solar zenith (psz) :math:`\theta`, the collector width
    :math:`L`, the row-to-row pitch :math:`P`, and the shadow length :math:`z`
    as shown in the image below.

    .. image:: /_images/FSLR_irrad_shade_loss_slope_terrain.png
       :alt: Cross-section of two arrays on a sloped terrain and the resulting
             shade.

    The ratio of the shadow length to the pitch, :math:`z/P`, is given by the
    following relation where the ground coverage ratio (GCR) is :math:`L/P`:

    .. math::
       \frac{z/P}{\sin{\left(\frac{\pi}{2}-\beta+\theta\right)}}
       = \frac{GCR}{\sin{\left(\frac{\pi}{2}-\theta-\theta_g\right)}}

    Then the shaded fraction :math:`w/L` is derived from :math:`z/P` as
    follows:

    .. math::
       \frac{w}{L} = 1 - \frac{P}{z\cos{\theta_g}}

    Finally, shade is zero if :math:`z\cos{\theta_g}/P \le 1`.

    References
    ----------
    .. [1] Mark A. Mikofski, "First Solar Irradiance Shade Losses on Sloped
       Terrain," PVPMC, 2023
    """
    theta_g_rad = np.radians(cross_axis_slope)
    # projected solar zenith:
    # consider the angle the sun direct beam has on the vertical plane which
    # contains the tracker normal vector, with respect to a horizontal line
    projected_solar_zenith = projected_solar_zenith_angle(
        solar_zenith,
        solar_azimuth,
        0,  # no rotation from the horizontal
        # the vector that defines the projection plane for prior conditions
        surface_azimuth-90,
    )
    # angle opposite shadow cast on the ground, z
    angle_z = (
        np.pi / 2 - np.radians(surface_tilt)
        + np.radians(projected_solar_zenith))
    # angle opposite the collector width, L
    angle_gcr = (
        np.pi / 2 - np.radians(projected_solar_zenith)
        - theta_g_rad)
    # ratio of shadow, z, to pitch, P
    zp = gcr * np.sin(angle_z) / np.sin(angle_gcr)
    # there's only row-to-row shade loss if the shadow on the ground, z, is
    # longer than row-to-row pitch projected on the ground, P*cos(theta_g)
    zp_cos_g = zp*np.cos(theta_g_rad)
    # shaded fraction (sf)
    sf = np.where(zp_cos_g <= 1, 0, 1 - 1/zp_cos_g)
    return sf


def linear_shade_loss(shaded_fraction, diffuse_fraction):
    """
    Fraction of power lost to linear shade loss applicable to monolithic thin
    film modules like First Solar CdTe, where the shadow is perpendicular to
    cell scribe lines.

    Parameters
    ----------
    shaded_fraction : numeric
        The fraction of the collector width shaded by an adjacent row. A
        value of 1 is completely shaded and zero is no shade.
    diffuse_fraction : numeric
        The ratio of diffuse plane of array (poa) irradiance to global poa.
        A value of 1 is completely diffuse and zero is no diffuse.

    Returns
    -------
    linear_shade_loss : numeric
        The fraction of power lost due to linear shading. A value of 1 is all
        power lost and zero is no loss.

    See also
    --------
    pvlib.shading.tracker_shaded_fraction

    Example
    -------
    >>> from pvlib import shading
    >>> sf = shading.tracker_shaded_fraction(45.0, 0.8, 45.0, 0)
    >>> loss = shading.linear_shade_loss(sf, 0.2)
    >>> P_no_shade = 100  # [kWdc]  DC output from modules
    >>> P_linear_shade = P_no_shade * (1-loss)  # [kWdc] output after loss
    # 90.71067811865476 [kWdc]
    """
    return shaded_fraction * (1 - diffuse_fraction)
