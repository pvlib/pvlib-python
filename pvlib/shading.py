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


def shaded_fraction1d(
    solar_zenith,
    solar_azimuth,
    axis_azimuth,
    shaded_row_rotation,
    *,
    collector_width,
    pitch,
    axis_tilt=0,
    surface_to_axis_offset=0,
    cross_axis_slope=0,
    shading_row_rotation=None,
):
    r"""
    Shaded fraction in the vertical dimension of tilted rows, or perpendicular
    to the axis of horizontal rows.

    If ``shading_row_rotation`` isn't provided, it is assumed that
    both the shaded row and the shading row (the one blocking the
    direct beam) have the same rotation and azimuth values.

    .. warning::
        The function assumes that the roles of the shaded and shading rows
        remain the same during the day. In the case where the shading and
        shaded rows change throughout the day, e.g. a N-S single-axis tracker,
        the inputs must be switched depending on the sign of the projected
        solar zenith angle. See the Examples section below.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    solar_zenith : numeric
        Solar zenith angle, in degrees.
    solar_azimuth : numeric
        Solar azimuth angle, in degrees.
    axis_azimuth : numeric
        Axis azimuth of the rotation axis of a tracker, in degrees.
        Fixed-tilt arrays can be considered as a particular case of a tracker.
        North=0º, South=180º, East=90º, West=270º.
    shaded_row_rotation : numeric
        Right-handed rotation of the row receiving the shade, with respect
        to ``axis_azimuth``. In degrees :math:`^{\circ}`.
    collector_width : numeric
        Vertical length of a tilted row. The returned ``shaded_fraction``
        is the ratio of the shadow over this value.
    pitch : numeric
        Axis-to-axis horizontal spacing of the row.
    axis_tilt : numeric, default 0
        Tilt of the rows axis from horizontal. In degrees :math:`^{\circ}`.
    surface_to_axis_offset : numeric, default 0
        Distance between the rotating axis and the collector surface.
        May be used to account for a torque tube offset.
    cross_axis_slope : numeric, default 0
        Angle of the plane containing the rows' axes from
        horizontal. Right-handed rotation with respect to the rows axes.
        In degrees :math:`^{\circ}`.
    shading_row_rotation : numeric, optional
        Right-handed rotation of the row casting the shadow, with respect
        to the row axis. In degrees :math:`^{\circ}`.

    Returns
    -------
    shaded_fraction : numeric
        The fraction of the collector width shaded by an adjacent row. A
        value of 1 is completely shaded and 0 is no shade.

    Notes
    -----
    All length parameters must have the same units.

    Parameters are defined as follow:

    .. figure:: ../../_images/Anderson_Jensen_2024_Fig3.png
        :alt: Diagram showing the two rows and the parameters of the model.

        Figure 3 of [1]_. See correspondence between this nomenclature and the
        function parameters in the table below.

    +------------------+----------------------------+---------------------+
    | Symbol           | Parameter                  | Units               |
    +==================+============================+=====================+
    | :math:`\theta_1` | ``shading_row_rotation``   |                     |
    +------------------+----------------------------+                     |
    | :math:`\theta_2` | ``shaded_row_rotation``    | Degrees             |
    +------------------+----------------------------+ :math:`^{\circ}`    |
    | :math:`\beta_c`  | ``cross_axis_slope``       |                     |
    +------------------+----------------------------+---------------------+
    | :math:`p`        | ``pitch``                  | Any consistent      |
    +------------------+----------------------------+ length unit across  |
    | :math:`\ell`     | ``collector_width``        | all these           |
    +------------------+----------------------------+ parameters, e.g.    |
    | :math:`z_0`      | ``surface_to_axis_offset`` | :math:`m`.          |
    +------------------+----------------------------+---------------------+
    | :math:`f_s`      | Return value               | Dimensionless       |
    +------------------+----------------------------+---------------------+

    Examples
    --------

    **Fixed-tilt south-facing array on flat terrain**

    Tilted row with a pitch of 3 m, a collector width of
    2 m, and row rotations of 30°. In the morning.

    >>> shaded_fraction1d(solar_zenith=80, solar_azimuth=135,
    ...     axis_azimuth=90, shaded_row_rotation=30, shading_row_rotation=30,
    ...     collector_width=2, pitch=3, axis_tilt=0,
    ...     surface_to_axis_offset=0.05, cross_axis_slope=0)
    0.47755694708090535

    **Fixed-tilt north-facing array on sloped terrain**

    Tilted row with a pitch of 4 m, a collector width of
    2.5 m, and row rotations of 50° for the shaded
    row and 30° for the shading row. The rows are on a
    10° slope, where their axis is on the most inclined
    direction (zero cross-axis slope). Shaded in the morning.

    >>> shaded_fraction1d(solar_zenith=80, solar_azimuth=75.5,
    ...     axis_azimuth=270, shaded_row_rotation=50, shading_row_rotation=30,
    ...     collector_width=2.5, pitch=4, axis_tilt=10,
    ...     surface_to_axis_offset=0.05, cross_axis_slope=0)
    0.793244836197256

    **N-S single-axis tracker on sloped terrain**

    Horizontal trackers with a pitch of 3 m, a collector width of
    1.4 m, and tracker rotations of 30° pointing east,
    in the morning. Terrain slope is 7° west-east (east-most
    tracker is higher than the west-most tracker).

    >>> shaded_fraction1d(solar_zenith=80, solar_azimuth=90, axis_azimuth=180,
    ...     shaded_row_rotation=-30, collector_width=1.4, pitch=3, axis_tilt=0,
    ...     surface_to_axis_offset=0.10, cross_axis_slope=7)
    0.8242176864434579

    Note the previous example only is valid for the shaded fraction of the
    west-most tracker in the morning, and assuming it is the
    shaded tracker during all the day is incorrect.
    During the afternoon, it is the one casting the shadow onto the
    east-most tracker.

    To calculate the shaded fraction for the east-most
    tracker, you must input the corresponding ``shaded_row_rotation``
    in the afternoon.

    >>> shaded_fraction1d(solar_zenith=80, solar_azimuth=270, axis_azimuth=180,
    ...     shaded_row_rotation=30, collector_width=1.4, pitch=3, axis_tilt=0,
    ...     surface_to_axis_offset=0.10, cross_axis_slope=7)
    0.018002567182254348

    You must switch the input/output depending on the
    sign of the projected solar zenith angle. See
    :py:func:`~pvlib.shading.projected_solar_zenith_angle` and the example
    :ref:`sphx_glr_gallery_shading_plot_shaded_fraction1d_ns_hsat_example.py`

    See also
    --------
    pvlib.shading.projected_solar_zenith_angle

    References
    ----------
    .. [1] Kevin S. Anderson, Adam R. Jensen; Shaded fraction and backtracking
        in single-axis trackers on rolling terrain. J. Renewable Sustainable
        Energy 1 March 2024; 16 (2): 023504. :doi:`10.1063/5.0202220`
    """
    # For nomenclature you may refer to [1].

    # rotation of row casting the shadow defaults to shaded row's one
    if shading_row_rotation is None:
        shading_row_rotation = shaded_row_rotation

    # projected solar zenith angle
    projected_solar_zenith = projected_solar_zenith_angle(
        solar_zenith,
        solar_azimuth,
        axis_tilt,
        axis_azimuth,
    )

    # calculate repeated elements
    thetas_1_S_diff = shading_row_rotation - projected_solar_zenith
    thetas_2_S_diff = shaded_row_rotation - projected_solar_zenith
    thetaS_rotation_diff = projected_solar_zenith - cross_axis_slope

    cos_theta_2_S_diff_abs = np.abs(cosd(thetas_2_S_diff))

    # Eq. (12) of [1]
    t_asterisk = (
        0.5
        + np.abs(cosd(thetas_1_S_diff)) / cos_theta_2_S_diff_abs / 2
        + (
            np.sign(projected_solar_zenith)
            * surface_to_axis_offset
            / collector_width
            / cos_theta_2_S_diff_abs
            * (sind(thetas_2_S_diff) - sind(thetas_1_S_diff))
        )
        - (
            pitch
            / collector_width
            * cosd(thetaS_rotation_diff)
            / cos_theta_2_S_diff_abs
            / cosd(cross_axis_slope)
        )
    )

    return np.clip(t_asterisk, 0, 1)


def direct_martinez(
    poa_global,
    poa_direct,
    shaded_fraction,
    shaded_blocks,
    total_blocks,
):
    r"""
    A shading loss power factor for non-monolithic silicon
    modules and arrays with an arbitrary number of bypass diodes.

    This experimental model reduces the direct and circumsolar
    irradiance reaching the module's cells based on the number of *blocks*
    affected by the shadow.
    More on blocks in the *Notes* section and in [1]_.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    poa_global : numeric
        Plane of array global irradiance. [W/m²].
    poa_direct : numeric
        Plane of array direct and circumsolar irradiance. [W/m²].
    shaded_fraction : numeric
        Fraction of module surface area that is shaded. [Unitless].
    shaded_blocks : numeric
        Number of blocks affected by the shadow. [Unitless].
        If a floating point number is provided, it will be rounded up.
    total_blocks : int
        Number of total blocks. Unitless.

    Returns
    -------
    shading_losses : numeric
        Fraction of DC power lost due to shading. [Unitless]

    Notes
    -----
    The implemented equations are (6) and (8) from [1]_:

    .. math::

        (1 - F_{ES}) = (1 - F_{GS}) \left(1 - \frac{N_{SB}}{N_{TB} + 1}\right)
        \quad \text{(6)}

        \left(1 - \frac{P_{S}}{P_{NS}}\right) = \left(1 -
        \frac{\left[(B + D^{CIR})(1 - F_{ES}) + D^{ISO} + R\right]}{G}\right)
        \quad \text{(8)}

    In (6), :math:`(1 - F_{ES})` is the correction factor to be multiplied by
    the direct and circumsolar irradiance, :math:`F_{GS}` is the shaded
    fraction of the collector, :math:`N_{SB}` is the number of shaded blocks
    and :math:`N_{TB}` is the number of total blocks.

    In (8), :math:`\frac{P_{S}}{P_{NS}}` is the fraction of DC power lost due
    to shading, :math:`P_{S}` is the power output of the shaded module,
    :math:`P_{NS}` is the power output of the non-shaded module,
    :math:`B + D^{CIR}` is the beam and circumsolar irradiance,
    :math:`D^{ISO} + R` is the sum of diffuse and albedo irradiances and
    :math:`G` is the global irradiance.

    **Blocks terminology:**

    A *block* is defined in [1]_ as a group of solar cells protected by a
    bypass diode. Also, a *block* is shaded when at least one of its
    cells is partially shaded.

    The total number of blocks and their layout depend on the module(s) used.
    Many manufacturers don't specify this information explicitly.
    However, these values can be inferred from:

    - the number of bypass diodes
    - where and how many junction boxes are present on the back of the module
    - whether or not the module is comprised of *half-cut cells*

    The latter two are heavily correlated.

    For example:

    1. A module with 1 bypass diode behaves as 1 block.
    2. A module with 3 bypass diodes and 1 junction box is likely to have 3
       blocks.
    3. A half-cut module with 3 junction boxes (split junction boxes) is
       likely to have 3x2 blocks. The number of blocks along the longest
       side of the module is 2 and along the shortest side is 3.
    4. A module without bypass diodes doesn't constitute a block, but may be
       part of one.

    Examples
    --------
    Minimal example. For a complete example, see
    :ref:`sphx_glr_gallery_shading_plot_martinez_shade_loss.py`.

    >>> import numpy as np
    >>> from pvlib import shading
    >>> total_blocks = 3  # blocks along the vertical of the module
    >>> POA_direct_and_circumsolar, POA_diffuse = 600, 80  # W/m²
    >>> POA_global = POA_direct_and_circumsolar + POA_diffuse
    >>> P_out_unshaded = 3000  # W
    >>> # calculation of the shaded fraction for the collector
    >>> shaded_fraction = shading.shaded_fraction1d(
    >>>     solar_zenith=80, solar_azimuth=180,
    >>>     axis_azimuth=90, shaded_row_rotation=25,
    >>>     collector_width=0.5, pitch=1, surface_to_axis_offset=0,
    >>>     cross_axis_slope=5.711, shading_row_rotation=50)
    >>> # calculation of the number of shaded blocks
    >>> shaded_blocks = np.ceil(total_blocks*shaded_fraction)
    >>> # apply the Martinez power losses to the calculated shading
    >>> loss_fraction = shading.direct_martinez(
    >>>     POA_global, POA_direct_and_circumsolar,
    >>>     shaded_fraction, shaded_blocks, total_blocks)
    >>> P_out_corrected = P_out_unshaded * (1 - loss_fraction)

    See Also
    --------
    shaded_fraction1d : to calculate 1-dimensional shaded fraction

    References
    ----------
    .. [1] F. Martínez-Moreno, J. Muñoz, and E. Lorenzo, 'Experimental model
        to estimate shading losses on PV arrays', Solar Energy Materials and
        Solar Cells, vol. 94, no. 12, pp. 2298-2303, Dec. 2010,
        :doi:`10.1016/j.solmat.2010.07.029`.
    """  # Contributed by Echedey Luis, 2024
    beam_factor = (  # Eq. (6) of [1]
        (1 - shaded_fraction)
        * (1 - np.ceil(shaded_blocks) / (1 + total_blocks))
    )
    return (  # Eq. (8) of [1]
        1
        - (
            poa_direct * beam_factor
            + (poa_global - poa_direct)  # diffuse and albedo
        )
        / poa_global
    )
