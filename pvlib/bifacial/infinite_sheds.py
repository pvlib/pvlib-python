r"""
Infinite Sheds
==============

The "infinite sheds" model is a 2-dimensional model of an array that assumes
rows are long enough that edge effects are negligible and therefore can be
treated as infinite. The infinite sheds model considers an array of adjacent
rows of PV modules versus just a single row. It is also capable of considering
both mono and bifacial modules. Sheds are defined as either fixed tilt or
trackers with uniform GCR on a horizontal plane. To consider arrays on an
non-horizontal planes, rotate the solar vector into the reference frame of the
sloped plane. The main purpose of the infinite shdes model is to modify the
plane of array irradiance components to account for adjancent rows that reduce
incident irradiance on the front and back sides versus a single isolated row
that can see the entire sky as :math:`(1+cos(\beta))/2` and ground as
:math:`(1-cos(\beta))/2`.

Therefore the model picks up after the transposition of diffuse and direct onto
front and back surfaces with the following steps:
1. Find the fraction of unshaded ground between rows, ``f_gnd_beam``. We assume
   there is no direct irradiance in the shaded fraction ``1 - f_gnd_beam``.
2. Calculate the view factor,``fz_sky``, of diffuse sky incident on ground
   between rows and not blocked by adjacent rows. This differs from the single
   row model which assumes ``f_gnd_beam = 1`` or that the ground can see the
   entire sky, and therefore, the ground reflected is just the product of GHI
   and albedo. Note ``f_gnd_beam`` also considers the diffuse sky visible
   between neighboring rows in front and behind the current row. If rows are
   higher off the ground, then the sky might be visible between multiple rows!
3. Calculate the view factor of the ground reflected irradiance incident on PV
    surface.
4. Find the fraction of PV surface shaded. We assume only diffuse in the shaded
   fraction. We treat these two sections differently and assume that the view
   factors of the sky and ground are linear in each section.
5. Sum up components of diffuse sky, diffuse ground, and direct on the front
   and  back PV surfaces.
6. Apply the bifaciality factor to the backside and combine with the front.
7. Treat the first and last row differently, because they aren't blocked on the
   front side for 1st row, or the backside for last row.

# TODO: explain geometry: primary axes and orientation, what is meant by
"previous" and "next rows", etc.


That's it folks! This model is influenced by the 2D model published by Marion,
*et al.* in [1].

References
----------
[1] A Practical Irradiance Model for Bifacial PV Modules, Bill Marion, et al.,
IEEE PVSC 2017
[2] Bifacial Performance Modeling in Large Arrays, Mikofski, et al., IEEE PVSC
2018
"""

from collections import OrderedDict
import numpy as np
import pandas as pd
from pvlib import irradiance, iam
from pvlib.tools import cosd, sind, tand
from pvlib.bifacial import utils
from pvlib.shading import shaded_fraction


def _gcr_prime(gcr, height, surface_tilt, pitch):
    """
    Slant length from the ground to the top of a row divided by the row pitch.

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch).
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    gcr_prime : numeric
        ground coverage ratio including height above ground
    """

    #  : \\                      \\
    #  :  \\                      \\
    #  :   \\ H = module length    \\
    #  :    \\                      \\
    #  :.....\\......................\\........ module lower edge
    #  :       \                       \    :
    #  :        \                       \   h = height above ground
    #  :         \                 tilt  \  :
    #  +----------\<---------P----------->\---- ground
    # TODO convert to degrees
    return gcr + height / sind(surface_tilt) / pitch


# TODO: overlaps with ground_sky_angles_prev in that both return
# angle to top of previous row. Could the three ground_sky_angle_xxx functions
# be combined and handle the cases of points behind the "previous" row or ahead
# of the next row?
def _ground_sky_angles(f_z, gcr, height, surface_tilt, pitch):
    """
    Angles from point z on ground to tops of next and previous rows.
    
    The point z lies between the extension to the ground of the previous row
    and the extension to the ground of the next row.

    .. math::
        \\tan{\\psi_0} = \\frac{\\sin{\\beta^\\prime}}{\\frac{F_z}
        {\\text{GCR}^\\prime} + \\cos{\\beta^\\prime}}

        \\tan{\\psi_1} = \\frac{\\sin{\\beta}}{\\frac{F_z^\\prime}
        {\\text{GCR}^\\prime} + \\cos{\\beta}}

    Parameters
    ----------
    f_z : numeric
        fraction of ground from previous to next row
    gcr : numeric
        ground coverage ratio, ratio of row slant length to row spacing
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    psi_0 : numeric
        Angle from horizontal of the line between a point on the ground and
        the top of the previous row. [degree]
    psi_1 : numeric
        Angle from horizontal of the line between a point on the ground and
        the top of the next row. [degree]

    Notes
    -----
    Assuming the first row is in the front of the array then previous rows are
    toward the front of the array and next rows are toward the back.
    
    See Also
    --------
    _ground_sky_angles_prev
    _ground_sky_angles_next

    """

    #  : \\*                    |\\             front of array
    #  :  \\ **                 | \\
    # next \\   **               | \\ previous row
    # row   \\     **            |  \\
    #  :.....\\.......**..........|..\\........ module lower edge
    #  :       \         **       |    \    :
    #  :        \           **     |    \   h = height above ground
    #  :   tilt  \      psi1   **  |psi0 \  :
    #  +----------\<---------P----*+----->\---- ground
    #             1<-----1-fz-----><--fz--0---- fraction of ground

    gcr_prime = _gcr_prime(gcr, height, surface_tilt, pitch)
    tilt_prime = 180. - surface_tilt
    opposite_side = sind(tilt_prime)
    adjacent_side = f_z/gcr_prime + cosd(tilt_prime)
    # tan_psi_0 = opposite_side / adjacent_side
    psi_0 = np.rad2deg(np.arctan2(opposite_side, adjacent_side))
    f_z_prime = 1 - f_z
    opposite_side = np.sin(surface_tilt)
    adjacent_side = f_z_prime/gcr_prime + cosd(surface_tilt)
    # tan_psi_1 = opposite_side / adjacent_side
    psi_1 = np.rad2deg(np.arctan2(opposite_side, adjacent_side))
    return psi_0, psi_1


def _ground_sky_angles_prev(f_z, gcr, height, surface_tilt, pitch):
    """
    Angles from point z on ground to bottom of previous row and to the top of
    the row beyond the previous row.

    The point z lies between the extension to the ground of the previous row
    and the extension to the ground of the next row.
    
    The function _ground_sky_angles_prev applies when the sky is visible
    between the bottom of the previous row, and the top of the row in front
    of the previous row.

    .. math::

        \\tan{\\psi_0} = \\frac{\\sin{\\beta^\\prime}}{\\frac{F_z}
        {\\text{GCR}^\\prime} + \\cos{\\beta^\\prime}}

        0 < F_z < F_{z0,limit}

        \\tan \\psi_1 = \\frac{h}{\\frac{h}{\\tan\\beta} - z}

    Parameters
    ----------
    f_z : numeric
        fraction of ground from previous to next row
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    psi_0 : numeric
        Angle from horizontal of the line between a point on the ground and
        the bottom of the previous row. [degree]
    psi_1 : numeric
        Angle from horizontal of the line between a point on the ground and
        the top of the row in front of the previous row. [degree]

    Notes
    -----
    Assuming the first row is in the front of the array then previous rows are
    toward the front of the array and next rows are toward the back.
    
    See Also
    --------
    _ground_sky_angles
    _ground_sky_angles_next

    """

    #  : \\        |            *\\ top of previous row
    #  :  \\      |          **   \\
    # prev \\    |         *       \\           front of array
    # row   \\  |       **          \\
    # bottom.\\|......*..............\\........ module lower edge
    #  :      |\   **                  \    :
    #  psi1  |  \* psi0                 \   h = height above ground
    #  :    | ** \                       \  :
    #  +---+*-----\<---------P----------->\---- ground
    #      <-1+fz-1<---------fz=1---------0---- fraction of ground

    gcr_prime = _gcr_prime(gcr, height, surface_tilt, pitch)
    tilt_prime = 180. - surface_tilt
    # angle to top of previous panel beyond the current row
    psi_0 = np.rad2deg(np.arctan2(
        sind(tilt_prime), (1+f_z)/gcr_prime + cosd(tilt_prime)))
    # angle to bottom of previous panel
    z = f_z*pitch
    # other forms raise division by zero errors
    # avoid division by zero errors
    psi_1 = np.rad2deg(np.arctan2(height, height/tand(surface_tilt) - z))
    return psi_0, psi_1


def _f_z0_limit(gcr, height, surface_tilt, pitch):
    """
    Limit from the ground where sky is visible between previous rows.

    .. math::
        F_{z0,limit} = \\frac{h}{P} \\left(
        \\frac{1}{\\tan \\beta} + \\frac{1}{\\tan \\psi_t}\\right)

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    z1 : numeric
        The point on the ground, :math:`z_0`, which is on the line
        tangent to the bottom of the previous row and the top of the row in
        front of the previous row.
    """
    _, tan_psi_t_x0 = _sky_angle(gcr, surface_tilt, 0.0)
    # tan_psi_t_x0 = gcr * np.sin(tilt) / (1.0 - gcr * np.cos(tilt))
    return height / pitch * (1. / tand(surface_tilt) + 1. / tan_psi_t_x0)


def _ground_sky_angles_next(f_z, gcr, height, surface_tilt, pitch):
    """
    Angles from point z on the ground to bottom of the next row and to the top
    of the row behind the next row.

    The point z lies between the extension to the ground of the previous row
    and the extension to the ground of the next row.
    
    The function _ground_sky_angles_next applies when the sky is visible
    between the bottom of the next row, and the top of the row behind
    of the next row.

    .. math::
        \\tan \\psi_0 = \\frac{h}{\\frac{h}{\\tan\\beta^\\prime}
        - \\left(P-z\\right)}

        \\tan{\\psi_1} = \\frac{\\sin{\\beta}}
        {\\frac{F_z^\\prime}{\\text{GCR}^\\prime} + \\cos{\\beta}}

    Parameters
    ----------
    f_z : numeric
        fraction of ground from previous to next row
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    psi_0 : numeric
        Angle from horizontal of the line between a point on the ground and
        the bottom of the previous row. [degree]
    psi_1 : numeric
        Angle from horizontal of the line between a point on the ground and
        the top of the row in front of the previous row. [degree]

    Notes
    -----
    Assuming the first row is in the front of the array then previous rows are
    toward the front of the array and next rows are toward the back.
    
    See Also
    --------
    _ground_sky_angles
    _ground_sky_angles_prev

    """

    #  : \\+           _         \\
    #  :  \\  `*+        - _      \\
    # next \\      `*+        _    \\
    # row   \\          `*+     -_  \\ next row bottom
    # top....\\..............`*+...-.\\_
    #  :       \                  `*+  \ -_  psi0
    #  :        \                psi1  `*+  -_
    #  :         \                       \  `*+ _
    #  +----------\<---------P----------->\------*----- ground
    #             1<---------fz=1---------0-1-fz->----- fraction of ground

    gcr_prime = _gcr_prime(gcr, height, surface_tilt, pitch)
    tilt_prime = 180. - surface_tilt
    # angle to bottom of next panel
    fzprime = 1 - f_z
    zprime = fzprime*pitch
    # other forms raise division by zero errors
    # avoid division by zero errors
    psi_0 = np.rad2deg(np.arctan2(height, height/tand(tilt_prime) - zprime))
    # angle to top of next panel beyond the current row
    psi_1 = np.rad2deg(np.arctan2(
        cosd(surface_tilt), (1 + fzprime)/gcr_prime + cosd(surface_tilt)))
    return psi_0, psi_1


def _f_z1_limit(gcr, height, surface_tilt, pitch):
    """
    Limit from the ground where sky is visible between the next row and the
    row behind the next row.

    .. math::
        F_{z1,limit} = \\frac{h}{P} \\left(
        \\frac{1}{\\tan \\psi_t} - \\frac{1}{\\tan \\beta}\\right)

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing

    Returns
    -------
    z1 : numeric
        The point on the ground, :math:`z_1^\\prime`, which is on the line
        tangent to the bottom of the next row and the top of the row behind
        the next row.
    """
    tan_psi_t_x1 = _sky_angle(gcr, 180. - surface_tilt, 0.0)
    # tan_psi_t_x1 = gcr * np.sin(pi-tilt) / (1.0 - gcr * np.cos(pi-tilt))
    return height / pitch * (1. / tan_psi_t_x1 - 1. / tand(surface_tilt))


def _calc_fz_sky(psi_0, psi_1):
    """
    Calculate the view factor for point "z" on the ground to the visible
    diffuse sky subtended by the angles :math:`\\psi_0` and :math:`\\psi_1`.

    Parameters
    ----------
    psi_0 : numeric
        angle from ground to sky before point "z". [degree]
    psi_1 : numeric
        angle from ground to sky after point "z". [degree]

    Returns
    -------
    fz_sky : numeric
        fraction of energy from the diffuse sky dome that is incident on the
        ground at point "z"
    """
    return (cosd(psi_0) + cosd(psi_1)) / 2.


# TODO: move to util?
# TODO: add argument to set number of rows, default is infinite
# TODO: add option for first or last row, default is middle row
def _ground_sky_diffuse_view_factor(gcr, height, surface_tilt, pitch,
                                    npoints=100):
    """
    Calculate the view factor from each point on the ground between adjacent,
    interior rows, to the sky.
    
    The view factor is equal to the fraction of sky hemisphere visible at each
    point on the ground.

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing
    npoints : int, default 100
        Number of points used to discretize distance along the ground.

    Returns
    -------
    fz : ndarray
        Fraction of distance from the previous row to the next row. [unitless]
    fz_sky : ndarray
        View factors at discrete points between adjacent, interior rows.
        [unitless]
        
    """
    args = gcr, height, surface_tilt, pitch
    fz0_limit = _f_z0_limit(*args)
    fz1_limit = _f_z1_limit(*args)
    # include extra space to account for sky visible from adjacent rows
    # divide ground between visible limits into 3x npoints
    fz = np.linspace(
        0.0 if (1 - fz1_limit) > 0 else (1 - fz1_limit),
        1.0 if fz0_limit < 1 else fz0_limit,
        3*npoints)
    # calculate the angles psi_0 and psi_1 that subtend the sky visible
    # from between rows
    psi_z = _ground_sky_angles(fz, *args)
    # front edge
    psi_z0 = _ground_sky_angles_prev(fz, *args)
    fz_sky_next = _calc_fz_sky(*psi_z0)
    fz0_sky_next = []
    prev_row = 0.0
    # loop over rows by adding 1.0 to fz until prev_row < ceil(fz0_limit)
    while (fz0_limit - prev_row) > 0:
        fz0_sky_next.append(np.interp(fz + prev_row, fz, fz_sky_next))
        prev_row += 1.0
    # back edge
    psi_z1 = _ground_sky_angles_next(fz, *args)
    fz_sky_prev = _calc_fz_sky(*psi_z1)
    fz1_sky_prev = []
    next_row = 0.0
    # loop over rows by subtracting 1.0 to fz until next_row < ceil(fz1_limit)
    while (fz1_limit - next_row) > 0:
        fz1_sky_prev.append(np.interp(fz - next_row, fz, fz_sky_prev))
        next_row += 1.0
    # calculate the view factor of the sky from the ground at point z
    fz_sky = (
        _calc_fz_sky(*psi_z)  # current row
        + np.sum(fz0_sky_next, axis=0)  # sum of all next rows
        + np.sum(fz1_sky_prev, axis=0))  # sum of all previous rows
    # we just need one row, fz in range [0, 1]
    fz_row = np.linspace(0, 1, npoints)
    return fz_row, np.interp(fz_row, fz, fz_sky)


def _vf_ground_sky(gcr, height, surface_tilt, pitch, npoints=100):
    """
    Integrated and per-point view factors from the ground to the sky at points
    between interior rows of the array.

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing
    npoints : int, default 100
        Number of points used to discretize distance along the ground.

    Returns
    -------
    fgnd_sky : float
        Integration of view factors over the length between adjacent, interior
        rows. [unitless]
    fz : ndarray
        Fraction of distance from the previous row to the next row. [unitless]
    fz_sky : ndarray
        View factors at discrete points between adjacent, interior rows.
        [unitless]

    """
    args = gcr, height, surface_tilt, pitch
    # calculate the view factor of the diffuse sky from the ground between rows
    z_star, fz_sky = _ground_sky_diffuse_view_factor(*args, npoints=npoints)

    # calculate the integrated view factor for all of the ground between rows
    fgnd_sky = np.trapz(fz_sky, z_star)

    return fgnd_sky, z_star, fz_sky


# TODO: not used
def _calc_fgndpv_zsky(fx, gcr, height, tilt, pitch, npoints=100):
    """
    Calculate the fraction of diffuse irradiance from the sky, reflecting from
    the ground, incident at a point "x" on the PV surface.

    Parameters
    ----------
    fx : numeric
        fraction of PV surface from bottom
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing
    npoints : int
        divide the ground into discrete points
    """
    args = gcr, height, tilt, pitch

    # calculate the view factor of the diffuse sky from the ground between rows
    # and integrate the view factor for all of the ground between rows
    fgnd_sky, _, _ = _vf_ground_sky(*args, npoints=npoints)

    # if fx is zero, point x is at the bottom of the row, psi_x_bottom is zero,
    # and all of the ground is visible, so the view factor is just
    # Fgnd_pv = (1 - cos(tilt)) / 2
    if fx == 0:
        psi_x_bottom = 0.0
    else:
        # how far on the ground can the point x see?
        psi_x_bottom, _ = _ground_angle(gcr, tilt, fx)

    # max angle from pv surface perspective
    psi_max = tilt - psi_x_bottom

    fgnd_pv = (1 - np.cos(psi_max)) / 2
    fskyz = fgnd_sky * fgnd_pv
    return fskyz, fgnd_pv


def _diffuse_fraction(ghi, dhi):
    """
    ratio of DHI to GHI

    Parameters
    ----------
    ghi : numeric
        global horizontal irradiance (GHI) in W/m^2
    dhi : numeric
        diffuse horizontal irradiance (DHI) in W/m^2

    Returns
    -------
    df : numeric
        diffuse fraction
    """
    return dhi / ghi


# TODO: docstring
def _poa_ground_sky(poa_ground, f_gnd_beam, df, vf_gnd_sky):
    """
    transposed ground reflected diffuse component adjusted for ground
    illumination AND accounting for infinite adjacent rows in both directions

    Parameters
    ----------
    poa_ground : numeric
        transposed ground reflected diffuse component in W/m^2
    f_gnd_beam : numeric
        fraction of interrow ground illuminated (unshaded)
    df : numeric
        ratio of DHI to GHI
    vf_gnd_sky : numeric
        fraction of diffuse sky visible from ground integrated from row to row

    Returns
    -------
    poa_gnd_sky : numeric
        adjusted irradiance on modules reflected from ground
    """
    # split the ground into shaded and unshaded sections with f_gnd_beam
    # the shaded sections only see DHI, while unshaded see GHI = DNI*cos(ze)
    # + DHI, the view factor vf_gnd_sky only applies to the shaded sections
    # see Eqn (2) "Practical Irradiance Model for Bifacial PV" Marion et al.
    # unshaded  + (DHI/GHI)*shaded
    # f_gnd_beam + (DHI/GHI)*(1 - f_gnd_beam)
    # f_gnd_beam + df       *(1 - f_gnd_beam)
    # f_gnd_beam + df - df*f_gnd_beam
    # f_gnd_beam - f_gnd_beam*df + df
    # f_gnd_beam*(1 - df)          + df
    # unshaded *(DNI*cos(ze)/GHI) + DHI/GHI
    # only apply diffuse sky view factor to diffuse component (df) incident on
    # ground between rows, not the direct component of the unshaded ground
    df = np.where(np.isfinite(df), df, 0.0)
    return poa_ground * (f_gnd_beam*(1 - df) + df*vf_gnd_sky)


def _sky_angle(gcr, surface_tilt, x):
    """
    Angle from from a point x along the module slant height to the
    top of the facing row.

    Parameters
    ----------
    gcr : numeric
        Ratio of row slant length to row spacing (pitch). [unitless]
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    x : numeric
        Fraction of slant length from row bottom edge. [unitless]

    Returns
    -------
    psi : numeric
        Angle. [degree]
    tan_psi
        Tangent of angle. [unitless]
    """
    #  : \\                    .*\\
    #  :  \\               .-*    \\
    #  :   \\          .-*         \\
    #  :    \\   . .*+  psi         \\  facing row
    #  :     \\.-*__________________ \\
    #  :       \  ^                    \
    #  :        \  x                    \
    #  :         \  v                    \
    #  :          \<---------P----------->\

    y = 1.0 - x
    x1 = y * sind(surface_tilt)
    x2 = (1/gcr - y * sind(surface_tilt))
    tan_psi_top = x1 / x2
    psi_top = np.rad2deg(np.arctan2(x1, x2))
    return psi_top, tan_psi_top


def _f_sky_diffuse_pv(surface_tilt, tan_psi_top, tan_psi_top_0):
    """
    View factors of the sky from the shaded and unshaded parts of the row slant
    length.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    tan_psi_top : numeric
        Tangent of angle between horizontal and the line from the shade line
        on the current row to the top of the facing row. [degree]
    tan_psi_top_0 : numeric
        Tangent of angle between horizontal and the line from the bottom of the
        current row to the top of the facing row. [degree]

    Returns
    -------
    f_sky_pv_shade : numeric
        View factor from the shaded part of the row to the sky. [unitless]
    f_sky_pv_noshade : numeric
        View factor from the unshaded part of the row to the sky. [unitless]

    Notes
    -----
    Assuming the view factor various roughly linearly from the top to the
    bottom of the slant height, we can take the average to get integrated view
    factor. We'll average the shaded and unshaded regions separately to improve
    the approximation slightly.

    .. math ::
        \\large{F_{sky \\rightarrow shade} = \\frac{ 1 + \\frac{\\cos
        \\left(\\psi_t + \\beta \\right) + \\cos \\left(\\psi_t
        \\left(x=0\\right) + \\beta \\right)}{2}  }{ 1 + \\cos \\beta}}

    The view factor from the top of the row is one because it's view is not
    obstructed.

    .. math::
        \\large{F_{sky \\rightarrow no\\ shade} = \\frac{1 + \\frac{1 +
        \\cos \\left(\\psi_t + \\beta \\right)}{1 + \\cos \\beta} }{2}}
    """
    # TODO: don't average, return fsky-pv vs. x point on panel
    tilt = np.deg2rad(surface_tilt)
    psi_top = np.arctan(np.deg2rad(tan_psi_top))
    psi_top_0 = np.arctan(np.deg2rad(tan_psi_top_0))
    f_sky_pv_shade = (
        (1 +
         (np.cos(psi_top + tilt) + np.cos(psi_top_0 + tilt))
         / 2 / (1 + np.cos(tilt))))

    f_sky_pv_noshade = (1 + (
        1 + np.cos(psi_top + tilt)) / (1 + np.cos(tilt))) / 2
    return f_sky_pv_shade, f_sky_pv_noshade


def _poa_sky_diffuse_pv(poa_sky_diffuse, f_x, f_sky_pv_shade,
                        f_sky_pv_noshade):
    """
    Sky diffuse POA from average view factor weighted by shaded and unshaded
    parts of the surface.

    Parameters
    ----------
    poa_sky_diffuse : numeric
        sky diffuse irradiance on the plane of array (W/m^2)
    f_x : numeric
        shade line fraction from bottom of module
    f_sky_pv_shade : numeric
        fraction of sky visible from shaded part of PV surface
    f_sky_pv_noshade : numeric
        fraction of sky visible from unshaded part of PV surface

    Returns
    -------
    poa_sky_diffuse_pv : numeric
        total sky diffuse irradiance incident on PV surface
    """
    return poa_sky_diffuse * (f_x*f_sky_pv_shade + (1 - f_x)*f_sky_pv_noshade)


def _ground_angle(gcr, surface_tilt, x):
    """
    Angle from horizontal to the line from a point x on the row slant length
    to the bottom of the facing row.

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    x : numeric
        fraction of row slant length from bottom, ``x = 0`` is at the row
        bottom, ``x = 1`` is at the top of the row.

    Returns
    -------
    psi : numeric
        Angle [degree].
    tan_psi : numeric
        Tangent of angle [unitless]
    """
    x1 = x * sind(surface_tilt)
    x2 = (x * cosd(surface_tilt) + 1/gcr)
    tan_psi = x1 / x2
    psi = np.rad2deg(np.arctan2(x1, x2))
    return psi, tan_psi


def _f_ground_pv(surface_tilt, psi_shade, psi_bottom):
    """
    View factors of ground from shaded and unshaded parts of a row.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    psi_shade : numeric
        Angle from shade line to bottom of facing row. [degree]
    psi_bottom : numeric
        Angle from row top to bottom of the facing row. [degree]

    Returns
    -------
    f_gnd_pv_shade : numeric
        View factor of ground from shaded part of row.
    f_gnd_pv_noshade : numeric
        View factor of ground from unshaded part of row.

    Notes
    -----
    At the bottom of rack, :math:`x = 0`, the angle is zero, and the view
    factor to the ground is one.

    .. math::
        \\large{F_{gnd \\rightarrow shade} = \\frac{1 + \\frac{1 - \\cos
        \\left(\\beta - \\psi_b \\right)}{1 - \\cos \\beta}}{2}}

    Take the average of the shaded and unshaded sections.

    .. math::
        \\large{F_{gnd \\rightarrow no\\ shade} = \\frac{1 - \\frac{\\cos
        \\left(\\beta - \\psi_b \\right) + \\cos \\left(\\beta - \\psi_b
        \\left(x=1\\right) \\right)}{2}}{1 - \\cos \\beta}}
    """
    # TODO: don't average, return fgnd-pv vs. x point on panel
    f_gnd_pv_shade = 0.5 * (1 + (1 - cosd(surface_tilt - psi_shade))
                            / (1 - cosd(surface_tilt)))
    f_gnd_pv_noshade = (
        (1 - (cosd(surface_tilt - psi_shade)
              + cosd(surface_tilt - psi_bottom))/2)
        / (1 - cosd(surface_tilt)))
    return f_gnd_pv_shade, f_gnd_pv_noshade


def _poa_ground_pv(poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade):
    """
    Ground diffuse POA from average view factor weighted by shaded and unshaded
    parts of the surface.

    Parameters
    ----------
    poa_gnd_sky : numeric
        diffuse ground POA accounting for ground shade but not adjacent rows
    f_x : numeric
        shade line fraction from bottom of module
    f_gnd_pv_shade : numeric
        fraction of ground visible from shaded part of PV surface
    f_gnd_pv_noshade : numeric
        fraction of ground visible from unshaded part of PV surface

    """
    return poa_gnd_sky * (f_x*f_gnd_pv_shade + (1 - f_x)*f_gnd_pv_noshade)


def _poa_diffuse_pv(poa_gnd_pv, poa_sky_pv):
    """diffuse incident on PV surface from sky and ground"""
    return poa_gnd_pv + poa_sky_pv


def _poa_direct_pv(poa_direct, iam, f_x):
    """direct incident on PV surface"""
    return poa_direct * iam * (1 - f_x)


def _poa_global_pv(poa_dir_pv, poa_dif_pv):
    """global incident on PV surface"""
    return poa_dir_pv + poa_dif_pv


def poa_global_bifacial(poa_global_front, poa_global_back, bifaciality=0.8,
                        shade_factor=-0.02, transmission_factor=0):
    """total global incident on bifacial PV surfaces"""
    effects = (1 + shade_factor) * (1 + transmission_factor)
    return poa_global_front + poa_global_back * bifaciality * effects


# TODO: rename to pvlib.bifacial.infinite_sheds?
# TODO: rework output to basics + optional
# TODO: not tested
def get_irradiance(solar_zenith, solar_azimuth, system_azimuth, gcr, height,
                   tilt, pitch, ghi, dhi, poa_ground, poa_sky_diffuse,
                   poa_direct, iam, npoints=100, all_output=False):
    r"""
    Get irradiance from infinite sheds model.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]

    Returns
    -------

    """
    # calculate solar projection
    tan_phi = utils.solar_projection_tangent(
        solar_zenith, solar_azimuth, system_azimuth)
    # fraction of ground illuminated accounting from shade from panels
    f_gnd_beam = utils.unshaded_ground_fraction(gcr, tilt, system_azimuth,
                                                solar_zenith, solar_azimuth)
    # diffuse fraction
    df = _diffuse_fraction(ghi, dhi)
    # TODO: move to bifacial.util
    # view factor from the ground in between infinite central rows to the sky
    vf_gnd_sky, _, _ = _vf_ground_sky(gcr, height, tilt, pitch, npoints)
    # diffuse from sky reflected from ground accounting from shade from panels
    # considering the fraction of ground blocked by infinite adjacent rows
    poa_gnd_sky = _poa_ground_sky(poa_ground, f_gnd_beam, df, vf_gnd_sky)
    # fraction of panel shaded
    f_x = shaded_fraction(solar_zenith, solar_azimuth, tilt, system_azimuth,
                          gcr)
    # angles from shadeline to top of next row
    _, tan_psi_top = _sky_angle(gcr, tilt, f_x)
    # angles from tops of next row to bottom of current row
    _, tan_psi_top_0 = _sky_angle(gcr, tilt, 0.0)
    # fraction of sky visible from shaded and unshaded parts of PV surfaces
    f_sky_pv_shade, f_sky_pv_noshade = _f_sky_diffuse_pv(
        tilt, tan_psi_top, tan_psi_top_0)
    # total sky diffuse incident on plane of array
    poa_sky_pv = _poa_sky_diffuse_pv(
        poa_sky_diffuse, f_x, f_sky_pv_shade, f_sky_pv_noshade)
    # angles from shadeline to bottom of next row
    psi_shade, _ = _ground_angle(gcr, tilt, f_x)
    # angles from top of row to bottom of facing row
    psi_bottom, _ = _ground_angle(gcr, tilt, 1.0)
    f_gnd_pv_shade, f_gnd_pv_noshade = _f_ground_pv(
        tilt, psi_shade, psi_bottom)
    poa_gnd_pv = _poa_ground_pv(
        poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade)
    poa_dif_pv = _poa_diffuse_pv(poa_gnd_pv, poa_sky_pv)
    poa_dir_pv = _poa_direct_pv(poa_direct, iam, f_x)
    poa_glo_pv = _poa_global_pv(poa_dir_pv, poa_dif_pv)
    output = OrderedDict(
        poa_global_pv=poa_glo_pv, poa_direct_pv=poa_dir_pv,
        poa_diffuse_pv=poa_dif_pv, poa_ground_diffuse_pv=poa_gnd_pv,
        poa_sky_diffuse_pv=poa_sky_pv)
    if all_output:
        output.update(
            solar_projection=tan_phi, ground_illumination=f_gnd_beam,
            diffuse_fraction=df, poa_ground_sky=poa_gnd_sky, shade_line=f_x,
            sky_angle_tangent=tan_psi_top, sky_angle_0_tangent=tan_psi_top_0,
            f_sky_diffuse_pv_shade=f_sky_pv_shade,
            f_sky_diffuse_pv_noshade=f_sky_pv_noshade,
            ground_angle=psi_shade,
            ground_angle_bottom=psi_bottom,
            f_ground_diffuse_pv_shade=f_gnd_pv_shade,
            f_ground_diffuse_pv_noshade=f_gnd_pv_noshade)
    if isinstance(poa_glo_pv, pd.Series):
        output = pd.DataFrame(output)
    return output


def get_poa_global_bifacial(solar_zenith, solar_azimuth, system_azimuth, gcr,
                            height, tilt, pitch, ghi, dhi, dni, dni_extra,
                            am_rel, iam_b0_front=0.05, iam_b0_back=0.05,
                            bifaciality=0.8, shade_factor=-0.02,
                            transmission_factor=0, method='haydavies'):
    """Get global bifacial irradiance from infinite sheds model."""
    # backside is rotated and flipped relative to front
    backside_tilt, backside_sysaz = _backside(tilt, system_azimuth)
    # AOI
    aoi_front = irradiance.aoi(
        tilt, system_azimuth, solar_zenith, solar_azimuth)
    aoi_back = irradiance.aoi(
        backside_tilt, backside_sysaz, solar_zenith, solar_azimuth)
    # transposition
    irrad_front = irradiance.get_total_irradiance(
        tilt, system_azimuth, solar_zenith, solar_azimuth,
        dni, ghi, dhi, dni_extra, am_rel, model=method)
    irrad_back = irradiance.get_total_irradiance(
        backside_tilt, backside_sysaz, solar_zenith, solar_azimuth,
        dni, ghi, dhi, dni_extra, am_rel, model=method)
    # iam
    iam_front = iam.ashrae(aoi_front, iam_b0_front)
    iam_back = iam.ashrae(aoi_back, iam_b0_back)
    # get front side
    poa_global_front = get_irradiance(
        solar_zenith, solar_azimuth, system_azimuth, gcr, height, tilt, pitch,
        ghi, dhi, irrad_front['poa_ground_diffuse'],
        irrad_front['poa_sky_diffuse'], irrad_front['poa_direct'], iam_front)
    # get backside
    poa_global_back = get_irradiance(
        solar_zenith, solar_azimuth, backside_sysaz, gcr, height,
        backside_tilt, pitch, ghi, dhi, irrad_back['poa_ground_diffuse'],
        irrad_back['poa_sky_diffuse'], irrad_back['poa_direct'], iam_back)
    # get bifacial
    poa_glo_bifi = poa_global_bifacial(
        poa_global_front['poa_global_pv'], poa_global_back['poa_global_pv'],
        bifaciality, shade_factor, transmission_factor)
    return poa_glo_bifi


def _backside(tilt, system_azimuth):
    backside_tilt = np.pi - tilt
    backside_sysaz = (np.pi + system_azimuth) % (2*np.pi)
    return backside_tilt, backside_sysaz


class InfiniteSheds(object):
    """An infinite sheds model"""
    def __init__(self, system_azimuth, gcr, height, tilt, pitch, npoints=100,
                 is_bifacial=True, bifaciality=0.8, shade_factor=-0.02,
                 transmission_factor=0):
        self.system_azimuth = system_azimuth
        self.gcr = gcr
        self.height = height
        self.tilt = tilt
        self.pitch = pitch
        self.npoints = npoints
        self.is_bifacial = is_bifacial
        self.bifaciality = bifaciality if is_bifacial else 0.0
        self.shade_factor = shade_factor
        self.transmission_factor = transmission_factor
        # backside angles
        self.backside_tilt, self.backside_sysaz = _backside(
            self.tilt, self.system_azimuth)
        # sheds parameters
        self.tan_phi = None
        self.f_gnd_beam = None
        self.df = None
        self.front_side = None
        self.back_side = None
        self.poa_global_bifacial = None

    def get_irradiance(self, solar_zenith, solar_azimuth, ghi, dhi, poa_ground,
                       poa_sky_diffuse, poa_direct, iam):
        self.front_side = _PVSurface(*get_irradiance(
            solar_zenith, solar_azimuth, self.system_azimuth,
            self.gcr, self.height, self.tilt, self.pitch, ghi, dhi, poa_ground,
            poa_sky_diffuse, poa_direct, iam, npoints=self.npoints,
            all_output=True))
        self.tan_phi = self.front_side.tan_phi
        self.f_gnd_beam = self.front_side.f_gnd_beam
        self.df = self.front_side.df
        if self.bifaciality > 0:
            self.back_side = _PVSurface(*get_irradiance(
                solar_zenith, solar_azimuth, self.backside_sysaz,
                self.gcr, self.height, self.backside_tilt, self.pitch, ghi,
                dhi, poa_ground, poa_sky_diffuse, poa_direct, iam,
                self.npoints, all_output=True))
            self.poa_global_bifacial = poa_global_bifacial(
                self.front_side.poa_global_pv, self.back_side.poa_global_pv,
                self.bifaciality, self.shade_factor, self.transmission_factor)
            return self.poa_global_bifacial
        else:
            return self.front_side.poa_global_pv


class _PVSurface(object):
    """A PV surface in an infinite shed"""
    def __init__(self, poa_glo_pv, poa_dir_pv, poa_dif_pv, poa_gnd_pv,
                 poa_sky_pv, tan_phi, f_gnd_beam, df, poa_gnd_sky, f_x,
                 tan_psi_top, tan_psi_top_0, f_sky_pv_shade, f_sky_pv_noshade,
                 tan_psi_bottom, tan_psi_bottom_1, f_gnd_pv_shade,
                 f_gnd_pv_noshade):
        self.poa_global_pv = poa_glo_pv
        self.poa_direct_pv = poa_dir_pv
        self.poa_diffuse_pv = poa_dif_pv
        self.poa_ground_pv = poa_gnd_pv
        self.poa_sky_diffuse_pv = poa_sky_pv
        self.tan_phi = tan_phi
        self.f_gnd_beam = f_gnd_beam
        self.df = df
        self.poa_ground_sky = poa_gnd_sky
        self.f_x = f_x
        self.tan_psi_top = tan_psi_top
        self.tan_psi_top_0 = tan_psi_top_0
        self.f_sky_pv_shade = f_sky_pv_shade
        self.f_sky_pv_noshade = f_sky_pv_noshade
        self.tan_psi_bottom = tan_psi_bottom
        self.tan_psi_bottom_1 = tan_psi_bottom_1
        self.f_gnd_pv_shade = f_gnd_pv_shade
        self.f_gnd_pv_noshade = f_gnd_pv_noshade
