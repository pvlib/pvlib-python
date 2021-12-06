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
from pvlib.irradiance import get_ground_diffuse, beam_component


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
        ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    height : numeric
        height of module lower edge above the ground.
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    pitch : numeric
        row spacing.

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

    Parameters `height` and `pitch` must have the same unit.
    
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


# TODO: make sure that it is clear psi_1 is a supplement (angle from negative
# x axis)
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


# TODO: move to util
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
    # TODO: explain this loop over rows in front of current row
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
        Integration of view factor over the length between adjacent, interior
        rows. [unitless]
    fz : ndarray
        Fraction of distance from the previous row to the next row. [unitless]
    fz_sky : ndarray
        View factors at discrete points between adjacent, interior rows.
        [unitless]

    """
    args = gcr, height, surface_tilt, pitch
    # calculate the view factor from the ground to the sky. Accounts for 
    # views between rows both towards the array front, and array back
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
        Global horizontal irradiance (GHI). [W/m^2]
    dhi : numeric
        Diffuse horizontal irradiance (DHI). [W/m^2]

    Returns
    -------
    df : numeric
        diffuse fraction
    """
    return dhi / ghi


def _poa_ground_shadows(poa_ground, f_gnd_beam, df, vf_gnd_sky):
    """
    Reduce ground-reflected irradiance to the tilted plane (poa_ground) to
    account for shadows on the ground.

    Parameters
    ----------
    poa_ground : numeric
        Ground reflected irradiance on the tilted surface, assuming full GHI
        illumination on all of the ground. [W/m^2]
    f_gnd_beam : numeric
        Fraction of the distance between rows that is illuminated (unshaded).
        [unitless]
    df : numeric
        Diffuse fraction, the ratio of DHI to GHI. [unitless]
    vf_gnd_sky : numeric
        View factor from the ground to the sky, integrated along the distance
        between rows. [unitless]

    Returns
    -------
    poa_gnd_sky : numeric
        Adjusted ground-reflected irradiance accounting for shadows on the
        ground. [W/m^2]

    """
    df = np.where(np.isfinite(df), df, 0.0)
    return poa_ground * (f_gnd_beam*(1 - df) + df*vf_gnd_sky)


def _sky_angle(gcr, surface_tilt, x):
    """
    Angle from a point x along the module slant height to the
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


def _f_sky_diffuse_pv(gcr, surface_tilt, f_x, npoints=100):
    """
    Integrated view factors of the sky from the shaded and unshaded parts of
    the row slant height.

    Parameters
    ----------
    gcr : numeric
        Ratio of row slant length to row spacing (pitch). [unitless]
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    f_x : numeric
        Fraction of row slant height from the bottom that is shaded. [unitless]
    npoints : int, default 100
        Number of points for integration. [unitless]

    Returns
    -------
    f_sky_pv_shade : numeric
        Integrated view factor from the shaded part of the row to the sky.
        [unitless]
    f_sky_pv_noshade : numeric
        Integrated view factor from the unshaded part of the row to the sky.
        [unitless]

    Notes
    -----
    The view factor to the sky at a point x along the row slant height is
    given by

    .. math ::
        \\large{f_{sky} = \frac{1}{2} \\left(\\cos\\left(\\psi_t\\right) + 
        \\cos \\left(\\beta\\right) \\right)

    where :math:`\\psi_t` is the angle from horizontal of the line from point
    x to the top of the facing row, and :math:`\\beta` is the surface tilt.

    View factors are integrated separately over shaded and unshaded portions
    of the row slant height.

    See Also
    --------
    _sky_angle

    """
    cst = cosd(surface_tilt)
    # shaded portion
    x = np.linspace(0 * f_x, f_x, num=npoints, axis=0)
    psi_t_shaded = _sky_angle(gcr, surface_tilt, x)
    y = 0.5 * (cosd(psi_t_shaded) + cst)
    # integrate view factors from each point in the discretization. This is an
    # improvement over the algorithm described in [2]
    f_sky_pv_shade = np.trapz(y, x, axis=0)
    # unshaded portion
    x = np.linspace(f_x, np.ones_like(f_x), num=npoints, axis=0)
    psi_t_unshaded = _sky_angle(gcr, surface_tilt, x)
    y = 0.5 * (cosd(psi_t_unshaded) + cst)
    f_sky_pv_noshade = np.trapz(y, x, axis=0)
    return f_sky_pv_shade, f_sky_pv_noshade


def _poa_sky_diffuse_pv(dhi, f_x, f_sky_pv_shade, f_sky_pv_noshade):
    """
    Sky diffuse POA from integrated view factors combined for both shaded and
    unshaded parts of the surface.

    Parameters
    ----------
    dhi : numeric
        Diffuse horizontal irradiance (DHI). [W/m^2]
    f_x : numeric
        Fraction of row slant height from the bottom that is shaded. [unitless]
    f_sky_pv_shade : numeric
        Integrated (average) view factor to the sky from the shaded part of
        the PV surface. [unitless]
    f_sky_pv_noshade : numeric
        Integrated (average) view factor to the sky from the unshaded part of
        the PV surface. [unitless]

    Returns
    -------
    poa_sky_diffuse_pv : numeric
        Total sky diffuse irradiance incident on the PV surface. [W/m^2]
    """
    return dhi * (f_x * f_sky_pv_shade + (1 - f_x) * f_sky_pv_noshade)


def _ground_angle(gcr, surface_tilt, x):
    """
    Angle from horizontal to the line from a point x on the row slant length
    to the bottom of the facing row.
    
    The angles returned are measured clockwise from horizontal, rather than
    the usual counterclockwise direction.

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
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
    x2 = (x * cosd(surface_tilt) + 1 / gcr)
    psi = np.arctan2(x1, x2)  # do this first because it handles 0 / 0
    tan_psi = np.tan(psi)
    return np.rad2deg(psi), tan_psi


def _vf_row_ground(gcr, surface_tilt, x):
    """
    View factor from a point x to the ground between rows.

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    x : numeric
        Fraction of row slant height from the bottom. [unitless]

    Returns
    -------
    vf : numeric
        View factor from the point at x to the ground. [unitless]

    """
    cst = cosd(surface_tilt)
    # angle from each point x on the row slant height to the bottom of the
    # facing row
    psi_t_shaded = _ground_angle(gcr, surface_tilt, x)
    # view factor from the point on the row to the ground
    return 0.5 * (cosd(psi_t_shaded) - cst)


def _f_ground_pv(gcr, surface_tilt, f_x, npoints=100):
    """
    View factors to the ground from shaded and unshaded parts of a row.

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    f_x : numeric
        Fraction of row slant height from the bottom that is shaded. [unitless]
    npoints:

    Returns
    -------
    f_gnd_pv_shade : numeric
        View factor from the row to the ground integrated over the shaded
        portion of the row slant height.
    f_gnd_pv_noshade : numeric
        View factor from the row to the ground integrated over the unshaded
        portion of the row slant height.

    Notes
    -----
    The view factor to the ground at a point x along the row slant height is
    given by

    .. math ::
        \\large{f_{gr} = \frac{1}{2} \\left(\\cos\\left(\\psi_t\\right) -
        \\cos \\left(\\beta\\right) \\right)

    where :math:`\\psi_t` is the angle from horizontal of the line from point
    x to the bottom of the facing row, and :math:`\\beta` is the surface tilt.

    Each view factor is integrated over the relevant portion of the row
    slant height.
    """
    # shaded portion of row slant height
    x = np.linspace(0 * f_x, f_x, num=npoints, axis=0)
    # view factor from the point on the row to the ground
    y = _vf_row_ground(gcr, surface_tilt, x)
    # integrate view factors along the shaded portion of the row slant height.
    # This is an improvement over the algorithm described in [2]
    f_gnd_pv_shade = np.trapz(y, x, axis=0)

    # unshaded portion of row slant height
    x = np.linspace(f_x, np.ones_like(f_x), num=npoints, axis=0)
    # view factor from the point on the row to the ground
    y = _vf_row_ground(gcr, surface_tilt, x)
    # integrate view factors along the unshaded portion.
    # This is an improvement over the algorithm described in [2]
    f_gnd_pv_noshade = np.trapz(y, x, axis=0)

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
        Fraction of row slant height from the bottom that is shaded. [unitless]
    f_gnd_pv_shade : numeric
        fraction of ground visible from shaded part of PV surface
    f_gnd_pv_noshade : numeric
        fraction of ground visible from unshaded part of PV surface

    Returns
    -------
    
    """
    return poa_gnd_sky * (f_x * f_gnd_pv_shade + (1 - f_x) * f_gnd_pv_noshade)


def poa_global_bifacial(poa_global_front, poa_global_back, bifaciality=0.8,
                        shade_factor=-0.02, transmission_factor=0):
    """total global incident on bifacial PV surfaces"""
    effects = (1 + shade_factor) * (1 + transmission_factor)
    return poa_global_front + poa_global_back * bifaciality * effects


# TODO: rename to pvlib.bifacial.infinite_sheds?
# TODO: rework output to basics + optional
# TODO: not tested
def get_poa_irradiance(solar_zenith, solar_azimuth, surface_tilt,
                       surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
                       albedo, iam=1.0, npoints=100, all_output=False):
    r"""
    Get irradiance on one side of an infinite row of modules using the infinite
    sheds model.

    Plane-of-array (POA) irradiance components include direct, diffuse and
    global (total). Irradiance values are not reduced by reflections, adjusted
    for solar spectrum, or reduced by a module's light collecting aperature,
    which is quantified by the module's bifaciality factor.

    Parameters
    ----------
    solar_zenith : array-like
        True (not refraction-corrected) solar zenith angles in decimal
        degrees.

    solar_azimuth : array-like
        Solar azimuth angles in decimal degrees.

    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The Azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]

    gcr : numeric
        Ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]

    height : numeric
        height of module lower edge above the ground.

    pitch : numeric
        row spacing.

    dni : numeric
        Direct normal irradiance. [W/m2]

    ghi : numeric
        Global horizontal irradiance. [W/m2]

    dhi : numeric
        Diffuse horizontal irradiance. [W/m2]
 
    albedo : numeric
        Surface albedo. [unitless]

    iam : numeric, default 1.0
        Incidence angle modifier, the fraction of direct irradiance incident
        on the surface that is not reflected away. [unitless]

    npoints : int, default 100
        Number of points used to discretize distance along the ground.

    all_ouputs : boolean, default False
        If True then detailed output is returned. If False, only plane-of-array
        irradiance components are returned.

    Returns
    -------
    output : OrderedDict or DataFrame
        Output is a DataFrame when input ghi is a Series. See Notes for
        descriptions of content.
        
    Notes
    -----
    Input parameters `height` and `pitch` must have the same unit.

    Output always includes:
  
    - poa_global : total POA irradiance. [W/m^2]
    - poa_diffuse : total diffuse POA irradiance from all sources. [W/m^2]
    - poa_direct : total direct POA irradiance. [W/m^2]

    Optionally, output includes:

    - poa_diffuse_sky : total sky diffuse irradiance on the plane of array.
      [W/m^2]
    - poa_diffuse_ground : total ground-reflected diffuse irradiance on the
      plane of array. [W/m^2]

    """
    # Calculate some geometric quantities
    # fraction of ground between rows that is illuminated accounting for
    # shade from panels
    f_gnd_beam = utils.unshaded_ground_fraction(gcr, surface_tilt, surface_azimuth,
                                                solar_zenith, solar_azimuth)
    # integrated view factor from the ground to the sky, integrated between
    # adjacent rows interior to the array
    vf_gnd_sky, _, _ = _vf_ground_sky(gcr, height, surface_tilt, pitch, npoints)

    # fraction of row slant height that is shaded
    f_x = shaded_fraction(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth,
                          gcr)
    # angle from the shadeline to top of next row
    _, tan_psi_top = _sky_angle(gcr, surface_tilt, f_x)
    # angle from top of next row to bottom of current row
    _, tan_psi_top_0 = _sky_angle(gcr, surface_tilt, 0.0)
    # fractions of the sky visible from the shaded and unshaded parts of the
    # row slant height
    f_sky_pv_shade, f_sky_pv_noshade = _f_sky_diffuse_pv(
        gcr, surface_tilt, f_x)
    # angle from shadeline to bottom of facing row
    psi_shade, _ = _ground_angle(gcr, surface_tilt, f_x)
    # angle from top of row to bottom of facing row
    psi_bottom, _ = _ground_angle(gcr, surface_tilt, 1.0)
    # view factors from the ground to shaded and unshaded portions of the row
    # slant height
    f_gnd_pv_shade, f_gnd_pv_noshade = _f_ground_pv(gcr, surface_tilt, f_x)

    # Calculate some preliminary irradiance quantities
    # diffuse fraction
    df = _diffuse_fraction(ghi, dhi)
    # sky diffuse reflected from the ground to an array consisting of a single
    # row
    poa_ground = get_ground_diffuse(surface_tilt, ghi, albedo)
    poa_beam = beam_component(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                              dni)
    # Total sky diffuse recieved by both shaded and unshaded portions
    poa_sky_pv = _poa_sky_diffuse_pv(
        dhi, f_x, f_sky_pv_shade, f_sky_pv_noshade)

    # Reduce ground-reflected irradiance because other rows in the array
    # block irradiance from reaching the ground.
    # [2], Eq. 9
    poa_gnd_sky = _poa_ground_shadows(poa_ground, f_gnd_beam, df, vf_gnd_sky)

    # Further reduce ground-reflected irradiance because adjacent rows block
    # the view to the ground.
    poa_gnd_pv = _poa_ground_pv(
        poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade)

    # add sky and ground-reflected irradiance on the row by irradiance
    # component
    poa_diffuse = poa_gnd_pv + poa_sky_pv
    poa_direct = poa_beam * (1 - f_x) * iam  # direct only on the unshaded part
    poa_global = poa_direct + poa_diffuse

    output = OrderedDict(
        poa_global=poa_global, poa_direct=poa_direct,
        poa_diffuse=poa_diffuse, poa_ground_diffuse=poa_gnd_pv,
        poa_sky_diffuse=poa_sky_pv)
    if all_output:
        output.update(poa_diffuse_sky=poa_sky_pv,
                      poa_diffuse_ground=poa_gnd_pv)
    if isinstance(ghi, pd.Series):
        output = pd.DataFrame(output)
    return output


def get_irradiance(solar_zenith, solar_azimuth, surface_tilt,
                   surface_azimuth, gcr, height, pitch, ghi, dhi, dni,
                   albedo, dni_extra, iam_front=1.0, iam_back=1.0,
                   bifaciality=0.8, shade_factor=-0.02,
                   transmission_factor=0):
    """
    Get bifacial irradiance using the infinite sheds model.
    
    Parameters
    ----------    
    iam_front : numeric, default 1.0
        Incidence angle modifier, the fraction of direct irradiance incident
        on the front surface that is not reflected away. [unitless]
    
    iam_back : numeric, default 1.0
        Incidence angle modifier, the fraction of direct irradiance incident
        on the back surface that is not reflected away. [unitless]
    
    """
    # backside is rotated and flipped relative to front
    backside_tilt, backside_sysaz = _backside(surface_tilt, surface_azimuth)
    # front side POA irradiance
    irrad_front = get_poa_irradiance(
        solar_zenith, solar_azimuth, surface_tilt, surface_azimuth, gcr,
        height, pitch, ghi, dhi, dni, albedo, iam_front)
    irrad_front.rename(columns={'poa_global': 'poa_front',
                                'poa_diffuse': 'poa_front_diffuse',
                                'poa_direct': 'poa_front_direct'})
    # back side POA irradiance
    irrad_back = get_poa_irradiance(
        solar_zenith, solar_azimuth, backside_tilt, backside_sysaz, gcr,
        height, pitch, ghi, dhi, dni, albedo, iam_back)
    irrad_back.rename(columns={'poa_global': 'poa_back',
                               'poa_diffuse': 'poa_back_diffuse',
                               'poa_direct': 'poa_back_direct'})
    effects = (1 + shade_factor) * (1 + transmission_factor)
    output = pd.concat([irrad_front, irrad_back], axis=1)
    output['poa_global'] = output['poa_front'] + \
        output['poa_back'] * bifaciality * effects
    return output


def _backside(tilt, system_azimuth):
    backside_tilt = np.pi - tilt
    backside_sysaz = (np.pi + system_azimuth) % (2*np.pi)
    return backside_tilt, backside_sysaz
