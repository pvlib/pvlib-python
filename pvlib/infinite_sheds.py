"""
Modify plane of array irradiance components to account for adjancent rows for
both monofacial and bifacia infinite sheds. Sheds are defined as fixed tilt or
trackers that a fixed GCR on horizontal surface. Future version will also
account for sloped surfaces. The process is follows several steps:
1. transposition of diffuse and direct onto front and back surfaces
2. view factor of diffuse sky incident on ground between rows, not blocked
3. fraction of ground unshaded between rows, no direct in shaded fraction
4. integrated view factor of ground reflected and sky diffuse incident on PV
    surface
5. fraction of PV surface shaded, diffuse only
6. sum up components on each surface
7. apply bifaciality factor to backside and combine with front
8. first and last row are different, because they are not blocked on front side
    for 1st row, or backside for last row

References
----------
[1] Bill Marion, et al. IEEE PVSC 2017
"""

from collections import OrderedDict
import numpy as np
import pandas as pd
from pvlib import irradiance, pvsystem

MAXP = 10


def solar_projection(solar_zenith, solar_azimuth, system_azimuth):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    .. math::
        \\tan \\phi = \\frac{\\cos\\left(\\text{solar azimuth} -
        \\text{system azimuth}\\right)\\sin\\left(\\text{solar zenith}
        \\right)}{\\cos\\left(\\text{solar zenith}\\right)}

    Parameters
    ----------
    solar_zenith : numeric
        apparent zenith in radians
    solar_azimuth : numeric
        azimuth in radians
    system_azimuth : numeric
        system rotation from north in radians

    Returns
    -------
    phi : numeric
        4-quadrant arc-tangent of solar projection in radians
    tan_phi : numeric
        tangent of the solar projection
    """
    rotation = solar_azimuth - system_azimuth
    x1 = np.cos(rotation) * np.sin(solar_zenith)
    x2 = np.cos(solar_zenith)
    tan_phi = x1 / x2
    phi = np.arctan2(x1, x2)
    return phi, tan_phi


def solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    .. math::
        \\tan \\phi = \\cos\\left(\\text{solar azimuth}-\\text{system azimuth}
        \\right)\\tan\\left(\\text{solar zenith}\\right)

    Parameters
    ----------
    solar_zenith : numeric
        apparent zenith in radians
    solar_azimuth : numeric
        azimuth in radians
    system_azimuth : numeric
        system rotation from north in radians

    Returns
    -------
    tan_phi : numeric
        tangent of the solar projection
    """
    rotation = solar_azimuth - system_azimuth
    tan_phi = np.cos(rotation) * np.tan(solar_zenith)
    return tan_phi


def unshaded_ground_fraction(gcr, tilt, tan_phi):
    """
    Calculate the fraction of the ground with incident direct irradiance

    .. math::
        F_{gnd,sky} &= 1 - \\min{\\left(1, \\text{GCR} \\left|\\cos \\beta +
        \\sin \\beta \\tan \\phi \\right|\\right)} \\newline

        \\beta &= \\text{tilt}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of module normal from vertical in radians, if bifacial use front
    tan_phi : numeric
        solar projection tangent

    Returns
    -------
    f_gnd_sky : numeric
        fration of ground illuminated from sky
    """
    f_gnd_sky = 1.0 - np.minimum(
        1.0, gcr * np.abs(np.cos(tilt) + np.sin(tilt) * tan_phi))
    return f_gnd_sky  # 1 - min(1, abs()) < 1 always


def _gcr_prime(gcr, height, tilt, pitch):
    """
    A parameter that includes the distance from the module lower edge to the
    point where the module tilt angle intersects the ground in the GCR.

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing

    Returns
    -------
    gcr_prime : numeric
        ground coverage ratio including height above ground
    """

    #  : \\                      \\
    #  :  \\                      \\
    #  :   \\ H = module height    \\
    #  :    \\                      \\
    #  :.....\\......................\\........ module lower edge
    #  :       \                       \    :
    #  :        \                       \   h = height above ground
    #  :         \                 tilt  \  :
    #  +----------\<---------P----------->\---- ground

    return gcr + height / np.sin(tilt) / pitch


def ground_sky_angles(f_z, gcr, height, tilt, pitch):
    """
    Angles from point z on ground to tops of next and previous rows.

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
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    """
    gcr_prime = _gcr_prime(gcr, height, tilt, pitch)
    tilt_prime = np.pi - tilt
    opposite_side = np.sin(tilt_prime)
    adjacent_side = f_z/gcr_prime + np.cos(tilt_prime)
    # tan_psi_0 = opposite_side / adjacent_side
    psi_0 = np.arctan2(opposite_side, adjacent_side)
    f_z_prime = 1 - f_z
    opposite_side = np.sin(tilt)
    adjacent_side = f_z_prime/gcr_prime + np.cos(tilt)
    # tan_psi_1 = opposite_side / adjacent_side
    psi_1 = np.arctan2(opposite_side, adjacent_side)
    return psi_0, psi_1


def ground_sky_angles_prev(f_z, gcr, height, tilt, pitch):
    """
    Angles from point z on ground to top and bottom of previous rows beyond.

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
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    """
    gcr_prime = _gcr_prime(gcr, height, tilt, pitch)
    tilt_prime = np.pi - tilt
    # angle to bottom of panel looking at sky between rows beyond
    psi_0 = np.arctan2(
        np.sin(tilt_prime), (1+f_z)/gcr_prime + np.cos(tilt_prime))
    # angle to front edge of row beyond
    z = f_z*pitch
    # other forms raise division by zero errors
    # avoid division by zero errors
    psi_1 = np.arctan2(height, height/np.tan(tilt) - z)
    return psi_0, psi_1


def f_z0_limit(gcr, height, tilt, pitch):
    """
    Limit from :math:`z_0` where sky is visible between previous rows.

    .. math::
        F_{z0,limit} = \\frac{h}{P} \\left(
        \\frac{1}{\\tan \\beta} + \\frac{1}{\\tan \\psi_t}\\right)

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    """
    tan_psi_t_x0 = sky_angle_0_tangent(gcr, tilt)
    # tan_psi_t_x0 = gcr * np.sin(tilt) / (1.0 - gcr * np.cos(tilt))
    return height/pitch * (1/np.tan(tilt) + 1/tan_psi_t_x0)


def ground_sky_angles_next(f_z, gcr, height, tilt, pitch):
    """
    Angles from point z on the ground to top and bottom of next row beyond.

    .. math::
        \\tan \\psi_0 = \\frac{h}{\\frac{h}{\\tan\\beta^\\prime}
        - \\left(P-z\\right)}

        \\tan{\\psi_1} = \\frac{\\sin{\\beta}}
        {\\frac{F_z^\\prime}{\\text{GCR}^\\prime} + \\cos{\\beta}}
    """
    gcr_prime = _gcr_prime(gcr, height, tilt, pitch)
    tilt_prime = np.pi - tilt
    # angle to bottom of panel looking at sky between rows beyond
    fzprime = 1-f_z
    zprime = fzprime*pitch
    # other forms raise division by zero errors
    # avoid division by zero errors
    psi_0 = np.arctan2(height, height/np.tan(tilt_prime) - zprime)
    # angle to front edge of row beyond
    psi_1 = np.arctan2(np.sin(tilt), (1+fzprime)/gcr_prime + np.cos(tilt))
    return psi_0, psi_1


def f_z1_limit(gcr, height, tilt, pitch):
    """
    Limit from :math:`z_1^\\prime` where sky is visible between next rows.

    .. math::
        F_{z1,limit} = \\frac{h}{P} \\left(
        \\frac{1}{\\tan \\psi_t} - \\frac{1}{\\tan \\beta}\\right)

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    """
    tan_psi_t_x1 = sky_angle_0_tangent(gcr, np.pi-tilt)
    # tan_psi_t_x1 = gcr * np.sin(pi-tilt) / (1.0 - gcr * np.cos(pi-tilt))
    return height/pitch * (1/tan_psi_t_x1 - 1/np.tan(tilt))


def calc_fz_sky(psi_0, psi_1):
    """
    Calculate the view factor for point "z" on the ground to the visible
    diffuse sky subtende by the angles :math:`\\psi_0` and :math:`\\psi_1`.

    Parameters
    ----------
    psi_0 : numeric
        angle from ground to sky before point "z"
    psi_1 : numeric
        angle from ground to sky after point "z"

    Returns
    -------
    fz_sky : numeric
        fraction of energy from the diffuse sky dome that is incident on the
        ground at point "z"
    """
    return (np.cos(psi_0) + np.cos(psi_1))/2


# TODO: add argument to set number of rows, default is infinite
# TODO: add option for first or last row, default is middle row
def ground_sky_diffuse_view_factor(gcr, height, tilt, pitch, npoints=100):
    """
    Calculate the fraction of diffuse irradiance from the sky incident on the
    ground.

    Parameters
    ----------
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    npoints : int
        divide the ground into discrete points
    """
    args = gcr, height, tilt, pitch
    fz0_limit = f_z0_limit(*args)
    fz1_limit = f_z1_limit(*args)
    # include extra space to account for sky visible from adjacent rows
    # divide ground between visible limits into 3x npoints
    fz = np.linspace(
        0.0 if (1-fz1_limit) > 0 else (1-fz1_limit),
        1.0 if fz0_limit < 1 else fz0_limit,
        3*npoints)
    # calculate the angles psi_0 and psi_1 that subtend the sky visible
    # from between rows
    psi_z = ground_sky_angles(fz, *args)
    # front edge
    psi_z0 = ground_sky_angles_prev(fz, *args)
    fz0_sky_next = []
    prev_row = 0.0
    # loop over rows by adding 1.0 to fz until prev_row < ceil(fz0_limit)
    while (fz0_limit - prev_row) > 0:
        fz0_sky_next.append(
            np.interp(fz + prev_row, fz, calc_fz_sky(*psi_z0)))
        prev_row += 1.0
    # back edge
    psi_z1 = ground_sky_angles_next(fz, *args)
    fz1_sky_prev = []
    next_row = 0.0
    # loop over rows by subtracting 1.0 to fz until next_row < ceil(fz1_limit)
    while (fz1_limit - next_row) > 0:
        fz1_sky_prev.append(
            np.interp(fz - next_row, fz, calc_fz_sky(*psi_z1)))
        next_row += 1.0
    # calculate the view factor of the sky from the ground at point z
    fz_sky = (
            calc_fz_sky(*psi_z)  # current row
            + np.sum(fz0_sky_next, axis=0)  # sum of all previous rows
            + np.sum(fz1_sky_prev, axis=0))  # sum of all next rows
    fz_row = np.linspace(0, 1, npoints)
    return fz_row, np.interp(fz_row, fz, fz_sky)


def _big_z(psi_x_bottom, height, tilt, pitch):
    # how far on the ground can the point x see?
    return pitch + height/np.tan(psi_x_bottom) - height / np.tan(tilt)


def calc_fgndpv_zsky(x, gcr, height, tilt, pitch, npoints=100, maxp=MAXP):
    """
    Calculate the fraction of diffuse irradiance from the sky, reflecting from
    the ground, incident at a point on the PV surface.

    Parameters
    ----------
    x : numeric
        point on PV surface
    gcr : numeric
        ground coverage ratio
    height : numeric
        height of module lower edge above the ground
    tilt : numeric
        module tilt in radians, between 0 and 180-degrees
    pitch : numeric
        row spacing
    npoints : int
        divide the ground into discrete points
    maxp : int
        maximum number of adjacent rows to consider, default is 10 rows
    """
    args = gcr, height, tilt, pitch
    hmod = gcr * pitch  # height of PV surface
    fx = x / hmod  # = x/gcr/pitch

    # calculate the view factor of the diffuse sky from the ground between rows
    z_star, fz_sky = ground_sky_diffuse_view_factor(*args, npoints=npoints)

    # calculate the integrated view factor for all of the ground between rows
    fgnd_sky = np.trapz(fz_sky, z_star)

    # if fx is zero, point x is at the bottom of the row, psi_x_bottom is zero,
    # bigz = Inf, and all of the ground is visible, so the view factor is just
    # Fgnd_pv = (1 - cos(beta)) / 2
    if fx == 0:
        fgnd_pv = (1 - np.cos(tilt)) / 2
        return fgnd_sky*fgnd_pv, fgnd_pv

    # how far on the ground can the point x see?
    psi_x_bottom, _ = ground_angle(gcr, tilt, fx)
    bigz = _big_z(psi_x_bottom, height, tilt, pitch)

    # scale bigz by pitch, and limit to maximum number of rows after which the
    # angle from x to the ground will cover more than a row, so we can just use
    # the integral fgnd_sky
    bigz_star = np.maximum(0, np.minimum(bigz/pitch, maxp))
    num_rows = np.ceil(bigz_star).astype(int)

    # repeat fz_sky num_row times, so we can use it to interpolate the values
    # to use in the integral with the angle from x on the PV surface to the
    # ground, note the first & last fz_sky are the same, so skip the last value
    fz_sky = np.tile(fz_sky[:-1], num_rows)
    # append the last value so we can interpolate over the entire row
    fz_sky = np.append(fz_sky, fz_sky[0])

    # increment the ground input segments that are delta wide
    delta = 1.0 / (npoints - 1)
    # start at the next row, z=1, and go back toward the previous row, z<1
    z_star = 1 - np.linspace(num_rows-1, 1, 1 + num_rows * (npoints - 1))
    # shift by a half delta
    halfdelta = delta / 2.0

    # max angle
    psi_ref = tilt - psi_x_bottom
    # use uniform angles
    psidelta = psi_ref * halfdelta
    psi_zuni = np.linspace(psidelta, psi_ref - psidelta, npoints - 1)
    zuni = np.flip(1-_big_z(tilt-psi_zuni, height, tilt, pitch)/pitch)

    fzuni_sky = np.interp(zuni, z_star, fz_sky, fgnd_sky, fgnd_sky)
    dfgnd_pv = (np.cos(psi_zuni-psidelta) - np.cos(psi_zuni+psidelta)) / 2
    fgnd_pv = np.sum(dfgnd_pv)
    # FIXME: not working for backside
    # fskyz = np.sum(fzuni_sky * dfgnd_pv)
    fskyz = fgnd_sky * fgnd_pv
    return fskyz, fgnd_pv


def diffuse_fraction(ghi, dhi):
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
    return dhi/ghi


def poa_ground_sky(poa_ground, f_gnd_sky, df):
    """
    transposed ground reflected diffuse component adjusted for ground
    illumination, but not accounting for adjacent rows

    Parameters
    ----------
    poa_ground : numeric
        transposed ground reflected diffuse component in W/m^2
    f_gnd_sky : numeric
        ground illumination
    df : numeric
        ratio of DHI to GHI

    Returns
    -------
    poa_gnd_sky : numeric
        adjusted irradiance on modules reflected from ground
    """
    # FIXME: DHI is reduced by obstruction of panels too
    return poa_ground * np.where(np.isnan(df), 0.0, f_gnd_sky * (1 - df) + df)


def shade_line(gcr, tilt, tan_phi):
    """
    calculate fraction of module shaded from the bottom

    .. math::
        F_x = \\max \\left( 0, \\min \\left(1 - \\frac{1}{\\text{GCR} \\left(
        \\cos \\beta + \\sin \\beta \\tan \\phi \\right)}, 1 \\right) \\right)

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in radians
    tan_phi : numeric
        solar projection tangent

    Returns
    -------
    f_x : numeric
        fraction of module shaded from the bottom
    """
    f_x = 1.0 - 1.0 / gcr / (np.cos(tilt) + np.sin(tilt) * tan_phi)
    return np.maximum(0.0, np.minimum(f_x, 1.0))


def sky_angle(gcr, tilt, f_x):
    """
    angle from shade line to top of next row

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in radians
    f_x : numeric
        fraction of module shaded from bottom

    Returns
    -------
    psi_top : numeric
        4-quadrant arc-tangent in radians
    tan_psi_top
        tangent of angle from shade line to top of next row
    """
    f_y = 1.0 - f_x
    x1 = f_y * np.sin(tilt)
    x2 = (1/gcr - f_y * np.cos(tilt))
    tan_psi_top = x1 / x2
    psi_top = np.arctan2(x1, x2)
    return psi_top, tan_psi_top


def sky_angle_tangent(gcr, tilt, f_x):
    """
    tangent of angle from shade line to top of next row

    .. math::

        \\tan{\\psi_t} &= \\frac{F_y \\text{GCR} \\sin{\\beta}}{1 - F_y
        \\text{GCR} \\cos{\\beta}} \\newline

        F_y &= 1 - F_x

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in radians
    f_x : numeric
        fraction of module shaded from bottom

    Returns
    -------
    tan_psi_top : numeric
        tangent of angle from shade line to top of next row
    """
    f_y = 1.0 - f_x
    return f_y * np.sin(tilt) / (1/gcr - f_y * np.cos(tilt))


def sky_angle_0_tangent(gcr, tilt):
    """
    tangent of angle to top of next row with no shade (shade line at bottom) so
    :math:`F_x = 0`

    .. math::

        \\tan{\\psi_t\\left(x=0\\right)} = \\frac{\\text{GCR} \\sin{\\beta}}
        {1 - \\text{GCR} \\cos{\\beta}}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in radians

    Returns
    -------
    tan_psi_top_0 : numeric
        tangent angle from bottom, ``x = 0``, to top of next row
    """
    # f_y = 1  b/c x = 0, so f_x = 0
    # tan psi_t0 = GCR * sin(tilt) / (1 - GCR * cos(tilt))
    return sky_angle_tangent(gcr, tilt, 0.0)


def f_sky_diffuse_pv(tilt, tan_psi_top, tan_psi_top_0):
    """
    view factors of sky from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in radians
    tan_psi_top : numeric
        tangent of angle from shade line to top of next row
    tan_psi_top_0 : numeric
        tangent of angle to top of next row with no shade (shade line at
        bottom)

    Returns
    -------
    f_sky_pv_shade : numeric
        view factor of sky from shaded part of PV surface
    f_sky_pv_noshade : numeric
        view factor of sky from unshaded part of PV surface

    Notes
    -----
    Assuming the view factor various roughly linearly from the top to the
    bottom of the rack, we can take the average to get integrated view factor.
    We'll average the shaded and unshaded regions separately to improve the
    approximation slightly.

    .. math ::
        \\large{F_{sky \\rightarrow shade} = \\frac{ 1 + \\frac{\\cos
        \\left(\\psi_t + \\beta \\right) + \\cos \\left(\\psi_t
        \\left(x=0\\right) + \\beta \\right)}{2}  }{ 1 + \\cos \\beta}}

    The view factor from the top of the rack is one because it's view is not
    obstructed.

    .. math::
        \\large{F_{sky \\rightarrow no\\ shade} = \\frac{1 + \\frac{1 +
        \\cos \\left(\\psi_t + \\beta \\right)}{1 + \\cos \\beta} }{2}}
    """
    # TODO: don't average, return fsky-pv vs. x point on panel
    psi_top = np.arctan(tan_psi_top)
    psi_top_0 = np.arctan(tan_psi_top_0)
    f_sky_pv_shade = (
        (1 + (np.cos(psi_top + tilt)
              + np.cos(psi_top_0 + tilt)) / 2) / (1 + np.cos(tilt)))

    f_sky_pv_noshade = (1 + (
        1 + np.cos(psi_top + tilt)) / (1 + np.cos(tilt))) / 2
    return f_sky_pv_shade, f_sky_pv_noshade


def poa_sky_diffuse_pv(poa_sky_diffuse, f_x, f_sky_pv_shade, f_sky_pv_noshade):
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


def ground_angle(gcr, tilt, f_x):
    """
    angle from shadeline to bottom of adjacent row

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in radians
    f_x : numeric
        fraction of module shaded from bottom, ``f_x = 0`` if shade line at
        bottom and no shade, ``f_x = 1`` if shade line at top and all shade

    Returns
    -------
    psi_bottom : numeric
        4-quadrant arc-tangent
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    """
    x1 = f_x * np.sin(tilt)
    x2 = (f_x * np.cos(tilt) + 1/gcr)
    tan_psi_bottom = x1 / x2
    psi_bottom = np.arctan2(x1, x2)
    return psi_bottom, tan_psi_bottom


def ground_angle_tangent(gcr, tilt, f_x):
    """
    tangent of angle from shadeline to bottom of adjacent row

    .. math::
        \\tan{\\psi_b} = \\frac{F_x \\sin \\beta}{F_x \\cos \\beta +
        \\frac{1}{\\text{GCR}}}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in radians
    f_x : numeric
        fraction of module shaded from bottom, ``f_x = 0`` if shade line at
        bottom and no shade, ``f_x = 1`` if shade line at top and all shade

    Returns
    -------
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    """
    return f_x * np.sin(tilt) / (
        f_x * np.cos(tilt) + 1/gcr)


def ground_angle_1_tangent(gcr, tilt):
    """
    tangent of angle to bottom of next row with all shade (shade line at top)
    so :math:`F_x = 1`

    .. math::
        \\tan{\\psi_b\\left(x=1\\right)} = \\frac{\\sin{\\beta}}{\\cos{\\beta}
        + \\frac{1}{\\text{GCR}}}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in radians

    Returns
    -------
    tan_psi_bottom_1 : numeric
        tangent of angle to bottom of next row with all shade (shade line at
        top)
    """
    return ground_angle_tangent(gcr, tilt, 1.0)


def f_ground_pv(tilt, tan_psi_bottom, tan_psi_bottom_1):
    """
    view factors of ground from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in radians
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    tan_psi_bottom_1 : numeric
        tangent of angle to bottom of next row with all shade

    Returns
    -------
    f_gnd_pv_shade : numeric
        view factor of ground from shaded part of PV surface
    f_gnd_pv_noshade : numeric
        view factor of ground from unshaded part of PV surface

    Notes
    -----
    At the bottom of rack, :math:`x = 0`, the angle is zero, and the view
    factor is one.

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
    psi_bottom = np.arctan(tan_psi_bottom)
    psi_bottom_1 = np.arctan(tan_psi_bottom_1)
    f_gnd_pv_shade = (1 + (1 - np.cos(tilt - psi_bottom))
                      / (1 - np.cos(tilt))) / 2
    f_gnd_pv_noshade = (
        (1 - (np.cos(tilt - psi_bottom) + np.cos(tilt - psi_bottom_1))/2)
        / (1 - np.cos(tilt)))
    return f_gnd_pv_shade, f_gnd_pv_noshade


def poa_ground_pv(poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade):
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


def poa_diffuse_pv(poa_gnd_pv, poa_sky_pv):
    """diffuse incident on PV surface from sky and ground"""
    return poa_gnd_pv + poa_sky_pv


def poa_direct_pv(poa_direct, iam, f_x):
    """direct incident on PV surface"""
    return poa_direct * iam * (1 - f_x)


def poa_global_pv(poa_dir_pv, poa_dif_pv):
    """global incident on PV surface"""
    return poa_dir_pv + poa_dif_pv


def poa_global_bifacial(poa_global_front, poa_global_back, bifaciality=0.8,
                        shade_factor=-0.02, transmission_factor=0):
    """total global incident on bifacial PV surfaces"""
    effects = (1+shade_factor)*(1+transmission_factor)
    return poa_global_front + poa_global_back * bifaciality * effects


def get_irradiance(solar_zenith, solar_azimuth, system_azimuth, gcr, tilt, ghi,
                   dhi, poa_ground, poa_sky_diffuse, poa_direct, iam,
                   all_output=False):
    """Get irradiance from infinite sheds model."""
    # calculate solar projection
    tan_phi = solar_projection_tangent(
        solar_zenith, solar_azimuth, system_azimuth)
    # fraction of ground illuminated accounting from shade from panels
    f_gnd_sky = unshaded_ground_fraction(gcr, tilt, tan_phi)
    # diffuse fraction
    df = diffuse_fraction(ghi, dhi)
    # diffuse from sky reflected from ground accounting from shade from panels
    # but not considering the fraction of ground blocked by the next row
    poa_gnd_sky = poa_ground_sky(poa_ground, f_gnd_sky, df)
    # fraction of panel shaded
    f_x = shade_line(gcr, tilt, tan_phi)
    # angles from shadeline to top of next row
    tan_psi_top = sky_angle_tangent(gcr, tilt, f_x)
    tan_psi_top_0 = sky_angle_0_tangent(gcr, tilt)
    # fraction of sky visible from shaded and unshaded parts of PV surfaces
    f_sky_pv_shade, f_sky_pv_noshade = f_sky_diffuse_pv(
        tilt, tan_psi_top, tan_psi_top_0)
    # total sky diffuse incident on plane of array
    poa_sky_pv = poa_sky_diffuse_pv(
        poa_sky_diffuse, f_x, f_sky_pv_shade, f_sky_pv_noshade)
    # angles from shadeline to bottom of next row
    tan_psi_bottom = ground_angle_tangent(gcr, tilt, f_x)
    tan_psi_bottom_1 = ground_angle_1_tangent(gcr, tilt)
    f_gnd_pv_shade, f_gnd_pv_noshade = f_ground_pv(
        tilt, tan_psi_bottom, tan_psi_bottom_1)
    poa_gnd_pv = poa_ground_pv(
        poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade)
    poa_dif_pv = poa_diffuse_pv(poa_gnd_pv, poa_sky_pv)
    poa_dir_pv = poa_direct_pv(poa_direct, iam, f_x)
    poa_glo_pv = poa_global_pv(poa_dir_pv, poa_dif_pv)
    output = OrderedDict(
        poa_global_pv=poa_glo_pv, poa_direct_pv=poa_dir_pv,
        poa_diffuse_pv=poa_dif_pv, poa_ground_diffuse_pv=poa_gnd_pv,
        poa_sky_diffuse_pv=poa_sky_pv)
    if all_output:
        output.update(
            solar_projection=tan_phi, ground_illumination=f_gnd_sky,
            diffuse_fraction=df, poa_ground_sky=poa_gnd_sky, shade_line=f_x,
            sky_angle_tangent=tan_psi_top, sky_angle_0_tangent=tan_psi_top_0,
            f_sky_diffuse_pv_shade=f_sky_pv_shade,
            f_sky_diffuse_pv_noshade=f_sky_pv_noshade,
            ground_angle_tangent=tan_psi_bottom,
            ground_angle_1_tangent=tan_psi_bottom_1,
            f_ground_diffuse_pv_shade=f_gnd_pv_shade,
            f_ground_diffuse_pv_noshade=f_gnd_pv_noshade)
    if isinstance(poa_global_pv, pd.Series):
        output = pd.DataFrame(output)
    return output


def get_poa_global_bifacial(solar_zenith, solar_azimuth, system_azimuth, gcr,
                            tilt, ghi, dhi, dni, dni_extra, am_rel,
                            iam_b0_front=0.05, iam_b0_back=0.05,
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
    iam_front = pvsystem.ashraeiam(aoi_front, iam_b0_front)
    iam_back = pvsystem.ashraeiam(aoi_back, iam_b0_back)
    # get front side
    poa_global_front = get_irradiance(
        solar_zenith, solar_azimuth, system_azimuth, gcr, tilt, ghi, dhi,
        irrad_front.poa_ground_diffuse, irrad_front.poa_sky_diffuse,
        irrad_front.poa_direct, iam_front)
    # get backside
    poa_global_back = get_irradiance(
        solar_zenith, solar_azimuth, backside_sysaz, gcr, backside_tilt, ghi,
        dhi, irrad_back.poa_ground_diffuse, irrad_back.poa_sky_diffuse,
        irrad_back.poa_direct, iam_back)
    # get bifacial
    poa_glo_bifi = poa_global_bifacial(
        poa_global_front[0], poa_global_back[0], bifaciality, shade_factor,
        transmission_factor)
    return poa_glo_bifi


def _backside(tilt, system_azimuth):
    backside_tilt = np.pi - tilt
    backside_sysaz = (np.pi + system_azimuth) % (2*np.pi)
    return backside_tilt, backside_sysaz


class InfiniteSheds(object):
    """An infinite sheds model"""
    def __init__(self, system_azimuth, gcr, tilt, is_bifacial=True,
                 bifaciality=0.8, shade_factor=-0.02, transmission_factor=0):
        self.system_azimuth = system_azimuth
        self.gcr = gcr
        self.tilt = tilt
        self.is_bifacial = is_bifacial
        self.bifaciality = bifaciality
        self.shade_factor = shade_factor
        self.transmission_factor = transmission_factor
        # backside angles
        self.backside_tilt, self.backside_sysaz = _backside(
            self.tilt, self.system_azimuth)
        # sheds parameters
        self.tan_phi = None
        self.f_gnd_sky = None
        self.df = None
        self.front_side = None
        self.back_side = None
        self.poa_global_bifacial = None

    def get_irradiance(self, solar_zenith, solar_azimuth, ghi, dhi, poa_ground,
                       poa_sky_diffuse, poa_direct, iam):
        self.front_side = _PVSurface(*get_irradiance(
            solar_zenith, solar_azimuth, self.system_azimuth,
            self.gcr, self.tilt, ghi, dhi, poa_ground, poa_sky_diffuse,
            poa_direct, iam, all_output=True))
        self.tan_phi = self.front_side.tan_phi
        self.f_gnd_sky = self.front_side.f_gnd_sky
        self.df = self.front_side.df
        if self.is_bifacial and self.bifaciality > 0:
            self.back_side = _PVSurface(*get_irradiance(
                solar_zenith, solar_azimuth, self.backside_sysaz,
                self.gcr, self.backside_tilt, ghi, dhi, poa_ground,
                poa_sky_diffuse, poa_direct, iam,
                all_output=True))
            self.poa_global_bifacial = poa_global_bifacial(
                self.front_side.poa_global_pv, self.back_side.poa_global_pv,
                self.bifaciality, self.shade_factor, self.transmission_factor)


class _PVSurface(object):
    """A PV surface in an infinite shed"""
    def __init__(self, poa_glo_pv, poa_dir_pv, poa_dif_pv, poa_gnd_pv,
                 poa_sky_pv, tan_phi, f_gnd_sky, df, poa_gnd_sky, f_x,
                 tan_psi_top, tan_psi_top_0, f_sky_pv_shade, f_sky_pv_noshade,
                 tan_psi_bottom, tan_psi_bottom_1, f_gnd_pv_shade,
                 f_gnd_pv_noshade):
        self.poa_global_pv = poa_glo_pv
        self.poa_direct_pv = poa_dir_pv
        self.poa_diffuse_pv = poa_dif_pv
        self.poa_ground_pv = poa_gnd_pv
        self.poa_sky_diffuse_pv = poa_sky_pv
        self.tan_phi = tan_phi
        self.f_gnd_sky = f_gnd_sky
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
