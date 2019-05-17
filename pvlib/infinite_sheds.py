"""
Modify plane of array irradiance components to account for adjancent rows for
both monofacial and bifacia infinite sheds. Sheds are defined as fixed tilt or
trackers that a fixed GCR on horizontal surface. Future version will also
account for sloped surfaces. The process is divide into 7 steps:
1. transposition
2. ground illumination
3. shade line
4. angle of sky/ground subtended by module
5. view factor of sky/ground from module vs. location of shade line
6. approximate view factors as linear across shade and light regions
7. sum up components

References
----------
[1] Bill Marion, et al. IEEE PVSC 2017
"""

import numpy as np


def _to_radians(*params, is_rad=False):
    if is_rad:
        return params
    p_rad = []
    for p in params:
        p_rad.append(np.radians(p))
    return p_rad


def get_solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth,
                                 is_rad=False):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    .. math::
        \\tan \\phi = \\frac{\\cos\\left(\\text{solar azimuth} -
        \\text{system azimuth}\\right)\\sin\\left(\\text{solar zenith}
        \\right)}{\\cos\\left(\\text{solar zenith}\\right)}

    Parameters
    ----------
    solar_zenith : numeric
        Apparent zenith in degrees.
    solar_azimuth : numeric
        Azimuth in degrees.
    system_azimuth : numeric
        System rotation from north in degrees.
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tanphi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, is_rad)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    tanphi = np.cos(rotation_rad) * np.tan(solar_zenith_rad)
    return tanphi


def get_solar_projection(solar_zenith, solar_azimuth, system_azimuth,
                         is_rad=False):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    Parameters
    ----------
    solar_zenith : numeric
        Apparent zenith in degrees.
    solar_azimuth : numeric
        Azimuth in degrees.
    system_azimuth : numeric
        System rotation from north in degrees.
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    phi : numeric
        4-quadrant arc-tangent of solar projection
    tanphi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, is_rad)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    x1 = np.cos(rotation_rad) * np.sin(solar_zenith_rad)
    x2 = np.cos(solar_zenith_rad)
    tanphi = x1 / x2
    phi = np.arctan2(x1, x2)
    return phi, tanphi


def get_ground_illumination(gcr, tilt, tanphi, is_rad=False):
    """
    Calculate the fraction of the ground visible from the sky.

    .. math::
        F_{gnd,sky} &= 1 - \\min{\\left(1, \\text{GCR} \\left|\\cos \\beta +
            \\sin \\beta \\tan \\phi \\right|\\right)} \\newline
        \\beta &= \\text{tilt}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of module normal from vertical in degrees, if bifacial use front
    tanphi : numeric
        solar projection tangent
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    f_gnd_sky : numeric
        fration of ground illuminated from sky
    """
    tilt_rad = _to_radians(tilt, is_rad)[0]
    f_gnd_sky = 1.0 - np.minimum(
        1.0, gcr * np.abs(np.cos(tilt_rad) + np.sin(tilt_rad) * tanphi))
    return f_gnd_sky  # 1 - min(1, abs()) < 1 always


def get_diffuse_fraction(ghi, dhi):
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


def get_poa_gnd_sky(poa_ground, f_gnd_sky, diffuse_fraction):
    """
    transposed ground reflected diffuse component adjusted for ground
    illumination, but not accounting for adjacent rows

    Parameters
    ----------
    poa_ground : numeric
        transposed ground reflected diffuse component in W/m^2
    f_gnd_sky : numeric
        ground illumination
    diffuse_fraction : numeric
        ratio of DHI to GHI

    Returns
    -------
    poa_gnd_sky : numeric
        adjusted irradiance on modules reflected from ground
    """
    return poa_ground * (f_gnd_sky * (1 - diffuse_fraction) + diffuse_fraction)


def get_shade_line(gcr, tilt, tanphi, is_rad=False):
    """
    calculate fraction of module shaded from the bottom

    .. math::
        F_x = \\max \\left( 0, \\min \\left(1 - \\frac{1}{\\text{GCR} \\left(
            \\cos \\beta + \\sin \\beta \\tan \\phi \\right)}, 1 \\right)
            \\right)

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    tanphi : numeric
        solar projection tangent
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    Fx : numeric
        fraction of module shaded from the bottom
    """
    tilt_rad = _to_radians(tilt, is_rad)[0]
    Fx = 1.0 - 1.0 / gcr / (np.cos(tilt_rad) + np.sin(tilt_rad) * tanphi)
    return np.maximum(0.0, np.minimum(Fx, 1.0))


def get_sky_angle_tangent(gcr, tilt, f_x, is_rad=False):
    """
    tangent of angle from shade line to top of next row

    .. math::

        \\tan{\\psi_t} &= \\frac{F_y \\text{GCR}
            \\sin{\\beta}}{1 - F_y \\text{GCR} \\cos{\\beta}} \\newline

        F_y &= 1 - F_x

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    f_x : numeric
        fraction of module shaded from bottom
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_top : numeric
        tangent of angle from shade line to top of next row
    """
    tilt_rad = _to_radians(tilt, is_rad)[0]
    f_y = 1.0 - f_x
    return f_y * np.sin(tilt_rad) / (1/gcr - f_y * np.cos(tilt_rad))


def get_sky_angle(gcr, tilt, f_x, is_rad=False):
    """
    angle from shade line to top of next row

    .. math::

        \\tan{\\psi_t} &= \\frac{F_y \\text{GCR}
            \\sin{\\beta}}{1 - F_y \\text{GCR} \\cos{\\beta}} \\newline

        F_y &= 1 - F_x

    Parameters
    ----------
    gcr : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    f_x : numeric
        fraction of module shaded from bottom
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    psi_top : numeric
        4-quadrant arc-tangent
    tan_psi_top
        tangent of angle from shade line to top of next row
    """
    tilt_rad = _to_radians(tilt, is_rad)[0]
    f_y = 1.0 - f_x
    x1 = f_y * np.sin(tilt_rad)
    x2 = (1/gcr - f_y * np.cos(tilt_rad))
    tan_psi_top = x1 / x2
    psi_top = np.arctan2(x1, x2)
    return psi_top, tan_psi_top



def get_sky_angle_0_tangent(gcr, tilt, is_rad=False):
    """
    tangent of angle to top of next row with no shade (shade line at bottom)

    .. math::

        \\tan{\\psi_t} = \\frac{\\text{GCR} \\sin{\\beta}}{1 - \\text{GCR}
            \\cos{\\beta}}

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_top_0 : numeric
        tangent angle from bottom, ``x = 0``, to top of next row
    """
    return get_sky_angle_tangent(gcr, tilt, 0.0, is_rad)


def get_f_sky_pv(tilt, sky_angle_tangent, sky_angle_0_tangent, is_rad=False):
    """
    view factors of sky from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in degrees
    sky_angle_tangent : numeric
        tangent of angle from shade line to top of next row
    sky_angle_0_tangent : numeric
        tangent of angle to top of next row with no shade (shade line at
        bottom)
    is_rad : bool, default is False
        if true, then input arguments are radians

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
        \\large{F_{nextrow \\rightarrow shade} = \\frac{ 1 + \\frac{\\cos
            \\left(\\psi_{next\\ row\\ top \\rightarrow shadeline} + \\beta
            \\right) + \\cos \\left(\\psi_{next\\ row\\ top \\rightarrow 0} +
            \\beta \\right)}{2}  }{ 1 + \\cos \\beta}}

    Recall that the view factor from the top of the rack is one because it's
    view is not obstructued.

    .. math::
        \\large{F_{nextrow \\rightarrow no\\ shade} = \\frac{1 + \\frac{1 +
        \\cos \\left(\\psi_{next\\ row\\ top \\rightarrow shadeline} + \\beta
        \\right)}{1 + \\cos \\beta} }{2}}
    """
    tilt_rad = _to_radians(tilt, is_rad)[0]
    sky_angle = np.arctan(sky_angle_tangent)
    sky_angle_0 = np.arctan(sky_angle_0_tangent)
    f_sky_pv_shade = (
        (1 + (np.cos(sky_angle + tilt_rad)
              + np.cos(sky_angle_0 + tilt_rad)) / 2) / (1 + np.cos(tilt_rad)))

    f_sky_pv_noshade = (1 + (
        1 + np.cos(sky_angle + tilt_rad)) / (1 + np.cos(tilt_rad))) / 2
    return f_sky_pv_shade, f_sky_pv_noshade


def get_poa_sky_pv(poa_sky_diffuse, shade_line, f_sky_pv_shade,
                   f_sky_pv_noshade):
    """
    Sky diffuse POA from average view factor weighted by shaded and unshaded
    parts of the surface.
    """
    return poa_sky_diffuse * (
        shade_line*f_sky_pv_shade + (1 - shade_line)*f_sky_pv_noshade)


def get_ground_angle(gcr, tilt, shade_line, is_rad=False):
    """
    angle from shadeline to bottom of adjacent row

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    shade_line : numeric
        fraction of module shaded from bottom, ``f_x = 0`` if shade line at
        bottom and no shade, ``f_x = 1`` if shade line at top and all shade
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    psi_bottom : numeric
        4-quadrant arc-tangent
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    """
    tilt_rad = _to_radians(tilt, is_rad)
    x1 = shade_line * np.sin(tilt_rad)
    x2 = (shade_line * np.cos(tilt_rad) + 1/gcr)
    tan_psi_bottom = x1 / x2
    psi_bottom = np.arctan2(x1, x2)
    return psi_bottom, tan_psi_bottom


def get_ground_angle_tangent(gcr, tilt, shade_line, is_rad=False):
    """
    tangent of angle from shadeline to bottom of adjacent row

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    shade_line : numeric
        fraction of module shaded from bottom, ``f_x = 0`` if shade line at
        bottom and no shade, ``f_x = 1`` if shade line at top and all shade
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    """
    tilt_rad = _to_radians(tilt, is_rad)
    return shade_line * np.sin(tilt_rad) / (
        shade_line * np.cos(tilt_rad) + 1/gcr)


def get_ground_angle_1_tangent(gcr, tilt, is_rad=False):
    """
    tangent of angle to bottom of next row with all shade (shade line at top)

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_bottom_1 : numeric
        tangent of angle to bottom of next row with all shade (shade line at
        top)
    """
    return get_ground_angle_tangent(gcr, tilt, 1.0, is_rad)


def get_f_gnd_pv(tilt, ground_angle_tangent, ground_angle_1_tangent,
                 is_rad=False):
    """
    view factors of ground from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in degrees
    ground_angle_tangent : numeric
        tangent of angle from shade line to bottom of next row
    ground_angle_1_tangent : numeric
        tangent of angle to bottom of next row with all shade
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    f_gnd_pv_shade : numeric
        view factor of ground from shaded part of PV surface
    f_gnd_pv_noshade : numeric
        view factor of ground from unshaded part of PV surface

    Notes
    -----
    Take the average of the shaded and unshaded sections.

    .. math::
        \\large{F_{gnd \\rightarrow shade} = \\frac{1 + \\frac{1 - \\cos
        \\left(\\beta - \\psi_{next\\ row\\ bottom \\rightarrow shadeline}
        \\right)}{1 - \\cos \\beta}}{2}}

    At the bottom of rack, ``x = 0``, the angle is zero, and the view factor is
    1.

    .. math::
        \\large{F_{gnd \\rightarrow no\\ shade} = \\frac{1 -
        \\frac{\\cos \\left(\\beta -
        \\psi_{next\\ row\\ bottom \\rightarrow shadeline} \\right) +
        \\cos \\left(\\beta - \\psi_{next\\ row\\ bottom \\rightarrow 1}
        \\right)}{2}}{1 - \\cos \\beta}}
    """
    tilt_rad = _to_radians(tilt, is_rad)
    ground_angle = np.arctan(ground_angle_tangent)
    ground_angle_1 = np.arctan(ground_angle_1_tangent)
    f_gnd_pv_shade = (1 + (1 - np.cos(tilt_rad - ground_angle))
                      / (1 - np.cos(tilt_rad))) / 2
    f_gnd_pv_noshade = (
        (1 - (np.cos(tilt_rad - ground_angle)
            + np.cos(tilt_rad - ground_angle_1))/2)
        / (1 - np.cos(tilt_rad)))
    return f_gnd_pv_shade, f_gnd_pv_noshade


def get_poa_gnd_pv(poa_gnd_sky, shade_line, f_gnd_pv_shade, f_gnd_pv_noshade):
    """
    Ground diffuse POA from average view factor weighted by shaded and unshaded
    parts of the surface.

    Parameters
    ----------
    poa_gnd_sky : numeric
        diffuse ground POA accounting for ground shade but not adjacent rows
    """
    return poa_gnd_sky * (shade_line*f_gnd_pv_shade
                          + (1 - shade_line)*f_gnd_pv_noshade)


def get_poa_diffuse_pv(poa_gnd_pv, poa_sky_pv):
    return poa_gnd_pv + poa_sky_pv


def get_poa_beam_pv(poa_beam, iam, shade_line):
    return poa_beam * iam * (1 - shade_line)


def get_poa_global_pv(poa_beam_pv, poa_diffuse_pv):
    return poa_beam_pv + poa_diffuse_pv


def get_poa_global_bifacial(poa_global_front, poa_global_back, bifaciality=0.8,
                        shade_factor=-0.02, transmission_factor=0):
    effects = (1+shade_factor)*(1+transmission_factor)
    return poa_global_front + poa_global_back * bifaciality * effects


def get_irradiance(solar_zenith, solar_azimuth, system_azimuth, gcr, tilt, ghi,
                   dhi, poa_ground, poa_sky_diffuse, poa_beam, is_rad=False):
    """Get irradiance from infinite sheds model."""
    solar_zenith, solar_azimuth, system_azimuth, tilt = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, tilt, is_rad
    )
    tanphi = get_solar_projection_tangent(
        solar_zenith, solar_azimuth, system_azimuth, is_rad=True)
    f_gnd_sky = get_ground_illumination(gcr, tilt, tanphi, is_rad=True)
    df = get_diffuse_fraction(ghi, dhi)
    poa_gnd_sky = get_poa_gnd_sky(poa_ground, f_gnd_sky, df)
    f_x = get_shade_line(gcr, tilt, tanphi, is_rad=True)
    tan_psi_top = get_sky_angle_tangent(gcr, tilt, f_x, is_rad=True)
    tan_psi_top_0 = get_sky_angle_0_tangent(gcr, tilt, is_rad=True)
    f_sky_pv_shade, f_sky_pv_noshade = get_f_sky_pv(
        tilt, tan_psi_top, tan_psi_top_0, is_rad=True)
    poa_sky_pv = get_poa_sky_pv(
        poa_sky_diffuse, f_x, f_sky_pv_shade, f_sky_pv_noshade)
    return 0


class InfiniteSheds():
    """hi"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
