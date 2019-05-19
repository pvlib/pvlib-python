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
from collections import OrderedDict


def _to_radians(*params, is_rad=False):
    if is_rad:
        return params
    p_rad = []
    for p in params:
        p_rad.append(np.radians(p))
    return p_rad


def solar_projection(solar_zenith, solar_azimuth, system_azimuth,
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
    tan_phi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, is_rad=is_rad)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    x1 = np.cos(rotation_rad) * np.sin(solar_zenith_rad)
    x2 = np.cos(solar_zenith_rad)
    tan_phi = x1 / x2
    phi = np.arctan2(x1, x2)
    return phi, tan_phi


def solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth,
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
    tan_phi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad, solar_azimuth_rad, system_azimuth_rad = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, is_rad=is_rad)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    tan_phi = np.cos(rotation_rad) * np.tan(solar_zenith_rad)
    return tan_phi


def ground_illumination(gcr, tilt, tan_phi, is_rad=False):
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
    tan_phi : numeric
        solar projection tangent
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    f_gnd_sky : numeric
        fration of ground illuminated from sky
    """
    tilt_rad = _to_radians(tilt, is_rad=is_rad)[0]
    f_gnd_sky = 1.0 - np.minimum(
        1.0, gcr * np.abs(np.cos(tilt_rad) + np.sin(tilt_rad) * tan_phi))
    return f_gnd_sky  # 1 - min(1, abs()) < 1 always


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
    return poa_ground * (f_gnd_sky * (1 - df) + df)


def shade_line(gcr, tilt, tan_phi, is_rad=False):
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
        angle of surface normal from vertical in degrees
    tan_phi : numeric
        solar projection tangent
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    f_x : numeric
        fraction of module shaded from the bottom
    """
    tilt_rad = _to_radians(tilt, is_rad=is_rad)[0]
    f_x = 1.0 - 1.0 / gcr / (np.cos(tilt_rad) + np.sin(tilt_rad) * tan_phi)
    return np.maximum(0.0, np.minimum(f_x, 1.0))


def sky_angle(gcr, tilt, f_x, is_rad=False):
    """
    angle from shade line to top of next row

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
    tilt_rad = _to_radians(tilt, is_rad=is_rad)[0]
    f_y = 1.0 - f_x
    x1 = f_y * np.sin(tilt_rad)
    x2 = (1/gcr - f_y * np.cos(tilt_rad))
    tan_psi_top = x1 / x2
    psi_top = np.arctan2(x1, x2)
    return psi_top, tan_psi_top


def sky_angle_tangent(gcr, tilt, f_x, is_rad=False):
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
    tilt_rad = _to_radians(tilt, is_rad=is_rad)[0]
    f_y = 1.0 - f_x
    return f_y * np.sin(tilt_rad) / (1/gcr - f_y * np.cos(tilt_rad))


def sky_angle_0_tangent(gcr, tilt, is_rad=False):
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
        angle of surface normal from vertical in degrees
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_top_0 : numeric
        tangent angle from bottom, ``x = 0``, to top of next row
    """
    return sky_angle_tangent(gcr, tilt, 0.0, is_rad)


def f_sky_diffuse_pv(tilt, tan_psi_top, tan_psi_top_0, is_rad=False):
    """
    view factors of sky from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in degrees
    tan_psi_top : numeric
        tangent of angle from shade line to top of next row
    tan_psi_top_0 : numeric
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
        \\large{F_{sky \\rightarrow shade} = \\frac{ 1 + \\frac{\\cos
        \\left(\\psi_t + \\beta \\right) + \\cos \\left(\\psi_t
        \\left(x=0\\right) + \\beta \\right)}{2}  }{ 1 + \\cos \\beta}}

    The view factor from the top of the rack is one because it's view is not
    obstructued.

    .. math::
        \\large{F_{sky \\rightarrow no\\ shade} = \\frac{1 + \\frac{1 +
        \\cos \\left(\\psi_t + \\beta \\right)}{1 + \\cos \\beta} }{2}}
    """
    tilt_rad = _to_radians(tilt, is_rad=is_rad)[0]
    psi_top = np.arctan(tan_psi_top)
    psi_top_0 = np.arctan(tan_psi_top_0)
    f_sky_pv_shade = (
        (1 + (np.cos(psi_top + tilt_rad)
              + np.cos(psi_top_0 + tilt_rad)) / 2) / (1 + np.cos(tilt_rad)))

    f_sky_pv_noshade = (1 + (
        1 + np.cos(psi_top + tilt_rad)) / (1 + np.cos(tilt_rad))) / 2
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


def ground_angle(gcr, tilt, f_x, is_rad=False):
    """
    angle from shadeline to bottom of adjacent row

    Parameters
    ----------
    gcr : numeric
        ratio of module length to row spacing
    tilt : numeric
        angle of surface normal from vertical in degrees
    f_x : numeric
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
    tilt_rad = _to_radians(tilt, is_rad=is_rad)
    x1 = f_x * np.sin(tilt_rad)
    x2 = (f_x * np.cos(tilt_rad) + 1/gcr)
    tan_psi_bottom = x1 / x2
    psi_bottom = np.arctan2(x1, x2)
    return psi_bottom, tan_psi_bottom


def ground_angle_tangent(gcr, tilt, f_x, is_rad=False):
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
        angle of surface normal from vertical in degrees
    f_x : numeric
        fraction of module shaded from bottom, ``f_x = 0`` if shade line at
        bottom and no shade, ``f_x = 1`` if shade line at top and all shade
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    """
    tilt_rad = _to_radians(tilt, is_rad=is_rad)
    return f_x * np.sin(tilt_rad) / (
        f_x * np.cos(tilt_rad) + 1/gcr)


def ground_angle_1_tangent(gcr, tilt, is_rad=False):
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
        angle of surface normal from vertical in degrees
    is_rad : bool, default is False
        if true, then input arguments are radians

    Returns
    -------
    tan_psi_bottom_1 : numeric
        tangent of angle to bottom of next row with all shade (shade line at
        top)
    """
    return ground_angle_tangent(gcr, tilt, 1.0, is_rad)


def f_ground_pv(tilt, tan_psi_bottom, tan_psi_bottom_1, is_rad=False):
    """
    view factors of ground from shaded and unshaded parts of PV module

    Parameters
    ----------
    tilt : numeric
        angle of surface normal from vertical in degrees
    tan_psi_bottom : numeric
        tangent of angle from shade line to bottom of next row
    tan_psi_bottom_1 : numeric
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
        \\left(\\beta - \\psi_b \\right)}{1 - \\cos \\beta}}{2}}

    At the bottom of rack, :math:`x = 0`, the angle is zero, and the view
    factor is one.

    .. math::
        \\large{F_{gnd \\rightarrow no\\ shade} = \\frac{1 - \\frac{\\cos
        \\left(\\beta - \\psi_b \\right) + \\cos \\left(\\beta - \\psi_b
        \\left(x=1\\right) \\right)}{2}}{1 - \\cos \\beta}}
    """
    tilt_rad = _to_radians(tilt, is_rad=is_rad)
    psi_bottom = np.arctan(tan_psi_bottom)
    psi_bottom_1 = np.arctan(tan_psi_bottom_1)
    f_gnd_pv_shade = (1 + (1 - np.cos(tilt_rad - psi_bottom))
                      / (1 - np.cos(tilt_rad))) / 2
    f_gnd_pv_noshade = (
        (1 - (np.cos(tilt_rad - psi_bottom)
            + np.cos(tilt_rad - psi_bottom_1))/2)
        / (1 - np.cos(tilt_rad)))
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
                   is_rad=False, all_output=False):
    """Get irradiance from infinite sheds model."""
    # convert angles to radians from degrees if necessary
    solar_zenith, solar_azimuth, system_azimuth, tilt = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, tilt, is_rad=is_rad)
    # calculate solar projection
    tan_phi = solar_projection_tangent(
        solar_zenith, solar_azimuth, system_azimuth, is_rad=True)
    # fraction of ground illuminated accounting from shade from panels
    f_gnd_sky = ground_illumination(gcr, tilt, tan_phi, is_rad=True)
    # diffuse fraction
    df = diffuse_fraction(ghi, dhi)
    # diffuse from sky reflected from ground accounting from shade from panels
    # but not considering the fraction of ground blocked by the next row
    poa_gnd_sky = poa_ground_sky(poa_ground, f_gnd_sky, df)
    # fraction of panel shaded
    f_x = shade_line(gcr, tilt, tan_phi, is_rad=True)
    # angles from shadeline to top of next row
    tan_psi_top = sky_angle_tangent(gcr, tilt, f_x, is_rad=True)
    tan_psi_top_0 = sky_angle_0_tangent(gcr, tilt, is_rad=True)
    # fraction of sky visible from shaded and unshaded parts of PV surfaces
    f_sky_pv_shade, f_sky_pv_noshade = f_sky_diffuse_pv(
        tilt, tan_psi_top, tan_psi_top_0, is_rad=True)
    # total sky diffuse incident on plane of array
    poa_sky_pv = poa_sky_diffuse_pv(
        poa_sky_diffuse, f_x, f_sky_pv_shade, f_sky_pv_noshade)
    # angles from shadeline to bottom of next row
    tan_psi_bottom = ground_angle_tangent(gcr, tilt, f_x, is_rad=True)
    tan_psi_bottom_1 = ground_angle_1_tangent(gcr, tilt, is_rad=True)
    f_gnd_pv_shade, f_gnd_pv_noshade = f_ground_pv(
        tilt, tan_psi_bottom, tan_psi_bottom_1, is_rad=True)
    poa_gnd_pv = poa_ground_pv(
        poa_gnd_sky, f_x, f_gnd_pv_shade, f_gnd_pv_noshade)
    poa_dif_pv = poa_diffuse_pv(poa_gnd_pv, poa_sky_pv)
    poa_dir_pv = poa_direct_pv(poa_direct, iam, f_x)
    poa_glo_pv = poa_global_pv(poa_dir_pv, poa_dif_pv)
    output = poa_glo_pv, poa_dir_pv, poa_dif_pv, poa_gnd_pv, poa_sky_pv
    if all_output:
        output += (
            tan_phi, f_gnd_sky, df, poa_gnd_sky, f_x,
            tan_psi_top, tan_psi_top_0, f_sky_pv_shade, f_sky_pv_noshade,
            tan_psi_bottom, tan_psi_bottom_1, f_gnd_pv_shade, f_gnd_pv_noshade)
    return output


def get_poa_global_bifacial(solar_zenith, solar_azimuth, system_azimuth, gcr,
                            tilt, ghi, dhi, poa_ground, poa_sky_diffuse,
                            poa_direct, iam, bifaciality=0.8,
                            shade_factor=-0.02, transmission_factor=0,
                            is_rad=False):
    """Get global bifacial irradiance from infinite sheds model."""
    # convert angles to radians from degrees if necessary
    solar_zenith, solar_azimuth, system_azimuth, tilt = _to_radians(
        solar_zenith, solar_azimuth, system_azimuth, tilt, is_rad=is_rad)
    # get front side
    poa_global_front = get_irradiance(
        solar_zenith, solar_azimuth, system_azimuth, gcr, tilt, ghi, dhi,
        poa_ground, poa_sky_diffuse, poa_direct, iam, bifaciality,
        shade_factor, transmission_factor, is_rad=True)
    # backside is rotated and flipped relative to front
    backside_tilt, backside_sysaz = _backside(tilt, system_azimuth)
    # get backside
    poa_global_back = get_irradiance(
        solar_zenith, solar_azimuth, backside_sysaz, gcr, backside_tilt, ghi,
        dhi, poa_ground, poa_sky_diffuse, poa_direct, iam, bifaciality,
        shade_factor, transmission_factor, is_rad=True)
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
    def __init__(self,system_azimuth, gcr, tilt, is_bifacial=True,
                 bifaciality=0.8, shade_factor=-0.02, transmission_factor=0,
                 is_rad=False):
        # convert angles to radians from degrees if necessary
        system_azimuth, tilt = _to_radians(system_azimuth, tilt, is_rad=is_rad)
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

    def _backside(self):
        backside_tilt = np.pi - self.tilt
        backside_sysaz = (np.pi + self.system_azimuth) % (2*np.pi)
        return backside_tilt, backside_sysaz

    def get_irradiance(self, solar_zenith, solar_azimuth, ghi, dhi, poa_ground,
                       poa_sky_diffuse, poa_direct, iam, is_rad=False):
        # convert angles to radians from degrees if necessary
        solar_zenith, solar_azimuth = _to_radians(
            solar_zenith, solar_azimuth, is_rad=is_rad)
        self.front_side = _PVSurface(*get_irradiance(
            solar_zenith, solar_azimuth, self.system_azimuth, self.gcr,
            self.tilt, ghi, dhi, poa_ground, poa_sky_diffuse, poa_direct, iam,
            is_rad=True, all_output=True))
        self.tan_phi = self.front_side.tan_phi
        self.f_gnd_sky = self.front_side.f_gnd_sky
        self.df = self.front_side.df
        if self.is_bifacial and self.bifaciality > 0:
            self.back_side = _PVSurface(*get_irradiance(
                solar_zenith, solar_azimuth, self.backside_sysaz, self.gcr,
                self.backside_tilt, ghi, dhi, poa_ground, poa_sky_diffuse,
                poa_direct, iam, is_rad=True, all_output=True))
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