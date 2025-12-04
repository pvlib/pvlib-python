r"""
Functions for the ANTS 2D bifacial irradiance model.
"""

import numpy as np
import pandas as pd
from pvlib.tools import cosd, sind, tand, acosd
from pvlib.bifacial import utils
from pvlib.irradiance import aoi_projection, haydavies, perez
from pvlib.shading import projected_solar_zenith_angle, shaded_fraction1d
from pvlib.tracking import calc_surface_orientation


def _shaded_fraction(tracker_rotation, phi, gcr, x0=0, x1=1):
    """
    Calculate fraction (from the bottom) of row slant height that is shaded
    from direct irradiance by the row in front toward the sun.

    Parameters
    ----------
    tracker_rotation : numeric
        Tracker rotation angle as a right-handed rotation around
        the axis defined by ``axis_tilt`` and ``axis_azimuth``.  For example,
        with ``axis_tilt=0`` and ``axis_azimuth=180``, ``tracker_theta > 0``
        results in ``surface_azimuth`` to the West while ``tracker_theta < 0``
        results in ``surface_azimuth`` to the East. [degree]
    phi : numeric
        Projected solar zenith angle. [degrees]
    gcr : numeric
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch). [unitless]
    x0 : numeric, default 0.
        Position on the row's slant length, as a fraction of the slant length.
        ``x0=0`` corresponds to the bottom of the row. ``x0`` should be less
        than ``x1``. [unitless]
    x1 : numeric, default 1.
        Position on the row's slant length, as a fraction of the slant length.
        ``x1`` should be greater than ``x0``. [unitless]

    Returns
    -------
    f_x : numeric
        Fraction of row slant height from the bottom that is shaded from
        direct irradiance.

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       https://www.nrel.gov/docs/fy20osti/76626.pdf
    """
    # keep track of scalar inputs so that we can have output match at the end
    squeeze = []
    if np.isscalar(x0) and np.isscalar(x1):
        squeeze.append(0)
    if np.isscalar(tracker_rotation):
        squeeze.append(1)

    # note: ground slope is already accounted for in phi and gcr, so don't
    # apply it here.
    # also, we have PSZA instead of solar position, so use fake azimuths to
    # trick shaded_fraction1d into accepting it as-is.
    # direction of positive phi by right-hand rule: 
    f_s = shaded_fraction1d(phi, solar_azimuth=90,
                            axis_azimuth=0,
                            shaded_row_rotation=tracker_rotation,
                            collector_width=1, pitch=1/gcr)

    # dimensions: row segment, time
    f_s = np.atleast_1d(f_s)[np.newaxis, :]
    x0 = np.atleast_1d(x0)[:, np.newaxis]
    x1 = np.atleast_1d(x1)[:, np.newaxis]

    swap = tracker_rotation < 0
    x0, x1 = np.where(swap, 1 - x1, x0), np.where(swap, 1 - x0, x1)

    f_s = np.clip((f_s - x0) / (x1 - x0), a_min=0, a_max=1)
    f_s = f_s.squeeze(axis=tuple(squeeze))

    return f_s


def _ants2d_singleside(tracker_rotation, cos_aoi, phi, gcr, height, pitch,
                       dni, dhi, ground_irradiance, albedo, x0, x1, g0, g1,
                       max_rows):
    r"""
    Calculate plane-of-array irradiance components on one side of a row
    of modules.

    Parameters
    ----------
    tracker_rotation : numeric
        Tracker rotation angle as a right-handed rotation around
        the axis defined by ``axis_tilt`` and ``axis_azimuth``.  For example,
        with ``axis_tilt=0`` and ``axis_azimuth=180``, ``tracker_theta > 0``
        results in ``surface_azimuth`` to the West while ``tracker_theta < 0``
        results in ``surface_azimuth`` to the East. [degree]
    cos_aoi : numeric
        Cosine of the angle of incidence of beam irradiance; can be
        calculated using :py:func:`pvlib.irradiance.aoi_projection`. [unitless]
    phi : numeric
        Project solar zenith angle; calculate with
        :py:func:`pvlib.shading.projected_solar_zenith_angle`. [degree]
    gcr : float
        Ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
    pitch : float
        Distance between two rows; must be in the same units as ``height``.
    dni : numeric
        Direct normal irradiance. [Wm⁻²]
    dhi : numeric
        Diffuse horizontal irradiance. [Wm⁻²]
    ground_irradiance : numeric
        Irradiance incident on the ground surface, partitioned according
        to ``x0`` and ``x1``. Sum of direct and diffuse components. [Wm⁻²]
    albedo : numeric
        Surface albedo. If a scalar, it is applied to all ground segments and
        timestamps.  Otherwise, must be specified as an array with shape
        (n_ground_segments, n_timestamps). [unitless]
    x0 : numeric, default 0
        Position on the row's slant length, as a fraction of the slant length.
        ``x0=0`` corresponds to the left side of the row.
        ``x0`` should be less than ``x1``.  If specified as array, it
        must have the same length as ``x1``. [unitless]
    x1 : numeric, default 1
        Position on the row's slant length, as a fraction of the slant length.
        ``x1=1`` corresponds to the right side of the row.
        ``x1`` should be greater than ``x0``. If specified as array, it
        must have the same length as ``x0``.[unitless]
    g0 : numeric
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g0=0`` corresponds to ground underneath the middle of the
        left row. ``g0`` should be less than ``g1``. If specified as array, it
        must have the same length as ``g1``.[unitless]
    g1 : numeric
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g1=1`` corresponds to ground underneath the middle of the
        right row. ``g1`` should be greater than ``g0``. If specified as array, it
        must have the same length as ``g0``.[unitless]
    max_rows : int
        Number of array units (sky wedges, ground segments, etc) to consider.
        [unitless]

    Returns
    -------
    output : dict of ``numpy.ndarray``

        ``output`` includes the following quantities:

        - ``poa_global``: total POA irradiance. [Wm⁻²]
        - ``poa_diffuse``: total diffuse POA irradiance from all sources.
          [Wm⁻²]
        - ``poa_direct``: direct POA irradiance. [Wm⁻²]
        - ``poa_sky_diffuse``: sky diffuse POA irradiance. [Wm⁻²]
        - ``poa_ground_diffuse``: ground-reflected diffuse POA irradiance.
          [Wm⁻²]
        - ``shaded_fraction``: fraction of row slant height from the bottom
          that is shaded from direct irradiance by adjacent rows. [unitless]

        Each array has shape (len(x0), len(tracker_rotation)).

    References
    ----------
    .. [1] TODO
    """
    # reminder of base dimensions: ground segment, row segment, time

    # in-plane beam component
    projection = np.array(np.clip(cos_aoi, a_min=0, a_max=None))
    row_shaded_fraction = _shaded_fraction(tracker_rotation, phi, gcr, x0, x1)
    poa_direct = dni * projection * (1 - row_shaded_fraction)
    poa_direct = poa_direct[0]  # drop ground segment dimension


    # in-plane sky diffuse component
    vf_row_sky = utils.vf_row_sky_2d_integ(tracker_rotation, gcr, x0, x1)
    poa_sky_diffuse = vf_row_sky * dhi
    poa_sky_diffuse = poa_sky_diffuse[0]  # drop ground segment dimension


    # in-plane ground-reflected component
    vf_row_ground = utils.vf_row_ground_2d_integ(surface_tilt=tracker_rotation,
                                                 gcr=gcr, height=height,
                                                 pitch=pitch,
                                                 x0=x0, x1=x1, g0=g0, g1=g1,
                                                 max_rows=max_rows)
    poa_ground_diffuse = vf_row_ground * albedo * ground_irradiance
    # sum over ground segments
    poa_ground_diffuse = np.sum(poa_ground_diffuse, axis=0)


    # add sky and ground-reflected irradiance on the row by irradiance
    # component
    poa_diffuse = poa_ground_diffuse + poa_sky_diffuse
    poa_global = poa_direct + poa_diffuse

    # all arrays are now 2D with shape (n_row_segments, len(tracker_rotation))
    output = {
        'poa_global': poa_global,
        'poa_direct': poa_direct,
        'poa_diffuse': poa_diffuse,
        'poa_sky_diffuse': poa_sky_diffuse,
        'poa_ground_diffuse': poa_ground_diffuse,
        'shaded_fraction': row_shaded_fraction
    }
    return output


def _apply_sky_diffuse_model(dni, dhi, model, solar_zenith, solar_azimuth,
                             dni_extra, airmass):

    if model in ['haydavies', 'perez']:
        # determine circumsolar irradiance, add it to DNI

        if model == 'haydavies':
            if dni_extra is None:
                raise ValueError(f'Must supply dni_extra for {model} model')
            diffuse_model_func = haydavies
            extra_kwargs = {}

        elif model == 'perez':
            # note: horizon brightening is ignored
            if dni_extra is None or airmass is None:
                raise ValueError(
                    f'Must supply dni_extra and airmass for {model} model')
            diffuse_model_func = perez
            extra_kwargs = {'airmass': airmass}

        kwargs = dict(
            dhi=dhi, dni=dni, dni_extra=dni_extra,
            solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
            return_components=True
        )
        # Call the model first time within the horizontal plane - to subtract
        # circumsolar_horizontal from DHI
        sky_diffuse_comps_horizontal = diffuse_model_func(
            surface_tilt=0, surface_azimuth=180, **kwargs, **extra_kwargs)
        circumsolar_horizontal = sky_diffuse_comps_horizontal['circumsolar']

        # Call the model a second time where circumsolar_normal is facing
        # directly towards sun, and can be added to DNI
        sky_diffuse_comps_normal = diffuse_model_func(
            surface_tilt=solar_zenith, surface_azimuth=solar_azimuth,
            **kwargs, **extra_kwargs)
        circumsolar_normal = sky_diffuse_comps_normal['circumsolar']

        dhi = dhi - circumsolar_horizontal
        dni = dni + circumsolar_normal
    elif model != 'isotropic':
        raise ValueError(f"Invalid model: {model}")

    return dni, dhi


def _apply_ground_slope(height, pitch, gcr, tracker_rotation, ghi, dni, dhi,
                        solar_zenith, solar_azimuth, axis_tilt, axis_azimuth,
                        cross_axis_slope):
    slope_azimuth = axis_azimuth + np.degrees(
        np.arctan2(sind(cross_axis_slope),
                   cosd(cross_axis_slope) * sind(axis_tilt))
    )
    slope_tilt = acosd(cosd(axis_tilt) * cosd(cross_axis_slope))

    height = height * cosd(slope_tilt)
    pitch = pitch / cosd(cross_axis_slope)
    gcr = gcr * cosd(cross_axis_slope)
    tracker_rotation = tracker_rotation - cross_axis_slope
    tracker_rotation = ((tracker_rotation + 180) % 360) - 180  # put back to [-180, 180]

    ghi = dhi + dni * np.maximum(
        aoi_projection(slope_tilt, slope_azimuth,
                       solar_zenith, solar_azimuth),
        0)
    # dhi: no need to adjust; the blocked view is only near the
    #      the horizon, and that part of the sky is blocked by rows anyway
    # dni: no adjustment needed; the measurement plane is not affected
    #dhi = dhi
    #dni = dni
    return height, pitch, gcr, tracker_rotation, ghi


def get_irradiance(tracker_rotation, axis_azimuth, solar_zenith, solar_azimuth,
                   gcr, height, pitch, ghi, dhi, dni,
                   albedo, model='perez', dni_extra=None, airmass=None,
                   row_segments=1, ground_segments=10, axis_tilt=0,
                   cross_axis_slope=0, max_rows=None,
                   return_ground_components=False):
    """
    Get front and rear irradiance using the ANTS-2D bifacial irradiance model.

    The ANTS-2D model [1] assumes the PV system comprises parallel,
    evenly spaced rows on flat or uniformly sloped ground. Rows can be on fixed
    racking or single axis trackers. The model calculates irradiance at a
    location far from the ends of any rows, in effect, assuming that the
    rows (sheds) are infinitely long.

    The model accounts for the following effects:

    - restricted view of the sky from module surfaces due to the nearby rows.
    - restricted view of the ground from module surfaces due to nearby rows.
    - restricted view of the sky from the ground due to rows.
    - shading of module surfaces by nearby rows.
    - nonuniform ground albedo.
    - sloped ground surface.

    Parameters
    ----------
    tracker_rotation : numeric
        Tracker rotation angle as a right-handed rotation around
        the axis defined by ``axis_tilt`` and ``axis_azimuth``.  For example,
        with ``axis_tilt=0`` and ``axis_azimuth=180``, ``tracker_theta > 0``
        results in ``surface_azimuth`` to the West while ``tracker_theta < 0``
        results in ``surface_azimuth`` to the East. [degree]
    axis_azimuth : numeric
        Axis azimuth angle in degrees.
        North = 0°; East = 90°; South = 180°; West = 270°
    solar_zenith : numeric
        Refraction-corrected solar zenith angle. [degree]
    solar_azimuth : numeric
        Solar azimuth angle. [degree]
    gcr : float
        Ground coverage ratio, ratio of row slant length to row spacing.
        [unitless]
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
    pitch : float
        Distance between two rows; must be in the same units as ``height``.
    ghi : numeric
        Global horizontal irradiance. [Wm⁻²]
    dhi : numeric
        Diffuse horizontal irradiance. [Wm⁻²]
    dni : numeric
        Direct normal irradiance. [Wm⁻²]
    albedo : numeric
        Surface albedo. If a scalar, it is applied to all ground segments and
        timestamps.  Otherwise, must be specified as an array with shape
        (``n_ground_segments``, ``len(tracker_rotation)``). [unitless]
    model : str, default 'perez'
        Irradiance model - can be one of 'isotropic', 'haydavies', or 'perez'.
    dni_extra : numeric, optional
        Extraterrestrial direct normal irradiance. Required when
        ``model='haydavies'`` or ``model='perez'``. [Wm⁻²]
    airmass : numeric, optional
        Relative airmass. Required when ``model='perez'``. [unitless]
    row_segments : int or list of pairs, default 1
        If ``row_segments`` is an int, it defines the number of equal-length
        segments the row width is divided into.  Otherwise, it must be a list
        of pairs ``(x0, x1)`` where ``x0`` and ``x1`` are fractions of the
        row width and ``x0 < x1``.  Irradiance will be computed and returned
        for each segment.
    ground_segments : int or list of pairs, default 10
        If ``ground_segments`` is an int, it defines the number of equal-length
        segments the ground surface is divided into.  Otherwise, it must be
        a list of pairs ``(g0, g1)`` where ``g0`` and ``g1`` are fractions of
        the pitch and ``g0 < g1``.  The pairs must be non-overlapping and must
        cover the entire ground surface.  ``albedo`` can be specified for
        each segment.
    axis_tilt : numeric, default 0
        Tilt of the axis of rotation with respect to horizontal. [degree]
    cross_axis_slope : numeric, default 0
        The angle, relative to horizontal, of the line formed by the
        intersection between the slope containing the tracker axes and a plane
        perpendicular to the tracker axes. The cross-axis slope should be
        specified using a right-handed convention. For example, trackers with
        axis azimuth of 180 degrees (heading south) will have a negative
        cross-axis tilt if the tracker axes plane slopes down to the east and
        positive cross-axis slope if the tracker axes plane slopes down to the
        west. Use :func:`~pvlib.tracking.calc_cross_axis_tilt` to calculate
        ``cross_axis_slope``. [degrees]
    max_rows : int, optional
        Number of array units (sky wedges, ground segments, etc) to consider.
        If not specified, units will be considered to within 4 degrees of the
        horizon. [unitless]
    return_ground_components : bool, default False
        If True, also return the direct and diffuse irradiance incident on the
        ground.  These values are returned in a second dict.

    Returns
    -------
    output : dict or DataFrame
        ``output`` is a DataFrame when inputs are Series
        and ``row_segments=1``, a dict of scalars when inputs are scalars
        and ``row_segments=1``, and a dict of ``np.ndarray``
        otherwise.  The following quantities are included:

        - ``poa_global``: sum of front- and back-side incident irradiance.
          [Wm⁻²]
        - ``poa_front``: total incident irradiance on the front surface. [Wm⁻²]
        - ``poa_back``: total incident irradiance on the back surface. [Wm⁻²]
        - ``poa_front_direct``: direct irradiance incident on the front
          surface. [Wm⁻²]
        - ``poa_front_diffuse``: total diffuse irradiance incident on the front
          surface. [Wm⁻²]
        - ``poa_front_sky_diffuse``: sky diffuse irradiance incident on the
          front surface. [Wm⁻²]
        - ``poa_front_ground_diffuse``: ground-reflected diffuse irradiance
          incident on the front surface. [Wm⁻²]
        - ``shaded_fraction_front``: fraction of row slant height that is
          shaded from direct irradiance on the front surface by adjacent
          rows. [unitless]
        - ``poa_back_direct``: direct irradiance incident on the back
          surface. [Wm⁻²]
        - ``poa_back_diffuse``: total diffuse irradiance incident on the back
          surface. [Wm⁻²]
        - ``poa_back_sky_diffuse``: sky diffuse irradiance incident on the
          back surface. [Wm⁻²]
        - ``poa_back_ground_diffuse``: ground-reflected diffuse irradiance
          incident on the back surface. [Wm⁻²]
        - ``shaded_fraction_back``: fraction of row slant height that is
          shaded from direct irradiance on the back surface by adjacent
          rows. [unitless]

    ground_irradiance : dict or DataFrame
        ``ground_irradiance`` is a DataFrame when inputs are Series
        and ``n_ground_segments=1``, a dict of scalars when inputs are scalars
        and ``n_ground_segments=1``, and a dict of ``np.ndarray``
        otherwise.  Only returned when ``return_ground_components=True``.
        The following quantities are included:

        - ``ground_direct``: direct irradiance incident on the ground surface.
          [Wm⁻²]
        - ``ground_diffuse``: diffuse irradiance incident on the ground
          surface. [Wm⁻²]

    References
    ----------
    .. [1] TODO
    """

    # so we can return scalars out if needed
    maybe_array_inputs = [
        tracker_rotation, axis_azimuth, solar_zenith, solar_azimuth,
        gcr, height, pitch, ghi, dhi, dni,
        albedo, dni_extra, airmass, axis_tilt, cross_axis_slope]
    all_scalar_inputs = all([
        np.isscalar(x) or x is None for x in maybe_array_inputs
    ])
    try:
        # get the index of the first pandas input, if there is one
        pd_index = next(
            x.index for x in maybe_array_inputs if isinstance(x, pd.Series)
        )
    except StopIteration:
        pd_index = None  # no pandas inputs

    # preparation steps

    dni, dhi = _apply_sky_diffuse_model(dni, dhi, model, solar_zenith,
                                        solar_azimuth, dni_extra, airmass)

    true_tracker_rotation = tracker_rotation

    if axis_tilt != 0 or cross_axis_slope != 0:
        height, pitch, gcr, tracker_rotation, ghi = _apply_ground_slope(
            height, pitch, gcr, tracker_rotation, ghi, dni, dhi,
            solar_zenith, solar_azimuth, axis_tilt, axis_azimuth,
            cross_axis_slope
        )

    if np.isscalar(row_segments):
        x_row = np.linspace(0, 1, row_segments+1)
        x0, x1 = x_row[:-1], x_row[1:]
    else:
        x0 = np.array([pair[0] for pair in row_segments])
        x1 = np.array([pair[1] for pair in row_segments])

    if np.isscalar(ground_segments):
        x_ground = np.linspace(0, 1, ground_segments+1)
        g0, g1 = x_ground[:-1], x_ground[1:]
    else:
        g0 = np.array([pair[0] for pair in ground_segments])
        g1 = np.array([pair[1] for pair in ground_segments])

    # dimensions: ground segment, row segment, time
    albedo = np.atleast_2d(albedo)[:, np.newaxis, :]
    ghi = np.atleast_1d(ghi)[np.newaxis, np.newaxis, :]
    dhi = np.atleast_1d(dhi)[np.newaxis, np.newaxis, :]
    dni = np.atleast_1d(dni)[np.newaxis, np.newaxis, :]
    tracker_rotation = np.atleast_1d(tracker_rotation)

    # Calculate some geometric quantities
    # rows to consider in front and behind current row
    # ensures that view factors to the sky are computed to within 4 degrees
    # of the horizon
    if max_rows is None:
        max_rows = np.ceil(height / (pitch * tand(4)))

    phi = projected_solar_zenith_angle(solar_zenith, solar_azimuth,
                                       axis_tilt, axis_azimuth)
    phi = phi - cross_axis_slope

    # compute this here, as it is expensive and does not differ between the
    # front and back sides
    vf_gnd_sky = utils.vf_ground_sky_2d_integ(
        tracker_rotation, gcr, height, pitch, g0=g0, g1=g1, max_rows=max_rows)
    vf_gnd_sky = vf_gnd_sky[:, np.newaxis, :]

    # irradiance components incident on ground surface
    ground_unshaded_fraction = utils._unshaded_ground_fraction(
        tracker_rotation, phi, gcr,
        pitch=pitch, height=height, g0=g0, g1=g1, max_rows=max_rows)

    ground_shaded_fraction = 1 - ground_unshaded_fraction
    ground_shaded_fraction = ground_shaded_fraction[:, np.newaxis, :]

    ground_direct = (1-ground_shaded_fraction) * (ghi - dhi)
    ground_diffuse = vf_gnd_sky * dhi
    ground_total = ground_direct + ground_diffuse

    # inputs shared between front and back calculations
    params = dict(phi=phi, gcr=gcr, height=height, pitch=pitch, dni=dni,
                  dhi=dhi, ground_irradiance=ground_total, albedo=albedo,
                  g0=g0, g1=g1, max_rows=max_rows)

    # front
    front_orientation = calc_surface_orientation(true_tracker_rotation,
                                                 axis_tilt, axis_azimuth)
    cos_aoi_front = aoi_projection(**front_orientation,
                                   solar_zenith=solar_zenith,
                                   solar_azimuth=solar_azimuth)
    poa_front = _ants2d_singleside(tracker_rotation, cos_aoi_front,
                                   x0=x0, x1=x1, **params)

    # back
    tracker_rotation_back = true_tracker_rotation + 180
    tracker_rotation_back = ((tracker_rotation_back + 180) % 360) - 180
    back_orientation = calc_surface_orientation(tracker_rotation_back,
                                                axis_tilt, axis_azimuth)
    cos_aoi_back = aoi_projection(**back_orientation,
                                  solar_zenith=solar_zenith,
                                  solar_azimuth=solar_azimuth)
    tracker_rotation_back = tracker_rotation + 180
    tracker_rotation_back = ((tracker_rotation_back + 180) % 360) - 180
    poa_back = _ants2d_singleside(tracker_rotation_back, cos_aoi_back,
                                  x0=1-x1, x1=1-x0, **params)

    colmap_front = {
        'poa_global': 'poa_front',
        'poa_direct': 'poa_front_direct',
        'poa_diffuse': 'poa_front_diffuse',
        'poa_sky_diffuse': 'poa_front_sky_diffuse',
        'poa_ground_diffuse': 'poa_front_ground_diffuse',
        'shaded_fraction': 'shaded_fraction_front',
    }
    colmap_back = {
        k: v.replace("front", "back") for k, v in colmap_front.items()
    }
    for old_key, new_key in colmap_front.items():
        poa_front[new_key] = poa_front.pop(old_key)
    for old_key, new_key in colmap_back.items():
        poa_back[new_key] = poa_back.pop(old_key)
    out = {**poa_front, **poa_back}

    if row_segments == 1:
        for k, v in out.items():
            out[k] = v[0]  # drop row segment dimension

        if all_scalar_inputs:
            # drop the second dimension too, so scalars are returned
            for k, v in out.items():
                out[k] = float(v[0])

        elif pd_index is not None:
            out = pd.DataFrame(out, index=pd_index)

    if return_ground_components:
        squeeze = []
        if ground_segments == 1:
            squeeze.append(0)  # drop ground segment dimension
        squeeze.append(1)  # always drop the row segment dimension
        if all_scalar_inputs:
            squeeze.append(2)  # drop time dimension
        squeeze = tuple(squeeze)
        out_ground = {
            'ground_direct': ground_direct.squeeze(axis=squeeze),
            'ground_diffuse': ground_diffuse.squeeze(axis=squeeze),
        }
        if squeeze == (0, 1, 2):
            for k, v in out_ground.items():
                out_ground[k] = float(v)

        if pd_index is not None and squeeze == (0, 1):
            out_ground = pd.DataFrame(out_ground, index=pd_index)

        return out, out_ground

    return out
