from pvlib.irradiance import perez, aoi_projection
from numpy import np
import pvlib
from scipy.integrate import quad
from pvlib.tools import cosd, sind, tand, atand, acotd, acosd


def albedo_model(surftilt, surfaz, etoh, albedo,
                 dhi, dni, hextra, sunzen, sunaz, am,
                 model='1990'):
    """
    calculates the collection of ground-reflected
    albedo light on the rear surface of a PV module while fully accounting
    for self-shading.

    This code is part of the Purdue Bifacial irradiance model [1] and it can
    simulate the albedo light intensity on both the front and rear sides of a
    bifacial solar module. This model takes two types of self-shading losses
    into account: 1) direct blocking of direct beam and circumsolar light by
    the module onto the ground and 2) sky masking of isotropic diffuse light
    by the module. This model employs a view-factor based approach and the
    detailed methodology is discussed in [1].

    Parameters
    ----------
    surftilt - numeric, array
        a scalar or vector of surface tilt angles in decimal degrees.
        If surftilt is a vector it must be of the same size as all other
        vector inputs. surftilt must be >=0 and <=180. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90).
    surfaz - numeric, array
        a scalar or vector of surface azimuth angles in decimal degrees.
        If surfaz is a vector it must be of the same size as all other vector
        inputs. surfaz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north.
        (e.g. North = 0, East = 90, West = 270)
    etoh - numeric, array
        a scalar or vector of the ratio of module elevation(E) to module
        height(H) Module height is the module dimension not parallel to
        the ground. If etoh is a vector it must be of the same size as
        all other vector inputs. etoh must be >=0.
    albedo - numeric, array
        a scalar or vector of groud albedo coefficient.
        If albedo is a vector it must be of the same size as all other vector
        inputs. albedo must be >=0 and <=1.
    dhi - numeric, array
        a scalar or vector of diffuse horizontal irradiance in W/m^2.
        If dhi is a vector it must be of the same size as all other vector
        inputs. dhi must be >=0.
    dni - numeric, array
        a scalar or vector of direct normal irradiance in W/m^2. If dni
        is a vector it must be of the same size as all other vector inputs
        dni must be >=0.
    hextra - numeric, array
        a scalar or vector of extraterrestrial normal irradiance in
        W/m^2. If |hextra| is a vector it must be of the same size as
        all other vector inputs. |hextra| must be >=0.
    sunzen - numeric, array
        a scalar or vector of apparent (refraction-corrected) zenith
        angles in decimal degrees. If |sunzen| is a vector it must be of the
        same size as all other vector inputs. |sunzen| must be >=0 and <=180.
    sunaz - numeric, array
        a scalar or vector of sun azimuth angles in decimal degrees.
        If sunaz is a vector it must be of the same size as all other vector
        inputs. sunaz must be >=0 and <=360. The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, East = 90, West = 270).
    am - numeric, array
        a scalar or vector of relative (not pressure-corrected) airmass
        values. If am is a vector it must be of the same size as all other
        vector inputs. am must be >=0.
    model - string
        a character string which selects the desired set of Perez
        coefficients. If model is not provided as an input, the default,
        '1990' will be used.
        All possible model selections are:
        '1990', 'allsitescomposite1990' (same as '1990'),
        'allsitescomposite1988', 'sandiacomposite1988',
        'usacomposite1988', 'france1988', 'phoenix1988',
        'elmonte1988', 'osage1988', 'albuquerque1988',
        'capecanaveral1988', or 'albany1988'

    Returns
    -------
    I_Alb - numeric, array
        the total ground-reflected albedo irradiance incident to the
        specified surface. I_Alb is a column vector vector with a number
        of elements equal to the input vector(s).

    References
    ----------
    .. [1] Sun, X., Khan, M. R., Alam, M. A., 2018. Optimization and
       performance of bifacial solar modules: A global perspective.
       Applied Energy 212, pp. 1601-1610.
    .. [2] Khan, M. R., Hanna, A., Sun, X., Alam, M. A., 2017. Vertical
       bifacial solar farms:Physics, design, and global optimization.
       Applied Energy, 206, 240-248.
    .. [3] Duffie, J. A., Beckman, W. A. 2013. Solar Engineering of Thermal
       Processes (4th Editio). Wiley.

    See Also
    --------
    bifacial_irradiance
    """

    vectorsizes = [len(np.ravel(surftilt)),
                   len(np.ravel(surfaz)), len(np.ravel(dhi)),
                   len(np.ravel(dni)), len(np.ravel(hextra)),
                   len(np.ravel(sunzen)), len(np.ravel(sunaz)),
                   len(np.ravel(am))]

    maxvectorsize = max(vectorsizes)

    assert np.sum(vectorsizes == maxvectorsize) == len(vectorsizes)

    # Calculate the diffuse light onto the ground by the Perez model
    i_alb_iso_g = np.zeros_like(maxvectorsize)
    i_alb_cir_g = np.zeros_like(maxvectorsize)
    for i in range(maxvectorsize):
        _, i_alb_iso_g[i], i_alb_cir_g[i], _ = perez(0.0, 0.0, dhi[i], dni[i],
                                                     hextra[i], sunzen[i],
                                                     sunaz[i], am[i],
                                                     model=model,
                                                     return_components=True)

    # Calculate the albedo light from the ground-reflected istropic
    # diffuse light (self-shading: sky masking)
    # see equation 11 in [1]

    i_alb_iso = i_alb_iso_g * albedo * vf_integral_diffuse(surftilt, etoh)

    # Calculate the albedo light from the ground-reflected circumsolar
    # diffuse and direct beam light (self-shading: direct blocking)
    # Self-shading of direct beam light
    vf_direct, shadowl_direct = vf_shadow(surfaz, surftilt,
                                          sunaz, sunzen, etoh)
    cirzen = sunzen
    cirzen[cirzen > 85] = 85
    vf_circum, shadowl_circum = vf_shadow(surfaz, surftilt,
                                          sunaz, cirzen, etoh)
    # Self-shading of circumsolar diffuse light
    # see equation 9 in [1]
    factor_direct = ((1 - cosd(surftilt)) / 2 - vf_direct * shadowl_direct)
    i_alb_direct = albedo * factor_direct * dni * cosd(sunzen)
    factor_cir = ((1 - cosd(surftilt)) / 2 - vf_circum * shadowl_circum)
    i_alb_cir = albedo * factor_cir * i_alb_cir_g

    # Sum up the total albedo light
    i_alb = i_alb_iso + i_alb_direct + i_alb_cir

    return i_alb


def integrand(x, a, b):
    """
    a = EtoW(i)
    b = surftilt(i)

    """
    # theta1 in Fig. 3 of Ref. [1]
    theta1 = (x < 0) * (180.0 - (acotd(-x / a))) + (x >= 0) * (acotd(-x / a))

    # theta2 in Fig. 3 of the Ref. [1]
    fac_1 = (acotd((cosd(180 - b) - x) / (a + sind(180 - b))))
    fac_2 = (180 - (acotd((x - cosd(180 - b)) / (a + sind(180 - a)))))
    theta2 = (x < cosd(180.0 - b)) * fac_1 + (x >= cosd(180 - b)) * fac_2

    # define integral term
    fac_int = (1 - (cosd(theta1) + cosd(theta2)) / 2)
    integ_term = fac_int * (cosd(theta1) + cosd(theta2)) / 2

    return integ_term


def vf_integral_diffuse(surftilt, etow):
    """
    This function is used to calculate the integral
    of view factors in eqn. 11 of Ref. [1]
    """

    vf_integral = np.zeros_like(surftilt, dtype=float)

    for i in range(len(surftilt)):

        # calculate xmin of the integral
        xmin = -etow[i] / tand(180.0 - surftilt[i])

        # perform integral
        vf_integral[i] = quad(integrand, xmin, np.inf,
                              args=(etow[i], surftilt[i]))[0]

        if surftilt[i] == 0:
            vf_integral[i] = 0

    return vf_integral


def vf_shadow(panel_azimuth, panel_tilt, azimuthangle_sun, zenithangle_sun,
              etow):
    """
    This function is used to calculate the view factor from the shaded ground
    to the module and the shadow length in eqn. 9 of Ref. [1]
    Please refer to Refs. [2,3] for the analytical equations used here
    """

    # limit to two parallel cases
    panel_tilt = (180.0 - np.array(panel_tilt, dtype=float))

    # consider the back of the module
    panel_azimuth = np.array(panel_azimuth, dtpe=float) + 180.0

    # parallel plate case
    panel_tilt[panel_tilt == 0] = 10 ** -4

    panel_azimuth[panel_azimuth >= 360] -= 360.0

    # Calculate AOI

    zenithangle_sun = np.array(zenithangle_sun, dtype=float)
    azimuthangle_sun = np.array(azimuthangle_sun, dtype=float)

    cosd_term = cosd(zenithangle_sun) * cosd(panel_tilt)
    sind_term = sind(panel_tilt) * sind(zenithangle_sun)
    temp = cosd_term + sind_term * cosd(azimuthangle_sun - panel_azimuth)
    temp[temp > 1] = 1
    temp[temp < -1] = -1
    aoi = acosd(temp)
    # aoi = aoi(:)

    # Calculate view factor
    azi_cos = cosd(panel_azimuth - azimuthangle_sun)
    tan_zen = tand(90.0 - zenithangle_sun)
    shadowextension = azi_cos * (sind(panel_tilt) / tan_zen)

    # shadow length
    shadowl = shadowextension + cosd(panel_tilt)

    thetaz = atand(tan_zen / azi_cos)

    h = etow / tand(thetaz) + etow / tand(panel_tilt)

    p = etow / sind(panel_tilt)

    vf = viewfactor_gap(1, shadowl, p, h, panel_tilt)

    vf[cosd(aoi) <= 0] = 0  # no shadow is cast

    return vf, shadowl


def viewfactor_gap(b, a, p, h, alpha):
    """
    calculate the view factor from a to b (infinite lines with alpha angle
    with distance to their cross point (b:p, a:h))
    """

    # first part
    vf1 = viewfactor_cross(b + p, h, alpha)  # h to b+p

    vf2 = viewfactor_cross(p, h, alpha)  # h to p

    vf3 = vf1 - vf2  # h to b

    vf3 = (vf3 * h) / b  # b to h

    # second part
    vf1_2 = viewfactor_cross(b + p, a + h, alpha)  # a+h to b+p

    vf2_2 = viewfactor_cross(p, a + h, alpha)  # a+h to p

    vf3_2 = vf1_2 - vf2_2  # a+h to b

    vf3_2 = (vf3_2 * (a + h)) / b  # b to a+h

    # third part
    vf3_3 = vf3_2 - vf3  # b to a

    vf = vf3_3 * b / a  # a to b

    # if a = 0 or b =0
    if np.isnan(vf):
        return 0
    else:
        return vf


def viewfactor_cross(b, a, alpha):
    """
    calculate the view factor from a to b (infinite lines with alpha angle)
    """
    sqrt_term = np.sqrt(1 - (2 * b) / (a * cosd(alpha)) + (b / a) ** 2)
    vf = 1 / 2 * (1 + b / a - sqrt_term)

    if np.isnan(vf):
        return 0
    else:
        return vf


def bifacial_irradiance(surftilt, surfaz, etoh, albedo, dhi, dni,
                        hextra, sunzen, sunaz, am, model='1990',
                        rloss='Yes',
                        rloss_para=[0.16, 0.4244, -0.074]):
    """
    calculates the irradiance on the front and rear sides of a
    bifacial solar module while fully accounting for
    the self-shading losses.

    The Purdue Bifacial irradiance model [1] simulates the total irradiance
    including direct, diffuse, and albedo light, on both the front and rear
    sides of a bifacial solar module. This model applies an analytical
    view-factor based approach to explicitly account for the self-shading
    losses of albedo light due to (1) direct blocking of direct and
    circumsolar diffuse light and 2) sky masking of isotropic diffuse light
    onto the ground. This model also incorporates an optional reflection-loss
    model [4]. This model has been validated against data spanning from
    Africa, Europe, Asia, and North America
    (please refer [1] for more detail).

    Parameters
    ----------
    surftilt - numeric, array
        a scalar or vector of surface tilt angles in decimal degrees. The
        tilt angle is defined as degrees from horizontal (e.g. surface facing
        up = 0, surface facing horizon = 90). surftilt must be >=0 and <=180.
        If surftilt is a vector it must be of the same size as all other
        vector inputs.
    surfaz - numeric, array
        a scalar or vector of surface azimuth angles in decimal degrees.
        If surfaz is a vector it must be of the same size as all other vector
        inputs. surfaz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north
        (e.g. North = 0, East = 90, West = 270).
    etoh - numeric, array
        a scalar or vector of the ratio of module elevation(E) to module
        height(H). Module height is the module dimension not parallel
        to the ground. If etoh is a vector it must be of the same size
        as all other vector inputs. etoh must be >=0.
    albedo - numeric, array
        a scalar or vector of groud albedo coefficient.
        If albedo is a vector it must be of the same size as all other vector
        inputs. albedo must be >=0 and <=1.
    dhi - numeric, array
        a scalar or vector of diffuse horizontal irradiance in W/m^2.
        If dhi is a vector it must be of the same size as all other v
        ector inputs. dhi must be >=0.
    dni - numeric, array
        a scalar or vector of direct normal irradiance in W/m^2. If
        dni is a vector it must be of the same size as all other
        vector inputs. dni must be >=0.
    hextra - numeric, array
        a scalar or vector of extraterrestrial normal irradiance in
        W/m^2. If hextra is a vector it must be of the same size as
        all other vector inputs. hextra must be >=0.
    sunzen - numeric, array
        a scalar or vector of apparent (refraction-corrected) zenith
        angles in decimal degrees. If sunzen is a vector it must be of the
        same size as all other vector inputs. sunzen must be >=0 and <=180.
    sunaz - numeric, array
        a scalar or vector of sun azimuth angles in decimal degrees.
        If sunaz is a vector it must be of the same size as all other vector
        inputs. sunaz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north
        (e.g. North = 0, East = 90, West = 270).
    am - numeric, array
        a scalar or vector of relative (not pressure-corrected) airmass
        values. If am is a vector it must be of the same size as all other
        vector inputs. am must be >=0.
    model - string
        a character string which selects the desired set of Perez
        coefficients. If model is not provided as an input, the default,
        '1990' will be used.
        All possible model selections are:
        '1990', 'allsitescomposite1990' (same as '1990'),
        'allsitescomposite1988', 'sandiacomposite1988',
        'usacomposite1988', 'france1988', 'phoenix1988',
        'elmonte1988', 'osage1988', 'albuquerque1988',
        'capecanaveral1988', or 'albany1988'
    rloss - string
        a character string which determines the inclusion of reflection
        loss model. By default, 'Yes' will be used. If 'No' is input,
        reflection loss will be neglected.
    rloss_para - numeric, array
        a three-element vector represents the parameters
        (in order, ar, c1, and c2) in the reflection models in Ref. [4].
        By default a parameter set for a glass-faced silicon solar module,
        [ar = 0.16, cl = 4/(3*pi), c2 = -0.074], will be used.

    Returns
    -------
    front_irradiance - numeric, array
        the total irradiance including direct beam, diffuse,
        and albedo light on the front side. front_irradiance is a column
        vector of the same size as the input vector(s).
    rear_irradiance - numeric, array
        the total irradiance includig direct beam, diffuse,
        and albedo light on the rear side.  rear_irradiance is a column vector
        of the same size as the input vector(s).

    References
    ----------
    .. [1] Sun, X., Khan, M. R., Alam, M. A., 2018. Optimization and
       performance of bifacial solar modules: A global perspective.
       Applied Energy 212, pp. 1601-1610.
    .. [2] Khan, M. R., Hanna, A., Sun, X., Alam, M. A., 2017. Vertical
       bifacial solar farms: Physics, design, and global optimization.
       Applied Energy, 206, 240-248.
    .. [3] Duffie, J. A., Beckman, W. A. 2013. Solar Engineering of Thermal
       Processes (4th Edition). Wiley.
    .. [4] Martin, N., Ruiz, J. M. 2005. Annual angular reflection losses in
       PV modules. Progress in Photovoltaics: Research and Applications,
       13(1), 75-84.

    See Also
    --------
    albedo_model

    """

    vectorsizes = [len(np.ravel(surftilt)),
                   len(np.ravel(surfaz)), len(np.ravel(dhi)),
                   len(np.ravel(dni)),
                   len(np.ravel(hextra)), len(np.ravel(sunzen)),
                   len(np.ravel(sunaz)), len(np.ravel(am))]

    maxvectorsize = max(vectorsizes)

    assert np.sum(vectorsizes == maxvectorsize) == len(vectorsizes), \
        "array size mismatch."

    # Irradiance calculation
    # Front Side

    # Direct beam

    aoi_front = np.zeros_like(surftilt, dtype=float)
    for i in range(len(aoi_front)):
        aoi_front[i] = aoi_projection(surftilt[i], surfaz[i], sunzen[i],
                                      sunaz[i])
    aoi_front[(aoi_front > 90) | (aoi_front < 0)] = 90.0
    ib_front = dni * cosd(aoi_front)

    # Account for reflection loss
    if rloss == 'Yes':
        # rloss_beam_front, rloss_Iso_Front, rloss_albedo_front
        # = pvl_iam_martinruiz_components(surftilt,aoi_front,rloss_para)
        rloss_iso_front, rloss_albedo_front = \
            pvlib.iam.martin_ruiz_diffuse(surftilt, rloss_para[0],
                                          rloss_para[1], rloss_para[2])

        rloss_beam_front = pvlib.iam.martin_ruiz(aoi_front, rloss_para[0])
        # for horizon brightening

        aoi_hor_front = np.zeros_like(surftilt, dtype=float)

        for i in range(len(aoi_hor_front)):
            aoi_hor_front[i] = aoi_projection(surftilt[i],
                                              surfaz[i], 90.0, sunaz[i])

        aoi_hor_front[(aoi_hor_front > 90) | (aoi_hor_front < 0)] = 90
        # rloss_hor_front = pvl_iam_martinruiz_components(surftilt,
        #                                                 aoi_hor_front,
        #                                                 rloss_para)
        rloss_hor_front = pvlib.iam.martin_ruiz(aoi_hor_front, rloss_para[0])
    else:
        rloss_beam_front = 0
        rloss_iso_front = 0
        rloss_albedo_front = 0
        rloss_hor_front = 0

    ib_front = ib_front * (1 - rloss_beam_front)
    ib_front[ib_front < 0] = 0

    # Sky diffuse
    # Perez Diffuse
    id_iso_front = np.zeros(maxvectorsize)
    id_cir_front = np.zeros(maxvectorsize)
    id_hor_front = np.zeros(maxvectorsize)
    for i in range(maxvectorsize):
        _, id_iso_front[i], id_cir_front[i], id_hor_front[i] = \
            perez(surftilt[i],
                  surfaz[i],
                  dhi[i], dni[i],
                  hextra[i], sunzen[i],
                  sunaz[i], am[i],
                  model='1990',
                  return_components=True)

    id_iso_front = id_iso_front * (1 - rloss_iso_front)
    id_cir_front = id_cir_front * (1 - rloss_beam_front)
    id_hor_front = id_hor_front * (1 - rloss_hor_front)

    # albedo light
    i_alb_front = albedo_model(surftilt, surfaz,
                               etoh, albedo, dhi,
                               dni, hextra,
                               sunzen, sunaz, am,
                               model)
    i_alb_front = i_alb_front * (1 - rloss_albedo_front)
    i_alb_front[i_alb_front < 0] = 0.0

    # Sum up the front-side irradiance
    front_term_1 = ib_front + i_alb_front
    front_term_2 = id_iso_front + id_cir_front + id_hor_front
    front_irradiance = front_term_1 + front_term_2

    # Define angle for the rear side
    surftilt_rear = 180.0 - surftilt
    surfaz_rear = surfaz + 180.0
    for i in range(len(surfaz)):
        if surfaz_rear[i] >= 360.0:
            surfaz_rear[i] = surfaz_rear[i] - 360.0

    # Direct beam
    aoi_rear = np.zeros_like(surftilt_rear, dtype=float)
    for i in range(len(aoi_front)):
        aoi_rear[i] = aoi_projection(surftilt_rear[i], surfaz_rear[i],
                                     sunzen[i], sunaz[i])
    aoi_rear[(aoi_rear > 90) | (aoi_rear < 0)] = 90.0
    ib_rear = dni * cosd(aoi_rear)
    # Account for reflection loss
    if rloss == 'Yes':
        # rloss_Beam_Rear,rloss_Iso_Rear,rloss_albedo_Rear = \
        # pvl_iam_martinruiz_components(surftilt_rear,aoi_rear,rloss_para)
        rloss_iso_rear, rloss_albedo_rear = \
            pvlib.iam.martin_ruiz_diffuse(surftilt_rear,
                                          rloss_para[0],
                                          rloss_para[1],
                                          rloss_para[2])
        rloss_beam_rear = pvlib.iam.martin_ruiz(aoi_rear, rloss_para[0])
        # Horizon Brightening
        # aoi_hor_rear = pvl_getaoi(surftilt_rear, surfaz_rear,
        #                           90, surfaz_rear)
        # check here sunaz or surfaz_rear
        aoi_hor_rear = np.zeros_like(surftilt_rear, dtype=float)
        for i in range(len(aoi_front)):
            aoi_hor_rear[i] = aoi_projection(surftilt_rear[i],
                                             surfaz_rear[i],
                                             90.0, surfaz_rear[i])

        aoi_hor_rear[(aoi_hor_rear > 90) | (aoi_hor_rear < 0)] = 90.0
        # rloss_Hor_Rear = pvl_iam_martinruiz_components(surftilt_rear,
        #                                                aoi_hor_rear,
        #                                                rloss_para)
        rloss_hor_rear = pvlib.iam.martin_ruiz(aoi_hor_rear, rloss_para[0])
    else:
        rloss_beam_rear = 0
        rloss_iso_rear = 0
        rloss_albedo_rear = 0
        rloss_hor_rear = 0

    ib_rear = ib_rear * (1 - rloss_beam_rear)
    ib_rear[ib_rear < 0] = 0

    # Sky diffuse light
    id_iso_rear = np.zeros(maxvectorsize)
    id_cir_rear = np.zeros(maxvectorsize)
    id_hor_rear = np.zeros(maxvectorsize)
    for i in range(maxvectorsize):
        _, id_iso_rear[i], id_cir_rear[i], id_hor_rear[i] = \
            perez(surftilt_rear[i],
                  surfaz_rear[i],
                  dhi[i], dni[i],
                  hextra[i], sunzen[i],
                  sunaz[i], am[i],
                  model='1990',
                  return_components=True)

    id_iso_rear = id_iso_rear * (1 - rloss_iso_rear)
    id_cir_rear = id_cir_rear * (1 - rloss_beam_rear)
    id_hor_rear = id_hor_rear * (1 - rloss_hor_rear)

    # albedo light
    i_alb_rear = albedo_model(surftilt_rear, surfaz_rear,
                              etoh, albedo, dhi, dni,
                              hextra, sunzen,
                              sunaz, am, model)
    i_alb_rear = i_alb_rear * (1 - rloss_albedo_rear)
    i_alb_rear[i_alb_rear < 0] = 0

    # Sum up the rear-side irradiance
    rear_term_1 = ib_rear + i_alb_rear
    rear_term_2 = id_iso_rear + id_cir_rear + id_hor_rear
    rear_irradiance = rear_term_1 + rear_term_2

    return front_irradiance, rear_irradiance
