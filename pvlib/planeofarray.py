"""
Contains methods to calculate total plane of array irradiance.
"""

import logging
pvl_logger = logging.getLogger('pvlib')

import os
import pdb

import numpy as np
import pandas as pd

import pvlib.pvl_tools as pvl_tools
import pvlib.diffuse_sky
import pvlib.diffuse_ground



def zenith_projection(surf_tilt, surf_az, sun_zen, sun_az):
    """
    Calculates the dot product of the solar vector and the surface normal.
    
    Input all angles in degrees.
    
    :param surf_tilt: float or Series. Panel tilt from horizontal.
    :param surf_az: float or Series. Panel azimuth from north.
    :param sun_zen: float or Series. Solar zenith angle.
    :param sun_az: float or Series. Solar azimuth angle.
    
    :returns: float or Series. Dot product of panel normal and solar angle.
    """
    
    projection = pvl_tools.cosd(surf_tilt)*pvl_tools.cosd(sun_zen) + pvl_tools.sind(surf_tilt)*pvl_tools.sind(sun_zen)*pvl_tools.cosd(sun_az - surf_az)
    
    try:
        projection.name = 'zenith_projection'
    except AttributeError:
        pass
    
    return projection



def poa_horizontal_ratio(surf_tilt, surf_az, sun_zen, sun_az):
    """
    Calculates the ratio of the beam components of the
    plane of array irradiance and the horizontal irradiance. 
    
    Input all angles in degrees.
    
    :param surf_tilt: float or Series. Panel tilt from horizontal.
    :param surf_az: float or Series. Panel azimuth from north.
    :param sun_zen: float or Series. Solar zenith angle.
    :param sun_az: float or Series. Solar azimuth angle.
    
    :returns: float or Series. Ratio of the plane of array irradiance to the
              horizontal plane irradiance
    """
    
    cos_poa_zen = zenith_projection(surf_tilt, surf_az, sun_zen, sun_az)
    
    cos_sun_zen = pvl_tools.cosd(sun_zen)
    
    # ratio of titled and horizontal beam irradiance
    ratio = cos_poa_zen / cos_sun_zen
    
    try:
        ratio.name = 'poa_ratio'
    except AttributeError:
        pass
    
    return ratio
    
    
    
def beam_component(surf_tilt, surf_az, sun_zen, sun_az, DNI):
    """
    Calculates the beam component of the plane of array irradiance.
    """
    beam = DNI * zenith_projection(surf_tilt, surf_az, sun_zen, sun_az)
    beam[beam < 0] = 0
    
    return beam
    


def isotropic(surf_tilt, surf_az, DNI, GHI, DHI, sun_zen, sun_az,
              albedo=.25, surface_type=None):
    '''
    Determine diffuse irradiance from the sky on a 
    tilted surface using the isotropic diffuse sky model.

    .. math::

       I = I_{h,b}R_b + I_{h,d} \frac{1 + \cos\beta}{2} + I_{h}\rho\frac{1 - \cos\beta}{2}

    Hottel and Woertz's model treats the sky as a uniform source of diffuse
    irradiance. Thus the diffuse irradiance from the sky (ground reflected
    irradiance is not included in this algorithm) on a tilted surface can
    be found from the diffuse horizontal irradiance and the tilt angle of
    the surface.

    Parameters
    ----------

    surf_tilt : float or Series
            Surface tilt angle in decimal degrees. 
            surf_tilt must be >=0 and <=180. The tilt angle is defined as
            degrees from horizontal (e.g. surface facing up = 0, surface facing
            horizon = 90)

    DHI : float or Series
            Diffuse horizontal irradiance in W/m^2.
            DHI must be >=0.


    Returns
    -------   

    DataFrame with columns 'total', 'beam', 'sky', 'ground'.


    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Hottel, H.C., Woertz, B.B., 1942. Evaluation of flat-plate solar heat
    collector. Trans. ASME 64, 91.

    See also    
    --------

    '''

    pvl_logger.debug('planeofarray.isotropic()')
    
    beam = beam_component(surf_tilt, surf_az, sun_zen, sun_az, DNI)
    
    sky = pvlib.diffuse_sky.isotropic(surf_tilt, DHI)
    
    ground = pvlib.diffuse_ground.get_diffuse_ground(surf_tilt, GHI,
                                                     albedo, surface_type)
    
    total = beam + sky + ground                                                
    
    all_irrad = pd.DataFrame({'total':total, 
                              'beam':beam, 
                              'sky':sky, 
                              'ground':ground})
    
    return all_irrad
    
    
    
def total_irrad(surf_tilt, surf_az, 
                sun_zen, sun_az,
                DNI, GHI, DHI, DNI_ET=None, AM=None,
                albedo=.25, surface_type=None,
                model='isotropic',
                model_perez='allsitescomposite1990'):
    '''
    Determine diffuse irradiance from the sky on a 
    tilted surface.

    .. math::

       I_{iso} = I_{h,b}R_b + I_{h,d} \frac{1 + \cos\beta}{2} + I_{h}\rho\frac{1 - \cos\beta}{2}


    Parameters
    ----------



    Returns
    -------   

    DataFrame with columns 'total', 'beam', 'sky', 'ground'.


    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267


    See also    
    --------

    '''

    pvl_logger.debug('planeofarray.total_irrad()')
    
    beam = beam_component(surf_tilt, surf_az, sun_zen, sun_az, DNI)
    
    model = model.lower()
    if model == 'isotropic':
        sky = pvlib.diffuse_sky.isotropic(surf_tilt, DHI)
    elif model == 'klutcher':
        sky = pvlib.diffuse_sky.klucher(surf_tilt, surf_az, DHI, GHI, sun_zen, sun_az)
    elif model == 'haydavies':
        sky = pvlib.diffuse_sky.haydavies(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az)
    elif model == 'reindl':
        sky = pvlib.diffuse_sky.reindl(surf_tilt, surf_az, DHI, DNI, GHI, DNI_ET, sun_zen, sun_az)
    elif model == 'king':
        sky = pvlib.diffuse_sky.king(surf_tilt, DHI, GHI, sun_zen)
    elif model == 'perez':
        sky = pvlib.diffuse_sky.perez(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az, AM, 
                                      modelt=model_perez)
    else:
        raise ValueError('invalid model selection {}'.format(model))
    
    ground = pvlib.diffuse_ground.get_diffuse_ground(surf_tilt, GHI,
                                                     albedo, surface_type)
    
    total = beam + sky + ground                                                
    
    all_irrad = pd.DataFrame({'total':total, 
                              'beam':beam, 
                              'sky':sky, 
                              'ground':ground})
    
    return all_irrad
    
    
    