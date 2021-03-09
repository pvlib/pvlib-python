from pvlib.tools import cosd, sind, tand, atand
from pvlib.irradiance import perez, aoi_projection
from numpy import np
from pvlib import iam, pvl_Purdue_albedo_model
import warnings


def pvl_Purdue_bifacial_irradiance(SurfTilt, SurfAz, EtoH, Albedo, DHI, DNI,
                                   HExtra, SunZen, SunAz, AM, model='1990', 
                                   Rloss = 'Yes', 
                                   Rloss_Para = [0.16, 0.4244, -0.074]):

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
    SurfTilt - numeric, array
        a scalar or vector of surface tilt angles in decimal degrees. The
        tilt angle is defined as degrees from horizontal (e.g. surface facing
        up = 0, surface facing horizon = 90). SurfTilt must be >=0 and <=180.
        If SurfTilt is a vector it must be of the same size as all other
        vector inputs.  
    SurfAz - numeric, array
        a scalar or vector of surface azimuth angles in decimal degrees.
        If SurfAz is a vector it must be of the same size as all other vector
        inputs. SurfAz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north
        (e.g. North = 0, East = 90, West = 270).
    EtoH - numeric, array
        a scalar or vector of the ratio of module elevation(E) to module 
        height(H). Module height is the module dimension not parallel
        to the ground. If EtoH is a vector it must be of the same size
        as all other vector inputs. EtoH must be >=0.
    Albedo - numeric, array
        a scalar or vector of groud albedo coefficient.
        If Albedo is a vector it must be of the same size as all other vector
        inputs. Albedo must be >=0 and <=1.
    DHI - numeric, array
        a scalar or vector of diffuse horizontal irradiance in W/m^2. 
        If DHI is a vector it must be of the same size as all other v
        ector inputs. DHI must be >=0.
    DNI - numeric, array
        a scalar or vector of direct normal irradiance in W/m^2. If
        DNI is a vector it must be of the same size as all other
        vector inputs. DNI must be >=0.
    HExtra - numeric, array
        a scalar or vector of extraterrestrial normal irradiance in
        W/m^2. If HExtra is a vector it must be of the same size as
        all other vector inputs. HExtra must be >=0.
    SunZen - numeric, array
        a scalar or vector of apparent (refraction-corrected) zenith
        angles in decimal degrees. If SunZen is a vector it must be of the
        same size as all other vector inputs. SunZen must be >=0 and <=180.
    SunAz - numeric, array
        a scalar or vector of sun azimuth angles in decimal degrees.
        If SunAz is a vector it must be of the same size as all other vector
        inputs. SunAz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north
        (e.g. North = 0, East = 90, West = 270).
    AM - numeric, array
        a scalar or vector of relative (not pressure-corrected) airmass
        values. If AM is a vector it must be of the same size as all other
        vector inputs. AM must be >=0.
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
    Rloss - string
        a character string which determines the inclusion of reflection 
        loss model. By default, 'Yes' will be used. If 'No' is input, 
        reflection loss will be neglected.
    Rloss_Para - numeric, array
        a three-element vector represents the parameters 
        (in order, ar, c1, and c2) in the reflection models in Ref. [4].
        By default a parameter set for a glass-faced silicon solar module, 
        [ar = 0.16, cl = 4/(3*pi), c2 = -0.074], will be used.  
    
    Returns
    -------
    Front_Irradiance - numeric, array
        the total irradiance including direct beam, diffuse,
        and albedo light on the front side. Front_Irradiance is a column 
        vector of the same size as the input vector(s).
    Rear_Irradiance - numeric, array
        the total irradiance includig direct beam, diffuse,
        and albedo light on the rear side.  Rear_Irradiance is a column vector
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
    pvl_Purdue_albedo_model
    
    """

    VectorSizes = [len(np.ravel(SurfTilt)), 
                len(np.ravel(SurfAz)), len(np.ravel(DHI)), len(np.ravel(DNI)),
                len(np.ravel(HExtra)), len(np.ravel(SunZen)),
                len(np.ravel(SunAz)), len(np.ravel(AM))]

    MaxVectorSize = max(VectorSizes)
    
    assert np.sum(VectorSizes == MaxVectorSize)==len(VectorSizes),"array size mismatch."


    # Irradiance calculation
    # Front Side

    # Direct beam 
    
    AOI_Front = np.zeros_like(SurfTilt, dtype = float)
    for i in range(len(AOI_Front)): 
        AOI_Front[i] = aoi_projection(SurfTilt[i], SurfAz[i], SunZen[i], SunAz[i])
    AOI_Front[(AOI_Front>90)|(AOI_Front<0)] = 90.0
    IB_Front = DNI * cosd(AOI_Front) 

    # Account for reflection loss
    if Rloss == 'Yes':
        # Rloss_Beam_Front, Rloss_Iso_Front, Rloss_Albedo_Front = pvl_iam_martinruiz_components(SurfTilt,AOI_Front,Rloss_Para)
        Rloss_Iso_Front, Rloss_Albedo_Front = pvlib.iam.martin_ruiz_diffuse(SurfTilt, Rloss_Para[0], Rloss_Para[1], Rloss_Para[2])
        Rloss_Beam_Front = pvlib.iam.martin_ruiz(AOI_Front, Rloss_Para[0])
        # for horizon brightening
        
        AOI_Hor_Front = np.zeros_like(SurfTilt, dtype = float)
        
        for i in range(len(AOI_Hor_Front)): 
            AOI_Hor_Front[i] = aoi_projection(SurfTilt[i], SurfAz[i], 90.0, SunAz[i])
    
        AOI_Hor_Front[(AOI_Hor_Front>90)|(AOI_Hor_Front<0)]=90
        # Rloss_Hor_Front = pvl_iam_martinruiz_components(SurfTilt,AOI_Hor_Front,Rloss_Para)
        Rloss_Beam_Front = pvlib.iam.martin_ruiz(AOI_Hor_Front, Rloss_Para[0])
    else:
        Rloss_Beam_Front = 0
        Rloss_Iso_Front = 0
        Rloss_Albedo_Front = 0
        Rloss_Hor_Front = 0
    
    
    IB_Front  = IB_Front * (1-Rloss_Beam_Front)
    IB_Front[IB_Front<0] = 0
                                             
    # Sky diffuse
    # Perez Diffuse

    for i in range(MaxVectorSize):
        _,ID_Iso_Front[i],ID_Cir_Front[i],ID_Hor_Front[i] = perez(SurfTilt[i],
                                                         SurfAz[i],
                                                         DHI[i], DNI[i],
                                                         HExtra[i], SunZen[i],
                                                         SunAz[i], AM[i],
                                                         model='1990',
                                                         return_components=True)    


    ID_Iso_Front = ID_Iso_Front * (1-Rloss_Iso_Front) 
    ID_Cir_Front = ID_Cir_Front * (1-Rloss_Beam_Front)
    ID_Hor_Front = ID_Hor_Front * (1-Rloss_Hor_Front)
        
    # Albedo light
    I_Alb_Front = pvlib.pvl_Purdue_albedo_model(SurfTilt, SurfAz, EtoH, Albedo, DHI, DNI, HExtra, SunZen, SunAz, AM ,model)
    I_Alb_Front = I_Alb_Front * (1-Rloss_Albedo_Front)
    I_Alb_Front[I_Alb_Front<0] = 0.0
    
    # Sum up the front-side irradiance
    Front_Irradiance = IB_Front + I_Alb_Front + ID_Iso_Front+ ID_Cir_Front+ ID_Hor_Front
    
    
    # Define angle for the rear side
    SurfTilt_Rear = 180.0 - SurfTilt
    SurfAz_Rear = SurfAz + 180.0
    for i in range(len(SurfAz)):
        if SurfAz_Rear[i] >= 360.0:
           SurfAz_Rear[i] = SurfAz_Rear[i] - 360.0
                                                         
                
    # Direct beam
    AOI_Rear = np.zeros_like(SurfTilt_Rear, dtype = float)
    for i in range(len(AOI_Front)): 
        AOI_Rear[i] = aoi_projection(SurfTilt_Rear[i], SurfAz_Rear[i], SunZen[i], SunAz[i])
    AOI_Rear[(AOI_Rear>90)|(AOI_Rear<0)] = 90.0
    IB_Rear = DNI * cosd(AOI_Rear)
    # Account for reflection loss
    if Rloss == 'Yes':
        # Rloss_Beam_Rear,Rloss_Iso_Rear,Rloss_Albedo_Rear = pvl_iam_martinruiz_components(SurfTilt_Rear,AOI_Rear,Rloss_Para)
        Rloss_Iso_Rear,Rloss_Albedo_Rear = pvlib.iam.martin_ruiz_diffuse(SurfTilt_Rear, Rloss_Para[0], Rloss_Para[1], Rloss_Para[2])
        Rloss_Beam_Rear = pvlib.iam.martin_ruiz(AOI_Rear, Rloss_Para[0])
        # Horizon Brightening
        # AOI_Hor_Rear = pvl_getaoi(SurfTilt_Rear, SurfAz_Rear, 90, SurfAz_Rear)
        # check here SunAz or SurfAz_Rear
        AOI_Hor_Rear = np.zeros_like(SurfTilt_Rear, dtype = float)
        for i in range(len(AOI_Front)): 
            AOI_Hor_Rear[i] = aoi_projection(SurfTilt_Rear[i], SurfAz_Rear[i], 90.0, SurfAz_Rear[i])

        AOI_Hor_Rear[(AOI_Hor_Rear>90)|(AOI_Hor_Rear<0)] = 90.0 
        # Rloss_Hor_Rear = pvl_iam_martinruiz_components(SurfTilt_Rear,AOI_Hor_Rear,Rloss_Para)
        Rloss_Hor_Rear = pvlib.iam.martin_ruiz(AOI_Hor_Rear, Rloss_Para[0])
    else:
        Rloss_Beam_Rear = 0
        Rloss_Iso_Rear = 0
        Rloss_Albedo_Rear = 0
        Rloss_Hor_Rear = 0


    IB_Rear = IB_Rear * (1-Rloss_Beam_Rear) 
    IB_Rear[IB_Rear<0] = 0
                                                   
    # Sky diffuse light

    for i in range(MaxVectorSize):
        _,ID_Iso_Rear[i],ID_Cir_Rear[i],ID_Hor_Rear[i] = perez(SurfTilt_Rear[i],
                                                         SurfAz_Rear[i],
                                                         DHI[i], DNI[i],
                                                         HExtra[i], SunZen[i],
                                                         SunAz[i], AM[i],
                                                         model='1990',
                                                         return_components=True)

    ID_Iso_Rear = ID_Iso_Rear * (1-Rloss_Iso_Rear)        
    ID_Cir_Rear = ID_Cir_Rear * (1-Rloss_Beam_Rear)
    ID_Hor_Rear = ID_Hor_Rear * (1-Rloss_Hor_Rear)  
    
    # Albedo light
    I_Alb_Rear = pvl_Purdue_albedo_model(SurfTilt_Rear, SurfAz_Rear, EtoH, Albedo, DHI, DNI, HExtra, SunZen, SunAz, AM ,model)
    I_Alb_Rear = I_Alb_Rear * (1-Rloss_Albedo_Rear)
    I_Alb_Rear[I_Alb_Rear<0] = 0
    
    # Sum up the rear-side irradiance
    Rear_Irradiance =  IB_Rear + I_Alb_Rear + ID_Iso_Rear+ ID_Cir_Rear+ ID_Hor_Rear
    
    return Front_Irradiance,Rear_Irradiance                  