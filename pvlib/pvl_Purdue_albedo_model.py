from pvlib.tools import cosd, sind, tand, atand
from pvlib.irradiance import perez
from numpy import np
from scipy.integrate import quad
import warnings


def pvl_Purdue_albedo_model(SurfTilt, SurfAz, EtoH, Albedo,
                             DHI, DNI, HExtra, SunZen, SunAz, AM, model = '1990'):
    
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
    SurfTilt - numeric, array
        a scalar or vector of surface tilt angles in decimal degrees.
        If SurfTilt is a vector it must be of the same size as all other
        vector inputs. SurfTilt must be >=0 and <=180. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90).
    SurfAz - numeric, array
        a scalar or vector of surface azimuth angles in decimal degrees.
        If SurfAz is a vector it must be of the same size as all other vector
        inputs. SurfAz must be >=0 and <=360. The Azimuth convention is
        defined as degrees east of north.
        (e.g. North = 0, East = 90, West = 270)
    EtoH - numeric, array 
        a scalar or vector of the ratio of module elevation(E) to module 
        height(H) Module height is the module dimension not parallel to 
        the ground. If EtoH is a vector it must be of the same size as 
        all other vector inputs. EtoH must be >=0.
    Albedo - numeric, array
        a scalar or vector of groud albedo coefficient.
        If Albedo is a vector it must be of the same size as all other vector
        inputs. Albedo must be >=0 and <=1.
    DHI - numeric, array
        a scalar or vector of diffuse horizontal irradiance in W/m^2.
        If DHI is a vector it must be of the same size as all other vector 
        inputs. DHI must be >=0.
    DNI - numeric, array
        a scalar or vector of direct normal irradiance in W/m^2. If DNI 
        is a vector it must be of the same size as all other vector inputs
        DNI must be >=0.
    HExtra - numeric, array
        a scalar or vector of extraterrestrial normal irradiance in
        W/m^2. If |HExtra| is a vector it must be of the same size as
        all other vector inputs. |HExtra| must be >=0.
    SunZen - numeric, array
        a scalar or vector of apparent (refraction-corrected) zenith
        angles in decimal degrees. If |SunZen| is a vector it must be of the
        same size as all other vector inputs. |SunZen| must be >=0 and <=180.
    SunAz - numeric, array
        a scalar or vector of sun azimuth angles in decimal degrees.
        If SunAz is a vector it must be of the same size as all other vector
        inputs. SunAz must be >=0 and <=360. The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, East = 90, West = 270).
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
    pvl_Purdue_Bifacial_irradiance
    """

    VectorSizes = [len(np.ravel(SurfTilt)), 
                len(np.ravel(SurfAz)), len(np.ravel(DHI)), len(np.ravel(DNI)),
                len(np.ravel(HExtra)), len(np.ravel(SunZen)),
                len(np.ravel(SunAz)), len(np.ravel(AM))]

    MaxVectorSize = max(VectorSizes);
    
    assert np.sum(VectorSizes == MaxVectorSize)==len(VectorSizes)

    # Calculate the diffuse light onto the ground by the Perez model
    I_Alb_Iso_G = np.zeros_like(MaxVectorSize)
    I_Alb_Cir_G = np.zeros_like(MaxVectorSize)
    for i in range(MaxVectorSize):
        _,I_Alb_Iso_G[i],I_Alb_Cir_G[i],_ = perez(0.0, 0.0, DHI[i], DNI[i],
                                                HExtra[i], SunZen[i],
                                                SunAz[i], AM[i], model='1990',
                                                return_components=True)

    # Calculate the albedo light from the ground-reflected istropic
    # diffuse light (self-shading: sky masking)
    # see equation 11 in [1]
    
    I_Alb_Iso = I_Alb_Iso_G*Albedo*VF_Integral_Diffuse(SurfTilt, EtoH)
    
    # Calculate the albedo light from the ground-reflected circumsolar
    # diffuse and direct beam light (self-shading: direct blocking)
    # Self-shading of direct beam light
    VF_Direct,ShadowL_Direct = VF_Shadow(SurfAz, SurfTilt,
                                        SunAz, SunZen, EtoH)
    CirZen = SunZen
    CirZen[CirZen>85] = 85; 
    VF_Circum,ShadowL_Circum = VF_Shadow(SurfAz, SurfTilt,
                                        SunAz,CirZen,EtoH)
    # Self-shading of circumsolar diffuse light
    # see equation 9 in [1]
    I_Alb_Direct = Albedo * ((1 - cosd(SurfTilt))/2 - VF_Direct * ShadowL_Direct) * DNI * cosd(SunZen)
    I_Alb_Cir = Albedo * ((1 - cosd(SurfTilt))/2 - VF_Circum * ShadowL_Circum) * I_Alb_Cir_G
    
    # Sum up the total albedo light
    I_Alb = I_Alb_Iso + I_Alb_Direct + I_Alb_Cir

    return I_Alb

def integrand(x, a, b):
    
    """
    a = EtoW(i)
    b = SurfTilt(i)

    """
    # theta1 in Fig. 3 of Ref. [1]
    theta1 = (x<0)*(180.0-(acotd(-x/a))) + (x>=0)*(acotd(-x/a))
    
    # theta2 in Fig. 3 of the Ref. [1]
    theta2 = (x<cosd(180.0-b))*(acotd((cosd(180-b)-x)/(a+sind(180-b)))) + (x>=cosd(180-b))*(180-(acotd((x-cosd(180-b))/(a+sind(180-a)))))
    
    # define integral term
    integ_term = (1-(cosd(theta1)+cosd(theta2))/2)*(cosd(theta1)+cosd(theta2))/2

    return integ_term 


def VF_Integral_Diffuse(SurfTilt, EtoW):
    
    """
    This function is used to calculate the integral of view factors in eqn. 11 of Ref. [1]
    """

    VF_Integral = np.zeros_like(SurfTilt, dtype = float);

    for i in range(len(SurfTilt)):

    
        xmin = -EtoW[i]/tand(180.0-SurfTilt[i]) # calculate xmin of the integral

        VF_Integral[i] = quad(integrand,xmin, np.inf, args=(EtoW[i], SurfTilt[i]))[0] # perform integral

        if (SurfTilt[i] == 0):

            VF_Integral[i] = 0

    
    return VF_Integral


def VF_Shadow(Panel_Azimuth, Panel_Tilt, AzimuthAngle_Sun, ZenithAngle_Sun,
              EtoW):
    
    """
    This function is used to calculate the view factor from the shaded ground
    to the module and the shadow length in eqn. 9 of Ref. [1]
    Please refer to Refs. [2,3] for the analytical equations used here
    """

    Panel_Tilt = (180.0 -np.array(Panel_Tilt, dtype = float)) # limit to two parallel cases

    Panel_Azimuth = np.array(Panel_Azimuth, dtpe=float) + 180.0  # consider the back of the module

    Panel_Tilt[Panel_Tilt==0] = 10**-4 # parallel plate case

    Panel_Azimuth[Panel_Azimuth>=360] = Panel_Azimuth[Panel_Azimuth>=360] - 360.0


    # Calculate AOI

    ZenithAngle_Sun = np.array(ZenithAngle_Sun, dtype = float)
    AzimuthAngle_Sun = np.array(AzimuthAngle_Sun, dtype = float)

    temp = cosd(ZenithAngle_Sun)*cosd(Panel_Tilt)+sind(Panel_Tilt)*sind(ZenithAngle_Sun)*cosd(AzimuthAngle_Sun-Panel_Azimuth)
    temp[temp>1] = 1
    temp[temp<-1] = -1
    AOI = acosd(temp)
    # AOI = AOI(:)


    # Calculate view factor

    ShadowExtension = cosd(Panel_Azimuth-AzimuthAngle_Sun)*(sind(Panel_Tilt)/tand(90.0-ZenithAngle_Sun))
          
    ShadowL = ShadowExtension + cosd(Panel_Tilt) # shadow length

    ThetaZ = atand(tand(90.0-ZenithAngle_Sun)/cosd(Panel_Azimuth - AzimuthAngle_Sun))

    H = EtoW/tand(ThetaZ) + EtoW/tand(Panel_Tilt)

    P = EtoW/sind(Panel_Tilt)

    VF = ViewFactor_Gap(1,ShadowL,P,H,Panel_Tilt)

    VF[cosd(AOI) <= 0] = 0 # no shadow is cast

    return VF, ShadowL


def ViewFactor_Gap(b, a, P, H, alpha):    
    
    """
    calculate the view factor from a to b (infinite lines with alpha angle
    with distance to their cross point (b:P, a:H))
    """

    # first part
    VF1 = ViewFactor_Cross(b+P,H,alpha) # H to b+P

    VF2 = ViewFactor_Cross(P,H,alpha) # H to P

    VF3 = VF1 - VF2 # H to b

    VF3 = (VF3*H)/b # b to H

    # second part
    VF1_2 = ViewFactor_Cross(b+P,a+H,alpha) # a+H to b+P

    VF2_2 = ViewFactor_Cross(P,a+H,alpha) # a+H to P

    VF3_2 = VF1_2 - VF2_2 # a+H to b
    
    VF3_2 = (VF3_2*(a+H))/b # b to a+H

    # third part
    VF3_3 = VF3_2 - VF3 # b to a

    VF = VF3_3* b/a # a to b

    # if a = 0 or b =0
    if(np.isnan(VF)):
        return 0
    else:
        return VF

def ViewFactor_Cross(b , a, alpha):
    
    """
    calculate the view factor from a to b (infinite lines with alpha angle)
    """

    VF = 1/2 * (1 + b/a - sqrt(1 - (2*b)/(a*cosd(alpha))+(b/a)**2));

    if(np.isnan(VF)):
        return 0
    else:
        return VF

