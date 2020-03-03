from pvlib.tools import cosd, sind, tand, atand
from pvlib.irradiance import perez
from numpy import np
from scipy.integrate import quad
import warnings


def pvl_Purdue_albedo_model(SurfTilt, SurfAz, EtoH, Albedo,
                             DHI, DNI, HExtra, SunZen, SunAz, AM, model = '1990'):


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
    '''
    a = EtoW(i)
    b = SurfTilt(i)

    '''
    # theta1 in Fig. 3 of Ref. [1]
    theta1 = (x<0)*(180.0-(acotd(-x/a))) + (x>=0)*(acotd(-x/a))
    
    # theta2 in Fig. 3 of the Ref. [1]
    theta2 = (x<cosd(180.0-b))*(acotd((cosd(180-b)-x)/(a+sind(180-b)))) + (x>=cosd(180-b))*(180-(acotd((x-cosd(180-b))/(a+sind(180-a)))))
    
    # define integral term
    integ_term = (1-(cosd(theta1)+cosd(theta2))/2)*(cosd(theta1)+cosd(theta2))/2

    return integ_term 


def VF_Integral_Diffuse(SurfTilt, EtoW):
    '''
    This function is used to calculate the integral of view factors in eqn. 11 of Ref. [1]
    '''
    VF_Integral = np.zeros_like(SurfTilt, dtype = float);

    for i in range(len(SurfTilt)):

    
        xmin = -EtoW[i]/tand(180.0-SurfTilt[i]) # calculate xmin of the integral

        VF_Integral[i] = quad(integrand,xmin, np.inf, args=(EtoW[i], SurfTilt[i]))[0] # perform integral

        if (SurfTilt[i] == 0):

            VF_Integral[i] = 0

    
    return VF_Integral


def VF_Shadow(Panel_Azimuth, Panel_Tilt, AzimuthAngle_Sun, ZenithAngle_Sun,
              EtoW):
    '''
    This function is used to calculate the view factor from the shaded ground
    to the module and the shadow length in eqn. 9 of Ref. [1]
    Please refer to Refs. [2,3] for the analytical equations used here
    '''

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
    
    '''
    calculate the view factor from a to b (infinite lines with alpha angle
    with distance to their cross point (b:P, a:H))
    '''

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
    '''
    calculate the view factor from a to b (infinite lines with alpha angle)
    '''

    VF = 1/2 * (1 + b/a - sqrt(1 - (2*b)/(a*cosd(alpha))+(b/a)**2));

    if(np.isnan(VF)):
        return 0
    else:
        return VF

