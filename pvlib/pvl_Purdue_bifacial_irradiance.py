from pvlib.tools import cosd, sind, tand, atand
from pvlib.irradiance import perez, aoi_projection
from numpy import np
from pvlib import iam, pvl_Purdue_albedo_model
import warnings


def pvl_Purdue_bifacial_irradiance(SurfTilt, SurfAz, EtoH, Albedo, DHI, DNI,
                                   HExtra, SunZen, SunAz, AM, model='1990', 
                                   Rloss = 'Yes', 
                                   Rloss_Para = [0.16, 0.4244, -0.074]):


    VectorSizes = [len(np.ravel(SurfTilt)), 
                len(np.ravel(SurfAz)), len(np.ravel(DHI)), len(np.ravel(DNI)),
                len(np.ravel(HExtra)), len(np.ravel(SunZen)),
                len(np.ravel(SunAz)), len(np.ravel(AM))]

    MaxVectorSize = max(VectorSizes);
    
    assert np.sum(VectorSizes == MaxVectorSize)==len(VectorSizes),"array size mismatch."


    # Irradiance calculation
    # Front Side

    # Direct beam 
    
    AOI_Front = np.zeros_like(SurfTilt, dtype = float)
    for i in range(len(AOI_Front)): 
        AOI_Front[i] = aoi_projection(SurfTilt[i], SurfAz[i], SunZen[i], SunAz[i])
    AOI_Front[(AOI_Front>90)|(AOI_Front<0)] = 90.0
    IB_Front = DNI * cosd(AOI_Front); 

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
        # Rloss_Hor_Front = pvl_iam_martinruiz_components(SurfTilt,AOI_Hor_Front,Rloss_Para);
        Rloss_Beam_Front = pvlib.iam.martin_ruiz(AOI_Hor_Front, Rloss_Para[0])
    else:
        Rloss_Beam_Front = 0
        Rloss_Iso_Front = 0
        Rloss_Albedo_Front = 0
        Rloss_Hor_Front = 0
    
    
    IB_Front  = IB_Front * (1-Rloss_Beam_Front);
    IB_Front[IB_Front<0] = 0;
                                             
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
    IB_Rear = DNI * cosd(AOI_Rear);
    # Account for reflection loss
    if Rloss == 'Yes':
        # Rloss_Beam_Rear,Rloss_Iso_Rear,Rloss_Albedo_Rear = pvl_iam_martinruiz_components(SurfTilt_Rear,AOI_Rear,Rloss_Para)
        Rloss_Iso_Rear,Rloss_Albedo_Rear = pvlib.iam.martin_ruiz_diffuse(SurfTilt_Rear, Rloss_Para[0], Rloss_Para[1], Rloss_Para[2])
        Rloss_Beam_Rear = pvlib.iam.martin_ruiz(AOI_Rear, Rloss_Para[0])
        # Horizon Brightening
        # AOI_Hor_Rear = pvl_getaoi(SurfTilt_Rear, SurfAz_Rear, 90, SurfAz_Rear);
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
    Rear_Irradiance =  IB_Rear + I_Alb_Rear + ID_Iso_Rear+ ID_Cir_Rear+ ID_Hor_Rear;
    
    return Front_Irradiance,Rear_Irradiance                  