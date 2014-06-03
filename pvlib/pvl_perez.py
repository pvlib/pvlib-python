'''
pvl_perez
=========

Collection of functions for determining diffuse irradiance from the sky on a tilted surface using one of the Perez models

'''




import numpy as np
import pandas as pd
import pvl_tools
import pdb    
def pvl_perez(SurfTilt, SurfAz, DHI, DNI, HExtra, SunZen, SunAz, AM,modelt='allsitescomposite1990'):
  ''' 
  Determine diffuse irradiance from the sky on a tilted surface using one of the Perez models

  Perez models determine the diffuse irradiance from the sky (ground
  reflected irradiance is not included in this algorithm) on a tilted
  surface using the surface tilt angle, surface azimuth angle, diffuse
  horizontal irradiance, direct normal irradiance, extraterrestrial
  irradiance, sun zenith angle, sun azimuth angle, and relative (not
  pressure-corrected) airmass. Optionally a selector may be used to use
  any of Perez's model coefficient sets.


  Parameters
  ----------
  
  SurfTilt : float or DataFrame
          Surface tilt angles in decimal degrees.
          SurfTilt must be >=0 and <=180. The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

  SurfAz : float or DataFrame
          Surface azimuth angles in decimal degrees.
          SurfAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, South=180 East = 90, West = 270).

  DHI : float or DataFrame
          diffuse horizontal irradiance in W/m^2. 
          DHI must be >=0.

  DNI : float or DataFrame
          direct normal irradiance in W/m^2. 
          DNI must be >=0.

  HExtra : float or DataFrame
          extraterrestrial normal irradiance in W/m^2. 
           HExtra must be >=0.
  
  SunZen : float or DataFrame
          apparent (refraction-corrected) zenith
          angles in decimal degrees. 
          SunZen must be >=0 and <=180.

  SunAz : float or DataFrame
          Sun azimuth angles in decimal degrees.
          SunAz must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, East = 90, West = 270).

  AM : float or DataFrame
          relative (not pressure-corrected) airmass 
          values. If AM is a DataFrame it must be of the same size as all other 
          DataFrame inputs. AM must be >=0 (careful using the 1/sec(z) model of AM
          generation)

  Other Parameters
  ----------------

  model : string (optional, default='allsitescomposite1990')

          a character string which selects the desired set of Perez
          coefficients. If model is not provided as an input, the default,
          '1990' will be used.
          All possible model selections are: 

          * '1990'
          * 'allsitescomposite1990' (same as '1990')
          * 'allsitescomposite1988'
          * 'sandiacomposite1988'
          * 'usacomposite1988'
          * 'france1988'
          * 'phoenix1988'
          * 'elmonte1988'
          * 'osage1988'
          * 'albuquerque1988'
          * 'capecanaveral1988'
          * 'albany1988'

  Returns
  --------

  SkyDiffuse : float or DataFrame

          the diffuse component of the solar radiation  on an
          arbitrarily tilted surface defined by the Perez model as given in
          reference [3].
          SkyDiffuse is the diffuse component ONLY and does not include the ground
          reflected irradiance or the irradiance due to the beam.
      

  References
  ----------

  [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
  solar irradiance on inclined surfaces for building energy simulation"
  2007, Solar Energy vol. 81. pp. 254-267

  [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D., 1987. A new
  simplified version of the Perez diffuse irradiance model for tilted
  surfaces. Solar Energy 39(3), 221-232.

  [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R., 1990.
  Modeling daylight availability and irradiance components from direct
  and global irradiance. Solar Energy 44 (5), 271-289. 

  [4] Perez, R. et. al 1988. "The Development and Verification of the
  Perez Diffuse Radiation Model". SAND88-7030

  See also
  --------
  pvl_ephemeris
  pvl_extraradiation
  pvl_isotropicsky
  pvl_haydavies1980
  pvl_reindl1990
  pvl_klucher1979
  pvl_kingdiffuse
  pvl_relativeairmass

  '''
  Vars=locals()
  Expect={'SurfTilt':('num','x>=0'),
      'SurfAz':('x>=-180'),
      'DHI':('x>=0'),
      'DNI':('x>=0'),
      'HExtra':('x>=0'),
      'SunZen':('x>=0'),
      'SunAz':('x>=-180'),
      'AM':('x>=0'),
      'modelt': ('default','default=allsitescomposite1990')}

  var=pvl_tools.Parse(Vars,Expect)

  kappa = 1.041 #for SunZen in radians
  z = var.SunZen*np.pi/180# # convert to radians

  Dhfilter = var.DHI > 0
  
  
  e = ((var.DHI[Dhfilter] + var.DNI[Dhfilter])/var.DHI[Dhfilter]+kappa*z[Dhfilter]**3)/(1+kappa*z[Dhfilter]**3).reindex_like(var.SunZen)
 


  ebin = pd.Series(np.zeros(var.DHI.shape[0]),index=e.index)

  # Select which bin e falls into
  ebin[(e<1.065)]= 1
  ebin[(e>=1.065) & (e<1.23)]= 2
  ebin[(e>=1.23) & (e<1.5)]= 3
  ebin[(e>=1.5) & (e<1.95)]= 4
  ebin[(e>=1.95) & (e<2.8)]= 5
  ebin[(e>=2.8) & (e<4.5)]= 6
  ebin[(e>=4.5) & (e<6.2)]= 7
  ebin[e>=6.2] = 8

  ebinfilter=ebin>0
  ebin=ebin-1 #correct for 0 indexing
  ebin[ebinfilter==False]=np.NaN
  ebin=ebin.dropna().astype(int)

  # This is added because in cases where the sun is below the horizon
  # (var.SunZen > 90) but there is still diffuse horizontal light (var.DHI>0), it is
  # possible that the airmass (var.AM) could be NaN, which messes up later
  # calculations. Instead, if the sun is down, and there is still var.DHI, we set
  # the airmass to the airmass value on the horizon (approximately 37-38).
  #var.AM(var.SunZen >=90 & var.DHI >0) = 37;

  var.HExtra[var.HExtra==0]=.00000001 #very hacky, fix this
  delt = var.DHI*var.AM/var.HExtra

  #

  # The various possible sets of Perez coefficients are contained
  # in a subfunction to clean up the code.
  F1c,F2c = GetPerezCoefficients(var.modelt)

  F1= F1c[ebin,0] + F1c[ebin,1]*delt[ebinfilter] + F1c[ebin,2]*z[ebinfilter]
  F1[F1<0]=0;
  F1=F1.astype(float)

  F2= F2c[ebin,0] + F2c[ebin,1]*delt[ebinfilter] + F2c[ebin,2]*z[ebinfilter]
  F2[F2<0]=0
  F2=F2.astype(float)

  A = pvl_tools.cosd(var.SurfTilt)*pvl_tools.cosd(var.SunZen) + pvl_tools.sind(var.SurfTilt)*pvl_tools.sind(var.SunZen)*pvl_tools.cosd(var.SunAz-var.SurfAz); #removed +180 from azimuth modifier: Rob Andrews October 19th 2012
  A[A < 0] = 0

  B = pvl_tools.cosd(var.SunZen);
  B[B < pvl_tools.cosd(85)] = pvl_tools.cosd(85)


  #Calculate Diffuse POA from sky dome

  #SkyDiffuse = pd.Series(np.zeros(var.DHI.shape[0]),index=data.index)

  SkyDiffuse = var.DHI[ebinfilter]*( 0.5* (1-F1[ebinfilter])*(1+pvl_tools.cosd(var.SurfTilt))+F1[ebinfilter] * A[ebinfilter]/ B[ebinfilter] + F2[ebinfilter]* pvl_tools.sind(var.SurfTilt))
  SkyDiffuse[SkyDiffuse <= 0]= 0


  return pd.DataFrame({'In_Plane_SkyDiffuse':SkyDiffuse})

def GetPerezCoefficients(perezmodelt):
  ''' 
  Find coefficients for the Perez model 

  Parameters
  ----------

  perezmodelt : string (optional, default='allsitescomposite1990')

          a character string which selects the desired set of Perez
          coefficients. If model is not provided as an input, the default,
          '1990' will be used.
  
  All possible model selections are: 

          * '1990'
          * 'allsitescomposite1990' (same as '1990')
          * 'allsitescomposite1988'
          * 'sandiacomposite1988'
          * 'usacomposite1988'
          * 'france1988'
          * 'phoenix1988'
          * 'elmonte1988'
          * 'osage1988'
          * 'albuquerque1988'
          * 'capecanaveral1988'
          * 'albany1988'

  Returns
  --------

          F1coeffs : array
          F1 coefficients for the Perez model
          F2coeffs : array
          F2 coefficients for the Perez model        

  References
  ----------

  [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
  solar irradiance on inclined surfaces for building energy simulation"
  2007, Solar Energy vol. 81. pp. 254-267

  [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D., 1987. A new
  simplified version of the Perez diffuse irradiance model for tilted
  surfaces. Solar Energy 39(3), 221-232.

  [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R., 1990.
  Modeling daylight availability and irradiance components from direct
  and global irradiance. Solar Energy 44 (5), 271-289. 

  [4] Perez, R. et. al 1988. "The Development and Verification of the
  Perez Diffuse Radiation Model". SAND88-7030

  '''
  coeffdict= {'allsitescomposite1990':
            [[-0.0080,    0.5880,   -0.0620,   -0.0600,    0.0720,   -0.0220],
             [ 0.1300,    0.6830,   -0.1510,   -0.0190,    0.0660,   -0.0290],
             [ 0.3300,    0.4870,   -0.2210,    0.0550,   -0.0640,   -0.0260],
             [ 0.5680,    0.1870,   -0.2950,    0.1090,   -0.1520,   -0.0140],
             [ 0.8730,   -0.3920,   -0.3620,    0.2260,   -0.4620,    0.0010],
             [ 1.1320,   -1.2370,   -0.4120,    0.2880,   -0.8230,    0.0560],
             [ 1.0600,   -1.6000,   -0.3590,    0.2640,   -1.1270,    0.1310],
             [ 0.6780,   -0.3270,   -0.2500,    0.1560,   -1.3770,    0.2510]],
             'allsitescomposite1988':
            [[-0.0180,    0.7050,   -0.071,0,   -0.0580,    0.1020,   -0.0260],
             [ 0.1910,    0.6450,   -0.1710,    0.0120,    0.0090,   -0.0270],
             [ 0.4400,    0.3780,   -0.2560,    0.0870,   -0.1040,   -0.0250],
             [ 0.7560,   -0.1210,   -0.3460,    0.1790,   -0.3210,   -0.0080],
             [ 0.9960,   -0.6450,   -0.4050,    0.2600,   -0.5900,    0.0170],
             [ 1.0980,   -1.2900,   -0.3930,    0.2690,   -0.8320,    0.0750],
             [ 0.9730,   -1.1350,   -0.3780,    0.1240,   -0.2580,    0.1490],
             [ 0.6890,   -0.4120,   -0.2730,    0.1990,   -1.6750,    0.2370]],
          
              'sandiacomposite1988':
             [[-0.1960,    1.0840,   -0.0060,   -0.1140,    0.1800,   -0.0190],
              [0.2360,    0.5190,   -0.1800,   -0.0110,    0.0200,   -0.0380],
              [0.4540,    0.3210,   -0.2550,    0.0720,   -0.0980,   -0.0460],
              [0.8660,   -0.3810,   -0.3750,    0.2030,   -0.4030,   -0.0490],
              [1.0260,   -0.7110,   -0.4260,    0.2730,   -0.6020,   -0.0610],
              [0.9780,   -0.9860,   -0.3500,    0.2800,   -0.9150,   -0.0240],
              [0.7480,   -0.9130,   -0.2360,    0.1730,   -1.0450,    0.0650],
              [0.3180,   -0.7570,    0.1030,    0.0620,   -1.6980,    0.2360]],
              'usacomposite1988':
              [[-0.0340,    0.6710,   -0.0590,   -0.0590,    0.0860,   -0.0280],
             [ 0.2550,    0.4740,   -0.1910,    0.0180,   -0.0140,   -0.0330],
             [ 0.4270,    0.3490,   -0.2450,    0.0930,   -0.1210,   -0.0390],
             [ 0.7560,   -0.2130,   -0.3280,    0.1750,   -0.3040,   -0.0270],
             [ 1.0200,   -0.8570,   -0.3850,    0.2800,   -0.6380,   -0.0190],
             [ 1.0500,   -1.3440,   -0.3480,    0.2800,   -0.8930,    0.0370],
             [ 0.9740,   -1.5070,   -0.3700,    0.1540,   -0.5680,    0.1090],
             [ 0.7440,   -1.8170,   -0.2560,    0.2460,   -2.6180,    0.2300]],
              'france1988':
          
             [[0.0130,    0.7640,   -0.1000,   -0.0580,    0.1270,   -0.0230],
              [0.0950,    0.9200,   -0.1520,         0,    0.0510,   -0.0200],
              [0.4640,    0.4210,   -0.2800,    0.0640,   -0.0510,   -0.0020],
              [0.7590,   -0.0090,   -0.3730,    0.2010,   -0.3820,    0.0100],
              [0.9760,   -0.4000,   -0.4360,    0.2710,   -0.6380,    0.0510],
              [1.1760,   -1.2540,   -0.4620,    0.2950,   -0.9750,    0.1290],
              [1.1060,   -1.5630,   -0.3980,    0.3010,   -1.4420,    0.2120],
              [0.9340,   -1.5010,   -0.2710,    0.4200,   -2.9170,    0.2490]],
              'phoenix1988':
            [[-0.0030,    0.7280,   -0.0970,   -0.0750,    0.1420,   -0.0430],
              [0.2790,    0.3540,   -0.1760,    0.0300,   -0.0550,   -0.0540],
              [0.4690,    0.1680,   -0.2460,    0.0480,   -0.0420,   -0.0570],
              [0.8560,   -0.5190,   -0.3400,    0.1760,   -0.3800,   -0.0310],
              [0.9410,   -0.6250,   -0.3910,    0.1880,   -0.3600,   -0.0490],
              [1.0560,   -1.1340,   -0.4100,    0.2810,   -0.7940,   -0.0650],
              [0.9010,   -2.1390,   -0.2690,    0.1180,   -0.6650,    0.0460],
              [0.1070,    0.4810,    0.1430,   -0.1110,   -0.1370,    0.2340]],
              'elmonte1988':
              [[0.0270,    0.7010,   -0.1190,   -0.0580,    0.1070 ,  -0.0600],
              [0.1810,    0.6710,   -0.1780,   -0.0790,    0.1940 ,  -0.0350],
              [0.4760,    0.4070,   -0.2880,    0.0540,   -0.0320 ,  -0.0550],
              [0.8750,   -0.2180,   -0.4030,    0.1870,   -0.3090 ,  -0.0610],
              [1.1660,   -1.0140,   -0.4540,    0.2110,   -0.4100 ,  -0.0440],
              [1.1430,   -2.0640,   -0.2910,    0.0970,   -0.3190 ,   0.0530],
              [1.0940,   -2.6320,   -0.2590,    0.0290,   -0.4220 ,   0.1470],
              [0.1550,    1.7230,    0.1630,   -0.1310,   -0.0190 ,   0.2770]],
              'osage1988':
             [[-0.3530,    1.4740 ,   0.0570,   -0.1750,    0.3120 ,   0.0090],
             [ 0.3630,    0.2180 ,  -0.2120,    0.0190,   -0.0340 ,  -0.0590],
             [-0.0310,    1.2620 ,  -0.0840,   -0.0820,    0.2310 ,  -0.0170],
             [ 0.6910,    0.0390 ,  -0.2950,    0.0910,   -0.1310 ,  -0.0350],
             [1.1820,   -1.3500 ,  -0.3210,    0.4080,   -0.9850 ,  -0.0880],
             [0.7640,    0.0190 ,  -0.2030,    0.2170,   -0.2940 ,  -0.1030],
             [0.2190,    1.4120 ,   0.2440,    0.4710,   -2.9880 ,   0.0340],
             [3.5780,   22.2310 , -10.7450,    2.4260,    4.8920 ,  -5.6870]],
              'albuquerque1988':
             [[0.0340,    0.5010,  -0.0940,   -0.0630,    0.1060 ,  -0.0440],
              [0.2290,    0.4670,  -0.1560,   -0.0050,   -0.0190 ,  -0.0230],
              [0.4860,    0.2410,  -0.2530,    0.0530,   -0.0640 ,  -0.0220],
              [0.8740,   -0.3930,  -0.3970,    0.1810,   -0.3270 ,  -0.0370],
              [1.1930,   -1.2960,  -0.5010,    0.2810,   -0.6560 ,  -0.0450],
              [1.0560,   -1.7580,  -0.3740,    0.2260,   -0.7590 ,   0.0340],
              [0.9010,   -4.7830,  -0.1090,    0.0630,   -0.9700 ,   0.1960],
              [0.8510,   -7.0550,  -0.0530,    0.0600,   -2.8330 ,   0.3300]],
              'capecanaveral1988':
             [[0.0750,    0.5330,   -0.1240 ,  -0.0670 ,   0.0420 ,  -0.0200],
             [ 0.2950,    0.4970,   -0.2180 ,  -0.0080 ,   0.0030 ,  -0.0290],
             [ 0.5140,    0.0810,   -0.2610 ,   0.0750 ,  -0.1600 ,  -0.0290],
             [ 0.7470,   -0.3290,   -0.3250 ,   0.1810 ,  -0.4160 ,  -0.0300],
             [ 0.9010,   -0.8830,   -0.2970 ,   0.1780 ,  -0.4890 ,   0.0080],
             [ 0.5910,   -0.0440,   -0.1160 ,   0.2350 ,  -0.9990 ,   0.0980],
             [ 0.5370,   -2.4020,    0.3200 ,   0.1690 ,  -1.9710 ,   0.3100],
             [-0.8050,    4.5460,    1.0720 ,  -0.2580 ,  -0.9500,    0.7530]],
              'albany1988':
             [[0.0120,    0.5540,   -0.0760 , -0.0520,   0.0840 ,  -0.0290],
              [0.2670,    0.4370,   -0.1940 ,  0.0160,   0.0220 ,  -0.0360],
              [0.4200,    0.3360,   -0.2370 ,  0.0740,  -0.0520 ,  -0.0320],
              [0.6380,   -0.0010,   -0.2810 ,  0.1380,  -0.1890 ,  -0.0120],
              [1.0190,   -1.0270,   -0.3420 ,  0.2710,  -0.6280 ,   0.0140],
              [1.1490,   -1.9400,   -0.3310 ,  0.3220,  -1.0970 ,   0.0800],
              [1.4340,   -3.9940,   -0.4920 ,  0.4530,  -2.3760 ,   0.1170],
              [1.0070,   -2.2920,   -0.4820 ,  0.3900,  -3.3680 ,   0.2290]],
              }
          
     
  array=np.array(coeffdict[perezmodelt])

  F1coeffs = array.T[0:3].T
  F2coeffs = array.T[3:7].T

  return F1coeffs ,F2coeffs










