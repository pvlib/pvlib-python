
'''
 PVL_DISC Estimate Direct Normal Irradiance from Global Horizontal Irradiance using the DISC model

 Syntax
 [DNI, Kt] = pvl_disc(GHI,Z, doy)
 [DNI, Kt] = pvl_disc(GHI,Z, doy, pressure)

 Description
   The DISC algorithm converts global horizontal irradiance to direct
   normal irradiance through empirical relationships between the global
   and direct clearness indices. 

 Inputs:   
   GHI - a scalar or vector of global horizontal irradiance in W/m^2. If GHI
     is a vector it must be of the same size as all other vector inputs. 
     GHI must be >=0.
   Z - a scalar or vector of true (not refraction-corrected) zenith
     angles in decimal degrees. If Z is a vector it must be of the
     same size as all other vector inputs. Z must be >=0 and <=180.
   doy - a scalar or vector of values providing the day of the year. If
     doy is a vector it must be of the same size as all other vector inputs.
     doy must be >= 1 and < 367.
   pressure - a scalar or vector of values providing the site pressure in
     Pascal. If pressure is a vector it must be of the same size as all
     other vector inputs. pressure must be >=0. Pressure may be measured
     or an average pressure may be calculated from site altitude. If
     pressure is omitted, standard pressure (101325 Pa) will be used, this
     is acceptable if the site is near sea level. If the site is not near
     sea-level, inclusion of a measured or average pressure is highly
     recommended.

 Output:   
   DNI - the modeled direct normal irradiance in W/m^2 provided by the
     Direct Insolation Simulation Code (DISC) model. 
   Kt - Ratio of global to extraterrestrial irradiance on a horizontal
     plane.

 Sources

 [1] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly 
   Global Horizontal to Direct Normal Insolation", Technical 
   Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research 
   Institute, 1987.

 [2] J.W. "Fourier series representation of the position of the sun". 
   Found at:
   http://www.mail-archive.com/sundial@uni-koeln.de/msg01050.html on
   January 12, 2012

 See also PVL_DATE2DOY PVL_EPHEMERIS PVL_ALT2PRES PVL_DIRINT
'''


import numpy as np
import pvl_tools as pvt
import pandas as pd

def pvl_disc(GHI,SunZen,Time,pressure=101325):
    Vars=locals()
    Expect={'GHI': ('array','num','x>=0'),
            'SunZen': ('array','num','x<=180','x>=0'),
            'Time':'',
            'pressure':('num','default','default=101325','x>=0'),
            }

    var=pvt.Parse(Vars,Expect)

    #create a temporary dataframe to house masked values, initially filled with NaN
    temp=pd.DataFrame(index=var.Time,columns=['A','B','C'])


    var.pressure=101325
    doy=var.Time.dayofyear
    DayAngle=2.0 * np.pi*((doy - 1)) / 365
    re=1.00011 + 0.034221*(np.cos(DayAngle)) + (0.00128)*(np.sin(DayAngle)) + 0.000719*(np.cos(2.0 * DayAngle)) + (7.7e-05)*(np.sin(2.0 * DayAngle))
    I0=re*(1370)
    I0h=I0*(np.cos(np.radians(var.SunZen)))
    Ztemp=var.SunZen
    Ztemp[var.SunZen > 87]=87
    AM=1.0 / (np.cos(np.radians(Ztemp)) + 0.15*(((93.885 - Ztemp) ** (- 1.253))))*(var.pressure) / 101325
    Kt=var.GHI / (I0h)
    Kt[Kt < 0]=0
    temp.A[Kt > 0.6]=- 5.743 + 21.77*(Kt[Kt > 0.6]) - 27.49*(Kt[Kt > 0.6] ** 2) + 11.56*(Kt[Kt > 0.6] ** 3)
    temp.B[Kt > 0.6]=41.4 - 118.5*(Kt[Kt > 0.6]) + 66.05*(Kt[Kt > 0.6] ** 2) + 31.9*(Kt[Kt > 0.6] ** 3)
    temp.C[Kt > 0.6]=- 47.01 + 184.2*(Kt[Kt > 0.6]) - 222.0 * Kt[Kt > 0.6] ** 2 + 73.81*(Kt[Kt > 0.6] ** 3)
    temp.A[(Kt <= 0.6-1)]=0.512 - 1.56*(Kt[(Kt <= 0.6-1)]) + 2.286*(Kt[(Kt <= 0.6-1)] ** 2) - 2.222*(Kt[(Kt <= 0.6-1)] ** 3)
    temp.B[(Kt <= 0.6-1)]=0.37 + 0.962*(Kt[(Kt <= 0.6-1)])
    temp.C[(Kt <= 0.6-1)]=- 0.28 + 0.932*(Kt[(Kt <= 0.6-1)]) - 2.048*(Kt[(Kt <= 0.6-1)] ** 2)
    #return to numeric after masking operations 
    temp=temp.astype(float)
    delKn=temp.A + temp.B*((temp.C*(AM)).apply(np.exp))
    Knc=0.866 - 0.122*(AM) + 0.0121*(AM ** 2) - 0.000653*(AM ** 3) + 1.4e-05*(AM ** 4)
    Kn=Knc - delKn
    DNI=(Kn)*(I0)

    # DNI[var.SunZen > 87]=0
    # DNI[var.GHI < 1]=0
    # DNI[DNI < 0]=0

    DFOut=pd.DataFrame({'DNI_gen_DISC':DNI})

    DFOut['Kt_gen_DISC']=Kt
    DFOut['AM']=AM
    DFOut['Ztemp']=Ztemp
    
    return DFOut
