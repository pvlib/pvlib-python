
## Import TMY data

### Use Sandia standard data 

# In[4]:

import pvl_tools
import pvl_readtmy3 as tmy3
reload(tmy3)
fname='723650TY.csv'
fname='/Users/robandrews/Dropbox/My_Documents/Documents/Projects/Data/TMY/tmy3/723650TY.csv'
g='722026TY.csv'
TMY, meta=tmy3.pvl_readtmy3(FileName=fname)


# In[5]:

meta.SurfTilt=30
meta.SurfAz=0
meta.Albedo=0.2


## Get solar angles

# In[6]:

import pvl_ephemeris as eph
reload(eph)
import pvl_spa as spa
TMY['SunAz'],TMY['SunEl'],TMY['ApparentSunEl'],TMY['SolarTime'], TMY['SunZen']=eph.pvl_ephemeris(Time=TMY.index,Location=meta)
#TMY['SunAz'],TMY['SunEl']=spa.pvl_spa(Time=TMY.index,Location=meta)


## Calculate Extraterrestrial Irradiaion and AirMass

# In[8]:

import pvl_extraradiation as ext
reload(ext)
TMY['HExtra']=ext.pvl_extraradiation(doy=TMY.index.dayofyear)

import pvl_relativeairmass as AM
reload(AM)
TMY['AM']=AM.pvl_relativeairmass(z=TMY.SunZen)


## Generate Clear Sky and DNI

# In[9]:

import pvl_disc as disc
reload(disc)
DFOut=disc.pvl_disc(Time=TMY.index,GHI=TMY.GHI, SunZen=TMY.SunZen)

TMY['DNI_gen_DISC']=DFOut['DNI_gen_DISC']
TMY['Kt_gen_DISC']=DFOut['Kt_gen_DISC']
TMY['AM']=DFOut['AM']
TMY['Ztemp']=DFOut['Ztemp']


## Plane Transformation

# In[10]:

import pvl_perez as perez
reload(perez)
TMY['In_Plane_SkyDiffuse']=perez.pvl_perez(SurfTilt=meta.SurfTilt,
                                            SurfAz=meta.SurfAz,
                                            DHI=TMY.DHI,
                                            DNI=TMY.DNI,
                                            HExtra=TMY.HExtra,
                                            SunZen=TMY.SunZen,
                                            SunAz=TMY.SunAz,
                                            AM=TMY.AM)


## Ground Diffuse reflection

# In[11]:

import pvl_grounddiffuse as diff
reload(diff)
TMY['GR']=diff.pvl_grounddiffuse(GHI=TMY.GHI,Albedo=meta.Albedo,SurfTilt=meta.SurfTilt)


## Get AOI

# In[12]:

import pvl_getaoi as aoi
reload(aoi)
TMY['AOI']=aoi.pvl_getaoi(SunAz=TMY.SunAz,SunZen=TMY.SunZen,SurfTilt=meta.SurfTilt,SurfAz=meta.SurfAz)


## Calculate Global in-plane

# In[13]:

import pvl_globalinplane as globalirr
reload(globalirr)
TMY['E'],TMY['Eb'],TMY['EDiff']=globalirr.pvl_globalinplane(AOI=TMY.AOI,
                                DNI=TMY.DNI,
                                In_Plane_SkyDiffuse=TMY.In_Plane_SkyDiffuse,
                                GR=TMY.GR,
                                SurfTilt=meta.SurfTilt,
                                SurfAz=meta.SurfAz)



## Calculate Cell Temperature

# In[15]:

import pvl_sapmcelltemp as temp
reload(temp)
TMY['Tcell'],TMY['Tmodule']=temp.pvl_sapmcelltemp(E=TMY.E,
                            Wspd=TMY.Wspd,
                            Tamb=TMY.DryBulb)





## Import module coefficients

# In[16]:

import pvl_retreiveSAM as SAM
reload(SAM)
moddb=SAM.pvl_retreiveSAM(name='SandiaMod')
module=moddb.Solar_World_SW175_Mono_Sun_Module___2009_
module


# Out[16]:

#     Vintage                                                   2009
#     Area                                                     1.286
#     Material                                                  c-Si
#     #Series                                                     72
#     #Parallel                                                    1
#     Isco                                                     5.143
#     Voco                                                    44.623
#     Impo                                                     4.805
#     Vmpo                                                    36.071
#     Aisc                                                  0.000624
#     Aimp                                                  -2.7e-05
#     C0                                                      1.0016
#     C1                                                     -0.0016
#     Bvoco                                                  -0.1646
#     Mbvoc                                                        0
#     Bvmpo                                                  -0.1722
#     Mbvmp                                                        0
#     N                                                        1.274
#     C2                                                    0.236148
#     C3                                                    -4.62278
#     A0                                                      0.9543
#     A1                                                     0.03574
#     A2                                                   -0.003519
#     A3                                                  -5.592e-05
#     A4                                                   1.569e-05
#     B0                                                           1
#     B1                                                   -0.002438
#     B2                                                   0.0003103
#     B3                                                  -1.246e-05
#     B4                                                   2.112e-07
#     B5                                                  -1.359e-09
#     DTC                                                          3
#     FD                                                           1
#     A                                                       -3.319
#     B                                                     -0.09116
#     C4                                                      0.9908
#     C5                                                      0.0092
#     IXO                                                     5.1494
#     IXXO                                                    3.4661
#     C6                                                      1.1233
#     C7                                                     -0.1233
#     Notes        Source: Sandia National Laboratories Updated 9...
#     Name: Solar_World_SW175_Mono_Sun_Module___2009_, dtype: object

##  import inverter coefficients

# In[17]:

Invdb=SAM.pvl_retreiveSAM(name='SandiaInverter')
inverter=Invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
inverter


# Out[17]:

#     Vac           277.000000
#     Paco         6000.000000
#     Pdco         6165.670000
#     Vdco          361.123000
#     Pso            36.792300
#     C0             -0.000002
#     C1             -0.000047
#     C2             -0.001861
#     C3              0.000721
#     Pnt             0.070000
#     Vdcmax        600.000000
#     Idcmax         32.000000
#     Mppt_low      200.000000
#     Mppt_high     500.000000
#     Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64

# In[18]:

import pvl_retreiveSAM as SAM
reload(SAM)
Invdb=SAM.pvl_retreiveSAM(name='SandiaInverter')


## Sandia Model

# In[19]:

import pvl_sapm as sapm
reload(sapm)
DFOut=sapm.pvl_sapm(Eb=TMY['Eb'],
                    Ediff=TMY['EDiff'],
                    Tcell=TMY['Tcell'],
                    AM=TMY['AM'],
                    AOI=TMY['AOI'],
                    Module=module)

TMY['Imp']=DFOut['Imp']
TMY['Voc']=DFOut['Voc']
TMY['Vmp']=DFOut['Vmp']
TMY['Pmp']=DFOut['Pmp']
TMY['Ix']=DFOut['Ix']
TMY['Ixx']=DFOut['Ixx']


## Single Diode Model

# In[20]:

import time
moddb=SAM.pvl_retreiveSAM(name='CECMod')
module=moddb.Canadian_Solar_CS5P_220P
import pvl_calcparams_desoto as calc
reload(calc)
IL,I0,Rs,Rsh,nNsVth=calc.pvl_calcparams_desoto(S=TMY.GHI,
                                               Tcell=TMY.DryBulb,
                                               alpha_isc=.003,
                                               ModuleParameters=module,
                                               EgRef=1.121,
                                               dEgdT= -0.0002677)
import pvl_singlediode as sd
reload(sd)

DFout= sd.pvl_singlediode(Module=module,
                               IL=IL,
                               I0=I0,
                               Rs=Rs,
                               Rsh=Rsh,
                               nNsVth=nNsVth)


TMY['sd_Imp']=DFOut['Imp']
TMY['sd_Voc']=DFOut['Voc']
TMY['sd_Vmp']=DFOut['Vmp']
TMY['sd_Pmp']=DFOut['Pmp']
TMY['sd_Ix']=DFOut['Ix']
TMY['sd_Ixx']=DFOut['Ixx']           


## Inverter Model

# In[21]:

import pvl_snlinverter as invmod
reload(invmod)
TMY['ACPower']=invmod.pvl_snlinverter(Vmp=TMY.Vmp,Pmp=TMY.Pmp,Inverter=inverter)


