import os
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import  pvl_tools


def makelocationstruct(latitude,longitude,TZ,altitude=100,Name='Usr_input',State='Usr_input'):
  '''
  Create a struct to define a site location

  Parameters
  ----------

  Latitude : float
            Positive north of equator, decimal notation
  Longitude : float 
            Positive east of prime meridian, decimal notation
  TZ  : int
            Timezone in GMT offset

  Other Parameters
  ----------------

  altitude : float (optional, default=100)
            Altitude from sea level. Set to 100m if none input

  Returns
  -------

  Location : struct

          *Location.latitude*

          *Location.longitude*

          *Location.TZ*

          *Location.altitude*


  See Also
  --------

  pvl_ephemeris
  pvl_alt2pres
  pvl_pres2alt

  '''

  Vars=locals()
  Expect={'latitude':('num','x>=-90','x<=90'),
          'longitude': ('num','x<=180','x>=-180'),
          'altitude':('num','default','default=100'),
          'TZ':('num'),
          'Name':'',
          'State':''
          }
  Location=pvl_tools.Parse(Vars,Expect)


  return Location

def systemdef(TMYmeta,SurfTilt, SurfAz,Albedo,SeriesModules,ParallelModules):

    '''
    Generates a dict of system paramters used throughout a simulation

    Parameters
    ----------

    TMYmeta : struct or dict
                meta file generated from a TMY file using pvl_readtmy2 or pvl_readtmy3.
                It should contain at least the following fields: 

                    ===============   ======  ====================  
                    meta field        format  description
                    ===============   ======  ====================  
                    meta.altitude     Float   site elevation
                    meta.latitude     Float   site latitude
                    meta.longitude    Float   site longitude
                    meta.Name         String  site name
                    meta.State        String  state
                    meta.TZ           Float   timezone
                    ===============   ======  ====================  

    SurfTilt : float or DataFrame
              Surface tilt angles in decimal degrees.
              SurfTilt must be >=0 and <=180. The tilt angle is defined as
              degrees from horizontal (e.g. surface facing up = 0, surface facing
              horizon = 90)

    SurfAz : float or DataFrame
            Surface azimuth angles in decimal degrees.
            SurfAz must be >=0 and <=360. The Azimuth convention is defined
            as degrees east of north (e.g. North = 0, South=180 East = 90, West = 270).

    Albedo : float or DataFrame 
            Ground reflectance, typically 0.1-0.4 for
            surfaces on Earth (land), may increase over snow, ice, etc. May also 
            be known as the reflection coefficient. Must be >=0 and <=1.

    SeriesModules : float
            Number of modules connected in series in a string. 

    ParallelModules : int
            Number of strings connected in parallel.
    
    

    Returns
    -------

    Result : dict

                A dict with the following fields.
      
                    * 'SurfTilt'
                    * 'SurfAz'
                    * 'Albedo'
                    * 'SeriesModules'
                    * 'ParallelModules'
                    * 'Lat'
                    * 'Long'
                    * 'TZ'
                    * 'name'
                    * 'altitude'


    See also
    --------
    pvl_readtmy3
    pvl_readtmy2


    '''


    Vars=locals()
    Expect={'TMYmeta':'',
            'SurfTilt':('num','x>=0'),
            'SurfAz':('num'),
            'Albedo':('num','x>=0'),
            'SeriesModules':('default','default=1','num','x>=0'),
            'ParallelModules':('default','default=1','num','x>=0')}

    var=pvl_tools.Parse(Vars,Expect)

    system={'SurfTilt':var.SurfTilt,
            'SurfAz':var.SurfAz,
            'Albedo':var.Albedo,
            'SeriesModules':var.SeriesModules,
            'ParallelModules':var.ParallelModules,
            'latitude':var.TMYmeta.latitude,
            'longitude':var.TMYmeta.longitude,
            'TZ':var.TMYmeta.TZ,
            'name':var.TMYmeta.Name,
            'altitude':var.TMYmeta.altitude}

    return pvl_tools.repack(system)



def ashraeiam(b,theta):
    '''
    Determine the incidence angle modifier using the ASHRAE transmission model.


    pvl_ashraeiam calculates the incidence angle modifier as developed in
    [1], and adopted by ASHRAE (American Society of Heating, Refrigeration,
    and Air Conditioning Engineers) [2]. The model has been used by model
    programs such as PVSyst [3].

    Note: For incident angles near 90 degrees, this model has a
    discontinuity which has been addressed in this function.

    Parameters
    ----------
    b : float
            A parameter to adjust the modifier as a function of angle of
            incidence. Typical values are on the order of 0.05 [3].
    theta : DataFrame
            The angle of incidence between the module normal vector and the
            sun-beam vector in degrees. Theta must be a numeric scalar or vector.
            For any values of theta where abs(theta)>90, IAM is set to 0. For any
            values of theta where -90 < theta < 0, theta is set to abs(theta) and
            evaluated. A warning will be generated if any(theta<0 or theta>90).
            For values of theta near 90 degrees, the ASHRAE model may be above 1
            or less than 0 due to the discontinuity of secant(theta). IAM values
            outside of [0,1] are set to 0 and a warning is generated.

    Returns
    -------
    IAM : DataFrame

        The incident angle modifier calculated as 1-b*(sec(theta)-1) as
        described in [2,3]. IAM is a column vector with the same number of 
        elements as the largest input vector.

    References
    ----------

    [1] Souka A.F., Safwat H.H., "Determindation of the optimum orientations
    for the double exposure flat-plate collector and its reflections".
    Solar Energy vol .10, pp 170-174. 1966.

    [2] ASHRAE standard 93-77

    [3] PVsyst Contextual Help. 
    http://files.pvsyst.com/help/index.html?iam_loss.htm retrieved on
    September 10, 2012

    See Also
    --------

    pvl_getaoi
    pvl_ephemeris
    pvl_spa  
    pvl_physicaliam
    
    '''
    Vars=locals()
    Expect={'b':'x >= 0',
            'theta':'num'}
    var=pvl_tools.Parse(Vars,Expect)

    if any((var.theta < 0) | (var.theta >= 90)):
        print('Input incident angles <0 or >=90 detected For input angles with absolute value greater than 90, the ' + 'modifier is set to 0. For input angles between -90 and 0, the ' + 'angle is changed to its absolute value and evaluated.')
        var.theta[(var.theta < 0) | (var.theta >= 90)]=abs((var.theta < 0) | (var.theta >= 90))

    IAM=1 - var.b*((1/np.cos(np.radians(var.theta)) - 1))

    IAM[abs(var.theta) > 90]=0

    if any((IAM > 1) | (IAM < 0)):
        print('It seems that we have encountered a discontinuity. Any incident angle modifiers calculated to be less than 0 or ' + 'greather than 1 have been set to 0.')
    IAM[((IAM > 1) | (IAM < 0))]=0

    return IAM

def physicaliam(K,L,n,theta):

    '''
    Determine the incidence angle modifier using refractive 
    index, glazing thickness, and extinction coefficient

    pvl_physicaliam calculates the incidence angle modifier as described in
    De Soto et al. "Improvement and validation of a model for photovoltaic
    array performance", section 3. The calculation is based upon a physical
    model of absorbtion and transmission through a cover. Required
    information includes, incident angle, cover extinction coefficient,
    cover thickness

    Note: The authors of this function believe that eqn. 14 in [1] is
    incorrect. This function uses the following equation in its place:
    theta_r = arcsin(1/n * sin(theta))

    Parameters
    ----------

    K : float

            The glazing extinction coefficient in units of 1/meters. Reference
            [1] indicates that a value of  4 is reasonable for "water white"
            glass. K must be a numeric scalar or vector with all values >=0. If K
            is a vector, it must be the same size as all other input vectors.

    L : float

            The glazing thickness in units of meters. Reference [1] indicates
            that 0.002 meters (2 mm) is reasonable for most glass-covered
            PV panels. L must be a numeric scalar or vector with all values >=0. 
            If L is a vector, it must be the same size as all other input vectors.

    n : float

            The effective index of refraction (unitless). Reference [1]
            indicates that a value of 1.526 is acceptable for glass. n must be a 
            numeric scalar or vector with all values >=0. If n is a vector, it 
            must be the same size as all other input vectors.

    theta :float

            The angle of incidence between the module normal vector and the
            sun-beam vector in degrees. Theta must be a numeric scalar or vector.
            For any values of theta where abs(theta)>90, IAM is set to 0. For any
            values of theta where -90 < theta < 0, theta is set to abs(theta) and
            evaluated. A warning will be generated if any(theta<0 or theta>90).

    Returns
    -------

    IAM : float

       The incident angle modifier as specified in eqns. 14-16 of [1].
         IAM is a column vector with the same number of elements as the
         largest input vector.

    References
    ----------

    [1] W. De Soto et al., "Improvement and validation of a model for
     photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
     2006.

    [2] Duffie, John A. & Beckman, William A.. (2006). Solar Engineering 
     of Thermal Processes, third edition. [Books24x7 version] Available 
     from http://common.books24x7.com/toc.aspx?bookid=17160. 

    See Also 
    --------
          
    pvl_getaoi   
    pvl_ephemeris   
    pvl_spa    
    pvl_ashraeiam

    '''
    Vars=locals()

    Expect={'K':'x >= 0',
            'L':'x >= 0',
            'n':'x >= 0',
            'theta':'num'}
    var=pvl_tools.Parse(Vars,Expect)



    if any((var.theta < 0) | (var.theta >= 90)):
        print('Input incident angles <0 or >=90 detected For input angles with absolute value greater than 90, the ' + 'modifier is set to 0. For input angles between -90 and 0, the ' + 'angle is changed to its absolute value and evaluated.')
        var.theta[(var.theta < 0) | (var.theta >= 90)]=abs((var.theta < 0) | (var.theta >= 90))

    thetar_deg=pvl_tools.asind(1.0 / n*(pvl_tools.sind(theta)))

    tau=np.exp(- 1.0 * (K*(L) / pvl_tools.cosd(thetar_deg)))*((1 - 0.5*((((pvl_tools.sind(thetar_deg - theta)) ** 2) / ((pvl_tools.sind(thetar_deg + theta)) ** 2) + ((pvl_tools.tand(thetar_deg - theta)) ** 2) / ((pvl_tools.tand(thetar_deg + theta)) ** 2)))))
    
    zeroang=1e-06
    
    thetar_deg0=pvl_tools.asind(1.0 / n*(pvl_tools.sind(zeroang)))
    
    tau0=np.exp(- 1.0 * (K*(L) / pvl_tools.cosd(thetar_deg0)))*((1 - 0.5*((((pvl_tools.sind(thetar_deg0 - zeroang)) ** 2) / ((pvl_tools.sind(thetar_deg0 + zeroang)) ** 2) + ((pvl_tools.tand(thetar_deg0 - zeroang)) ** 2) / ((pvl_tools.tand(thetar_deg0 + zeroang)) ** 2)))))
    
    IAM=tau / tau0
    
    IAM[theta == 0]=1
    
    IAM[abs(theta) > 90 | (IAM < 0)]=0
    
    return IAM




def calcparams_desoto(S, Tcell, alpha_isc, ModuleParameters, EgRef, dEgdT,
                          M=1, Sref=1000, Tref=25):
    '''
    Applies the temperature and irradiance corrections to inputs for pvl_singlediode

    Applies the temperature and irradiance corrections to the IL, I0,
    Rs, Rsh, and a parameters at reference conditions (IL_ref, I0_ref,
    etc.) according to the De Soto et. al description given in [1]. The
    results of this correction procedure may be used in a single diode
    model to determine IV curves at irradiance = S, cell temperature =
    Tcell.

    Parameters
    ----------
    S : float or DataFrame
          The irradiance (in W/m^2) absorbed by the module. S must be >= 0.
          Due to a division by S in the script, any S value equal to 0 will be set to 1E-10.

    Tcell : float or DataFrame
          The average cell temperature of cells within a module in C.
          Tcell must be >= -273.15.

    alpha_isc : float

          The short-circuit current temperature coefficient of the module in units of 1/C.

    ModuleParameters : struct
          parameters describing PV module performance at reference conditions according
          to DeSoto's paper. Parameters may be generated or found by lookup. For ease of use,
          PVL_RETREIVESAM can automatically generate a struct based on the most recent SAM CEC module
          database. The ModuleParameters struct must contain (at least) the
          following 5 fields:

              *ModuleParameters.a_ref* - modified diode ideality factor parameter at
                  reference conditions (units of eV), a_ref can be calculated from the
                  usual diode ideality factor (n), number of cells in series (Ns),
                  and cell temperature (Tcell) per equation (2) in [1].

              *ModuleParameters.IL_ref* - Light-generated current (or photocurrent)
                  in amperes at reference conditions. This value is referred to
                  as Iph in some literature.

              *ModuleParameters.I0_ref* - diode reverse saturation current in amperes,
                  under reference conditions.

              *ModuleParameters.Rsh_ref* - shunt resistance under reference conditions (ohms)

              *ModuleParameters.Rs_ref* - series resistance under reference conditions (ohms)

    EgRef : float

          The energy bandgap at reference temperature (in eV). 1.121 eV for silicon. EgRef must be >0.

    dEgdT : float

          The temperature dependence of the energy bandgap at SRC (in 1/C).
          May be either a scalar value (e.g. -0.0002677 as in [1]) or a
          DataFrame of dEgdT values corresponding to each input condition (this
          may be useful if dEgdT is a function of temperature).

    Other Parameters
    ----------------

    M : float or DataFrame (optional, Default=1)
          An optional airmass modifier, if omitted, M is given a value of 1,
          which assumes absolute (pressure corrected) airmass = 1.5. In this
          code, M is equal to M/Mref as described in [1] (i.e. Mref is assumed
          to be 1). Source [1] suggests that an appropriate value for M
          as a function absolute airmass (AMa) may be:

          >>> M = np.polyval([-0.000126, 0.002816, -0.024459, 0.086257, 0.918093], AMa)

          M may be a DataFrame.

    Sref : float (optional, Default=1000)

          Optional reference irradiance in W/m^2. If omitted, a value of
          1000 is used.

    Tref : float (Optional, Default=25)
          Optional reference cell temperature in C. If omitted, a value of
          25 C is used.

    Returns
    -------

    IL : float or DataFrame
          Light-generated current in amperes at irradiance=S and
          cell temperature=Tcell.
    I0 : float or DataFrame
          Diode saturation curent in amperes at irradiance S and cell temperature Tcell.

    Rs : float
          Series resistance in ohms at irradiance S and cell temperature Tcell.

    Rsh : float or DataFrame
          Shunt resistance in ohms at irradiance S and cell temperature Tcell.

    nNsVth : float or DataFrame
          Modified diode ideality factor at irradiance S and cell temperature
          Tcell. Note that in source [1] nNsVth = a (equation 2). nNsVth is the
          product of the usual diode ideality factor (n), the number of
          series-connected cells in the module (Ns), and the thermal voltage
          of a cell in the module (Vth) at a cell temperature of Tcell.



    References
    ----------

    [1] W. De Soto et al., "Improvement and validation of a model for
       photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
       2006.

    [2] System Advisor Model web page. https://sam.nrel.gov.

    [3] A. Dobos, "An Improved Coefficient Calculator for the California
       Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
       Solar Energy Engineering, vol 134, 2012.

    [4] O. Madelung, "Semiconductors: Data Handbook, 3rd ed." ISBN
       3-540-40488-0

    See Also
    --------
    pvl_sapm
    pvl_sapmcelltemp
    pvl_singlediode
    pvl_retreivesam


    Notes
    -----

    If the reference parameters in the ModuleParameters struct are read
    from a database or library of parameters (e.g. System Advisor Model),
    it is important to use the same EgRef and dEgdT values that
    were used to generate the reference parameters, regardless of the
    actual bandgap characteristics of the semiconductor. For example, in
    the case of the System Advisor Model library, created as described in
    [3], EgRef and dEgdT for all modules were 1.121 and -0.0002677,
    respectively.

    This table of reference bandgap energies (EgRef), bandgap energy
    temperature dependence (dEgdT), and "typical" airmass response (M) is
    provided purely as reference to those who may generate their own
    reference module parameters (a_ref, IL_ref, I0_ref, etc.) based upon the
    various PV semiconductors. Again, we stress the importance of
    using identical EgRef and dEgdT when generation reference
    parameters and modifying the reference parameters (for irradiance,
    temperature, and airmass) per DeSoto's equations.

     Silicon (Si):
         EgRef = 1.121
         dEgdT = -0.0002677

         >>> M = polyval([-0.000126 0.002816 -0.024459 0.086257 0.918093], AMa)

         Source = Reference 1

     Cadmium Telluride (CdTe):
         EgRef = 1.475
         dEgdT = -0.0003

         >>> M = polyval([-2.46E-5 9.607E-4 -0.0134 0.0716 0.9196], AMa)

         Source = Reference 4

     Copper Indium diSelenide (CIS):
         EgRef = 1.010
         dEgdT = -0.00011

         >>> M = polyval([-3.74E-5 0.00125 -0.01462 0.0718 0.9210], AMa)

         Source = Reference 4

     Copper Indium Gallium diSelenide (CIGS):
         EgRef = 1.15
         dEgdT = ????

         >>> M = polyval([-9.07E-5 0.0022 -0.0202 0.0652 0.9417], AMa)

         Source = Wikipedia

     Gallium Arsenide (GaAs):

         EgRef = 1.424
         dEgdT = -0.000433
         M = unknown
         Source = Reference 4
    '''

    Vars=locals()

    Expect={'S':('x >= 0') ,
            'Tcell':('x >= - 273.15') ,
            'alpha_isc': (''),
            'ModuleParameters': (''),
            'EgRef': ('x > 0'),
            'dEgdT': (''),
            'M': ('num','default','default=1'),
            'Sref':('default','default=1000'),
            'Tref':('default','default=25')
            }

    var=pvl_tools.Parse(Vars,Expect)

    var.M=np.max(var.M,0)
    a_ref=var.ModuleParameters.A_ref
    IL_ref=var.ModuleParameters.I_l_ref
    I0_ref=var.ModuleParameters.I_o_ref
    Rsh_ref=var.ModuleParameters.R_sh_ref
    Rs_ref=var.ModuleParameters.R_s


    k=8.617332478e-05
    Tref_K=var.Tref + 273.15
    Tcell_K=var.Tcell + 273.15

    var.S[var.S == 0]=1e-10
    E_g=var.EgRef * ((1 + var.dEgdT*((Tcell_K - Tref_K))))

    nNsVth=a_ref*((Tcell_K / Tref_K))

    IL=var.S / var.Sref *(var.M) *((IL_ref + var.alpha_isc * ((Tcell_K - Tref_K))))
    I0=I0_ref * (((Tcell_K / Tref_K) ** 3)) * (np.exp((var.EgRef / (k*(Tref_K))) - (E_g / (k*(Tcell_K)))))
    Rsh=Rsh_ref * ((var.Sref / var.S))
    Rs=Rs_ref

    return IL,I0,Rs,Rsh,nNsVth

def getaoi(SurfTilt,SurfAz,SunZen,SunAz):
  '''
  Determine angle of incidence from surface tilt/azimuth and apparent sun zenith/azimuth 

  The surface is defined by its tilt angle from horizontal and its azimuth pointing angle. 
  The sun position is defined by the apparent (refraction corrected)sun zenith angle and the sun 
  azimuth angle.

  Parameters
  ----------
  SurfTilt :  scalar or DataFrame of surface tilt angles in decimal degrees

               If SurfTilt is a DataFrame it must be of the same size as all other DataFrame
               inputs. SurfTilt must be >=0 and <=180. The tilt angle is defined as
               degrees from horizontal (e.g. surface facing up = 0, surface facing
               horizon = 90)

  SurfAz :  scalar or DataFrame of the surface azimuth angles in decimal degrees

               If SurfAz is a DataFrame it must be of the same size as all other DataFrame
               inputs. SurfAz must be >=0 and <=360. The Azimuth convention is defined
               as degrees east of north (e.g. North = 0, East = 90, West = 270).

  SunZen : scalar or DataFrame of apparent (refraction-corrected) zenith angles in decimal degrees. 

               If SunZen is a DataFrame it must be of the same size as all other DataFrame 
               inputs. SunZen must be >=0 and <=180.

  SunAz : scalar or DataFrame of sun azimuth angles in decimal degrees

               If SunAz is a DataFrame it must be of the same size as all other DataFrame
               inputs. SunAz must be >=0 and <=360. The Azimuth convention is defined
               as degrees east of north (e.g. North = 0, East = 90, West = 270).

  Returns
  -------
  AOI : DataFrame
      The angle, in decimal degrees, between the surface normal DataFrame and the sun beam DataFrame. 

  References
  ----------
  D.L. King, J.A. Kratochvil, W.E. Boyson. "Spectral and
  Angle-of-Incidence Effects on Photovoltaic Modules and Solar Irradiance
  Sensors". 26th IEEE Photovoltaic Specialists Conference. Sept. 1997.

  See Also
  --------
  PVL_EPHEMERIS
  '''

  Vars=locals()
  Expect={'SurfTilt':('num','x>=0'),
        'SurfAz':('num','x>=-180','x<=180'),
        'SunZen':('x>=0'),
        'SunAz':('x>=0')
  }

  var=pvl_tools.Parse(Vars,Expect)

  AOI=np.degrees(np.arccos(np.cos(np.radians(var.SunZen))*(np.cos(np.radians(var.SurfTilt))) + np.sin(np.radians(var.SurfTilt))*(np.sin(np.radians(var.SunZen)))*(np.cos(np.radians(var.SunAz) - np.radians(var.SurfAz))))) #Duffie and Beckmann 1.6.3


  return pd.DataFrame({'AOI':AOI})

def retreiveSAM(name,FileLoc='none'):
    '''
    Retreive lastest module and inverter info from SAM website

    PVL_RETREIVESAM Retreive lastest module and inverter info from SAM website.
    This function will retreive either:

        * CEC module database
        * Sandia Module database
        * Sandia Inverter database

    and export it as a pandas dataframe


    Parameters
    ----------

    name: String
                Name can be one of:

                * 'CECMod'- returns the CEC module database
                * 'SandiaInverter- returns the Sandia Inverter database
                * 'SandiaMod'- returns the Sandia Module database
    FileLoc: String

                Absolute path to the location of local versions of the SAM file. 
                If FileLoc is specified, the latest versions of the SAM database will
                not be downloaded. The selected file must be in .csv format. 

                If set to 'select', a dialogue will open allowing the suer to navigate 
                to the appropriate page. 
    Returns
    -------

    df: DataFrame

                A DataFrame containing all the elements of the desired database. 
                Each column representa a module or inverter, and a specific dataset
                can be retreived by the command

                >>> df.module_or_inverter_name

    Examples
    --------

    >>> Invdb=SAM.pvl_retreiveSAM(name='SandiaInverter')
    >>> inverter=Invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
    >>> inverter    
    Vac           277.000000
    Paco         6000.000000
    Pdco         6165.670000
    Vdco          361.123000
    Pso            36.792300
    C0             -0.000002
    C1             -0.000047
    C2             -0.001861
    C3              0.000721
    Pnt             0.070000
    Vdcmax        600.000000
    Idcmax         32.000000
    Mppt_low      200.000000
    Mppt_high     500.000000
    Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64
    
    '''
    Vars=locals()
    Expect={'name':('str',('CECMod','SandiaMod','SandiaInverter')),
            'FileLoc':('optional')}

    var=pvl_tools.Parse(Vars,Expect)


    if var.name=='CECMod':
        url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-cec-modules-2014-1-14.csv'
    elif var.name=='SandiaMod':
        url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-modules-2014-1-14.csv'
    elif var.name=='SandiaInverter':
        url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-inverters-2014-1-14.csv'
    
    if FileLoc=='none':
        return read_url_to_pandas(url)
    elif FileLoc=='select':
        try:
            import Tkinter 
            from tkFileDialog import askopenfilename
            Tkinter.Tk().withdraw()                  #Start interactive file input
            return read_relative_to_pandas(askopenfilename())                               
        except:
            raise Exception ('Python not configured for TKinter. Try installing XQuartz and rerunning')
    else: 
        return read_relative_to_pandas(FileLoc)
        
def read_url_to_pandas(url):

    data = urlopen(url)
    df=pd.read_csv(data,index_col=0)
    parsedindex=[]
    for index in df.index:
        parsedindex.append(index.replace(' ','_').replace('-','_').replace('.','_').replace('(','_').replace(')','_').replace('[','_').replace(']','_').replace(':','_'))
        
    df.index=parsedindex
    df=df.transpose()
    return df

def read_relative_to_pandas(FileLoc):

    df=pd.read_csv(FileLoc,index_col=0)
    parsedindex=[]
    for index in df.index:
        parsedindex.append(index.replace(' ','_').replace('-','_').replace('.','_').replace('(','_').replace(')','_').replace('[','_').replace(']','_').replace(':','_'))
        
    df.index=parsedindex
    df=df.transpose()
    return df

def sapm(Module,Eb,Ediff,Tcell,AM,AOI):
    '''
    Performs Sandia PV Array Performance Model to get 5 points on IV curve given SAPM module parameters, Ee, and cell temperature

    The Sandia PV Array Performance Model (SAPM) generates 5 points on a PV
    module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
    SAND2004-3535. Assumes a reference cell temperature of 25 C.

    parameters
    ----------

    Module : DataFrame

            A DataFrame defining the SAPM performance parameters (see
            pvl_retreivesam)

    Eb : float of DataFrame

            The effective irradiance incident upon the module (suns). Any Ee<0
            are set to 0.

    celltemp : float of DataFrame

            The cell temperature (degrees C)

    Returns
    -------
    Result - DataFrame

            A DataFrame with:

            * Result.Isc
            * Result.Imp
            * Result.Ix
            * Result.Ixx
            * Result.Voc
            * Result.Vmp
            * Result.Pmp

    Notes
    -----

    The particular coefficients from SAPM which are required in Module
    are:

    ================   ======================================================================================================================
    Module field        Description
    ================   ======================================================================================================================
    Module.c           1x8 vector with the C coefficients Module.c(1) = C0
    Module.Isc0        Short circuit current at reference condition (amps)
    Module.Imp0        Maximum power current at reference condition (amps)
    Module.AlphaIsc    Short circuit current temperature coefficient at reference condition (1/C)
    Module.AlphaImp    Maximum power current temperature coefficient at reference condition (1/C)
    Module.BetaVoc     Open circuit voltage temperature coefficient at reference condition (V/C)
    Module.mBetaVoc    Coefficient providing the irradiance dependence for the BetaVoc temperature coefficient at reference irradiance (V/C)
    Module.BetaVmp     Maximum power voltage temperature coefficient at reference condition
    Module.mBetaVmp    Coefficient providing the irradiance dependence for the BetaVmp temperature coefficient at reference irradiance (V/C)
    Module.n           Empirically determined "diode factor" (dimensionless)
    Module.Ns          Number of cells in series in a module's cell string(s)
    ================   ======================================================================================================================

    References
    ----------

    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND Report
    3535, Sandia National Laboratories, Albuquerque, NM

    See Also
    --------

    pvl_retreivesam
    pvl_sapmcelltemp 

    '''
    Vars=locals()
    Expect={'Module':(''),
            'Eb':('x>0'),
            'Ediff':('x>0'),
            'Tcell':('x>0'),
            'AM':('x>0'),
            'AOI':('x>0')
            }
    var=pvl_tools.Parse(Vars,Expect)

    T0=25
    q=1.60218e-19
    k=1.38066e-23
    E0=1000

    AMcoeff=[var.Module['A4'],var.Module['A3'],var.Module['A2'],var.Module['A1'],var.Module['A0']]
    AOIcoeff=[var.Module['B5'],var.Module['B4'],var.Module['B3'],var.Module['B2'],var.Module['B1'],var.Module['B0']]

    F1 = np.polyval(AMcoeff,var.AM)
    F2 = np.polyval(AOIcoeff,var.AOI)
    var.Ee= F1*((var.Eb*F2+var.Module['FD']*var.Ediff)/E0)
    #var['Ee']=F1*((var.Eb+var.Ediff)/E0)
    #print "Ee modifed, revert for main function"
    var.Ee.fillna(0)
    var.Ee[var.Ee < 0]=0

    Filt=var.Ee[var.Ee >= 0.001]

    Isc=var.Module.ix['Isco']*(var.Ee)*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))

    DFOut=pd.DataFrame({'Isc':Isc})

    DFOut['Imp']=var.Module.ix['Impo']*((var.Module.ix['C0']*(var.Ee) + var.Module.ix['C1'] * (var.Ee ** 2)))*((1 + var.Module.ix['Aimp']*((var.Tcell - T0))))
    Bvoco=var.Module.ix['Bvoco'] + var.Module.ix['Mbvoc']*((1 - var.Ee))
    delta=var.Module.ix['N']*(k)*((var.Tcell + 273.15)) / q
    DFOut['Voc']=(var.Module.ix['Voco'] + var.Module.ix['#Series']*(delta)*(np.log(var.Ee)) + Bvoco*((var.Tcell - T0)))
    Bvmpo=var.Module.ix['Bvmpo'] + var.Module.ix['Mbvmp']*((1 - var.Ee))
    DFOut['Vmp']=(var.Module.ix['Vmpo'] + var.Module.ix['C2']*(var.Module.ix['#Series'])*(delta)*(np.log(var.Ee)) + var.Module.ix['C3']*(var.Module.ix['#Series'])*((delta*(np.log(var.Ee))) ** 2) + Bvmpo*((var.Tcell - T0)))
    DFOut['Vmp'][DFOut['Vmp']<0]=0
    DFOut['Pmp']=DFOut.Imp*DFOut.Vmp
    DFOut['Ix']=var.Module.ix['IXO'] * (var.Module.ix['C4']*(var.Ee) + var.Module.ix['C5']*((var.Ee) ** 2))*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))
    DFOut['Ixx']=var.Module.ix['IXXO'] * (var.Module.ix['C6']*(var.Ee) + var.Module.ix['C7']*((var.Ee) ** 2))*((1 + var.Module.ix['Aisc']*((var.Tcell - T0))))

    return  DFOut


def sapmcelltemp(E, Wspd, Tamb,modelt='Open_rack_cell_glassback',**kwargs):
    '''
    Estimate cell temperature from irradiance, windspeed, ambient temperature, and module parameters (SAPM)

    Estimate cell and module temperatures per the Sandia PV Array
    Performance model (SAPM, SAND2004-3535), when given the incident
    irradiance, wind speed, ambient temperature, and SAPM module
    parameters.

    Parameters
    ----------

    E : float or DataFrame
            Total incident irradiance in W/m^2. Must be >=0.


    windspeed : float or DataFrame
            Wind speed in m/s at a height of 10 meters. Must be >=0

    Tamb : float or DataFrame
            Ambient dry bulb temperature in degrees C. Must be >= -273.15.


    Other Parameters
    ----------------

    modelt :  string

    Model to be used for parameters, can be:

            * 'Open_rack_cell_glassback' (DEFAULT)
            * 'Roof_mount_cell_glassback'
            * 'Open_rack_cell_polymerback'
            * 'Insulated_back_polumerback'
            * 'Open_rack_Polymer_thinfilm_steel'
            * '22X_Concentrator_tracker'

    a : float (optional)
            SAPM module parameter for establishing the upper limit for module 
            temperature at low wind speeds and high solar irradiance (see SAPM
            eqn. 11). Must be a scalar.If not input, this value will be taken from the chosen
            model
    b : float (optional)

            SAPM module parameter for establishing the rate at which the module
            temperature drops as wind speed increases (see SAPM eqn. 11). Must be
            a scalar.If not input, this value will be taken from the chosen
            model

    deltaT : float (optional) 

            SAPM module parameter giving the temperature difference
            between the cell and module back surface at the reference irradiance,
            E0. Must be a numeric scalar >=0. If not input, this value will be taken from the chosen
            model

    Returns
    --------
    Tcell : float or DataFrame
            Cell temperatures in degrees C.

    Tmodule : float or DataFrame
            Module back temperature in degrees C.

    References
    ----------

    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND Report
    3535, Sandia National Laboratories, Albuquerque, NM

    See Also 
    --------

    pvl_sapm
    '''
    Vars=locals()
    Expect={'a':('optional','num'),
            'b':('optional','num'),
            'deltaT':('optional','num'), 
            'E':('x>=0'),
            'Wspd':('x>=0'),
            'Tamb':('x>=0'),
            'modelt': ('default','default=Open_rack_cell_glassback')
            }

    var=pvl_tools.Parse(Vars,Expect)

    TempModel={'Open_rack_cell_glassback':[-3.47, -.0594, 3],
                'Roof_mount_cell_glassback':[-2.98, -.0471, 1],
                'Open_rack_cell_polymerback': [-3.56, -.0750, 3],
                'Insulated_back_polumerback': [-2.81, -.0455, 0 ],
                'Open_rack_Polymer_thinfilm_steel':[-3.58, -.113, 3],
                '22X_Concentrator_tracker':[-3.23, -.130, 13]
            }
    try: 
        a=var.a
        b=var.b
        deltaT=var.deltaT
    except:
        a=TempModel[var.modelt][0]
        b=TempModel[var.modelt][1]
        deltaT=TempModel[var.modelt][2]

    E0=1000 # Reference irradiance
    Tmodule=var.E*((np.exp(a + b*var.Wspd))) + var.Tamb

    Tcell=Tmodule + var.E / E0*(deltaT)

    return pd.DataFrame({'Tcell':Tcell,'Tmodule':Tmodule})

def singlediode(Module,IL,I0,Rs,Rsh,nNsVth,**kwargs):
    '''
    Solve the single-diode model to obtain a photovoltaic IV curve


    pvl_singlediode solves the single diode equation [1]:
    I = IL - I0*[exp((V+I*Rs)/(nNsVth))-1] - (V + I*Rs)/Rsh
    for I and V when given IL, I0, Rs, Rsh, and nNsVth (nNsVth = n*Ns*Vth) which
    are described later. pvl_singlediode returns a struct which contains
    the 5 points on the I-V curve specified in SAND2004-3535 [3]. 
    If all IL, I0, Rs, Rsh, and nNsVth are scalar, a single curve
    will be returned, if any are DataFrames (of the same length), multiple IV
    curves will be calculated.

    Parameters
    ----------

    These imput parameters can be calculated using PVL_CALCPARAMS_DESOTO from 
    meterological data. 

    IL : float or DataFrame
                Light-generated current (photocurrent) in amperes under desired IV
                curve conditions. 

    I0 : float or DataFrame
                Diode saturation current in amperes under desired IV curve
                conditions. 

    Rs : float or DataFrame
                Series resistance in ohms under desired IV curve conditions. 

    Rsh : float or DataFrame
                Shunt resistance in ohms under desired IV curve conditions. May
                be a scalar or DataFrame, but DataFrames must be of same length as all
                other input DataFrames.

    nNsVth : float or DataFrame
                the product of three components. 1) The usual diode ideal 
                factor (n), 2) the number of cells in series (Ns), and 3) the cell 
                thermal voltage under the desired IV curve conditions (Vth).
                The thermal voltage of the cell (in volts) may be calculated as 
                k*Tcell/q, where k is Boltzmann's constant (J/K), Tcell is the
                temperature of the p-n junction in Kelvin, and q is the elementary 
                charge of an electron (coulombs). 
    Other Parameters
    ----------------

    NumPoints : integer

                Number of points in the desired IV curve (optional). Must be a finite 
                scalar value. Non-integer values will be rounded to the next highest
                integer (ceil). If ceil(NumPoints) is < 2, no IV curves will be produced
                (i.e. Result.V and Result.I will not be generated). The default
                value is 0, resulting in no calculation of IV points other than
                those specified in [3].

    Returns

    Result : DataFrame

                A DataFrame with the following fields. All fields have the
                same number of rows as the largest input DataFrame:
                
                * Result.Isc -  short circuit current in amperes.
                * Result.Voc -  open circuit voltage in volts.
                * Result.Imp -  current at maximum power point in amperes. 
                * Result.Vmp -  voltage at maximum power point in volts.
                * Result.Pmp -  power at maximum power point in watts.
                * Result.Ix -  current, in amperes, at V = 0.5*Voc.
                * Result.Ixx -  current, in amperes, at V = 0.5*(Voc+Vmp).


    Notes
    -----

    The solution employed to solve the implicit diode equation utilizes
    the Lambert W function to obtain an explicit function of V=f(i) and
    I=f(V) as shown in [2].

    References
    -----------

    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" 
    ISBN 0 86758 909 4

    [2] A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of 
    real solar cells using Lambert W-function", Solar Energy Materials 
    and Solar Cells, 81 (2004) 269-277.

    [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
    SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    See also
    --------
    pvl_sapm
    pvl_calcparams_desoto


    '''
    Vars=locals()
    Expect={'Module':(''),
            'IL':('x>0'),
            'I0':('x>0'),
            'Rs':('x>0'),
            'Rsh':('x>0'),
            'nNsVth':('x>0'),
    }

    var=pvl_tools.Parse(Vars,Expect)

    # Find Isc using Lambert W
    Isc = I_from_V(Rsh=var.Rsh, Rs=var.Rs, nNsVth=var.nNsVth, V=0.01, I0=var.I0, IL=var.IL)


    #If passed a dataframe, output a dataframe, if passed a list or scalar,
    #return a dict 
    if isinstance(var.Rsh,pd.Series):
        DFOut=pd.DataFrame({'Isc':Isc})
        DFOut.index=var.Rsh.index
    else:
        DFOut={'Isc':Isc}


    DFOut['Rsh']=var.Rsh
    DFOut['Rs']=var.Rs
    DFOut['nNsVth']=var.nNsVth
    DFOut['I0']=var.I0
    DFOut['IL']=var.IL

    __,Voc_return = golden_sect_DataFrame(DFOut,0,var.Module.V_oc_ref*1.6,Voc_optfcn)
    Voc=Voc_return.copy() #create an immutable copy 

    Pmp,Vmax = golden_sect_DataFrame(DFOut,0,var.Module.V_oc_ref*1.14,pwr_optfcn)
    Imax = I_from_V(Rsh=var.Rsh, Rs=var.Rs, nNsVth=var.nNsVth, V=Vmax, I0=var.I0, IL=var.IL)
    # Invert the Power-Current curve. Find the current where the inverted power
    # is minimized. This is Imax. Start the optimization at Voc/2

    # Find Ix and Ixx using Lambert W
    Ix = I_from_V(Rsh=var.Rsh, Rs=var.Rs, nNsVth=var.nNsVth, V=.5*Voc, I0=var.I0, IL=var.IL)
    Ixx = I_from_V(Rsh=var.Rsh, Rs=var.Rs, nNsVth=var.nNsVth, V=0.5*(Voc+Vmax), I0=var.I0, IL=var.IL)

    '''
    # If the user says they want a curve of with number of points equal to
    # NumPoints (must be >=2), then create a voltage array where voltage is
    # zero in the first column, and Voc in the last column. Number of columns
    # must equal NumPoints. Each row represents the voltage for one IV curve.
    # Then create a current array where current is Isc in the first column, and
    # zero in the last column, and each row represents the current in one IV
    # curve. Thus the nth (V,I) point of curve m would be found as follows:
    # (Result.V(m,n),Result.I(m,n)).
    if NumPoints >= 2
       s = ones(1,NumPoints); # shaping DataFrame to shape the column DataFrame parameters into 2-D matrices
       Result.V = (Voc)*(0:1/(NumPoints-1):1);
       Result.I = I_from_V(Rsh*s, Rs*s, nNsVth*s, Result.V, I0*s, IL*s);
    end
    '''

    DFOut['Imp']=Imax
    DFOut['Voc']=Voc
    DFOut['Vmp']=Vmax
    DFOut['Pmp']=Pmp
    DFOut['Ix']=Ix
    DFOut['Ixx']=Ixx

    return  DFOut



'''
Created April,2014
Author: Rob Andrews, Calama Consulting
'''

def golden_sect_DataFrame(df,VL,VH,func):
    '''
    Vectorized golden section search for finding MPPT from a dataframe timeseries

    Parameters
    ----------

    df : DataFrame

            Dataframe containing a timeseries of inputs to the function to be optimized.
            Each row should represent an independant optimization

    VL: float
            low bound of the optimization

    VH: float
            Uppoer bound of the optimization

    func: function
            function to be optimized must be in the form f(dataframe,x)

    Returns
    -------
    func(df,'V1') : DataFrame
            function evaluated at the optimal point

    df['V1']: Dataframe
            Dataframe of optimal points

    Notes
    -----

    This funtion will find the MAXIMUM of a function

    '''

    df['VH']=VH
    df['VL']=VL
      
    err=df['VH']-df['VL']
    errflag=True
    iterations=0
    while errflag:

        phi=(np.sqrt(5)-1)/2*(df['VH']-df['VL'])
        df['V1']=df['VL']+phi
        df['V2']=df['VH']-phi
        
        df['f1']=func(df,'V1')
        df['f2']=func(df,'V2')
        df['SW_Flag']=df['f1']>df['f2']
        
        df['VL']=df['V2']*df['SW_Flag']+df['VL']*(~df['SW_Flag'])
        df['VH']=df['V1']*~df['SW_Flag']+df['VH']*(df['SW_Flag'])
        
        err=(df['V1']-df['V2'])
        if isinstance(df,pd.DataFrame):
            errflag=all(abs(err)>.01)
        else:
            errflag=(abs(err)>.01)

        iterations=iterations+1

        if iterations >50:
            raise Exception("EXCEPTION:iterations exeeded maximum (50)")


    return func(df,'V1') , df['V1']

def pwr_optfcn(df,loc):
    '''
    function to find power from I_from_V
    '''

    I=I_from_V(Rsh=df['Rsh'],Rs=df['Rs'], nNsVth=df['nNsVth'], V=df[loc], I0=df['I0'], IL=df['IL'])
    return I*df[loc]

def Voc_optfcn(df,loc):
    '''
    function to find V_oc from I_from_V
    '''
    I=-abs(I_from_V(Rsh=df['Rsh'], Rs=df['Rs'], nNsVth=df['nNsVth'], V=df[loc], I0=df['I0'], IL=df['IL']))
    return I


def I_from_V(Rsh, Rs, nNsVth, V, I0, IL):
    '''
    # calculates I from V per Eq 2 Jain and Kapoor 2004
    # uses Lambert W implemented in wapr_vec.m
    # Rsh, nVth, V, I0, IL can all be DataFrames
    # Rs can be a DataFrame, but should be a scalar
    '''
    try:
        from scipy.special import lambertw
    except ImportError:
        raise ImportError('The I_from_V function requires scipy')
    
    argW = Rs*I0*Rsh*np.exp(Rsh*(Rs*(IL+I0)+V)/(nNsVth*(Rs+Rsh)))/(nNsVth*(Rs + Rsh))
    inputterm =lambertw(argW)

    # Eqn. 4 in Jain and Kapoor, 2004
    I = -V/(Rs + Rsh) - (nNsVth/Rs) * inputterm + Rsh*(IL + I0)/(Rs + Rsh)
    

    return I.real


def snlinverter(Inverter,Vmp,Pmp):
  '''
  Converts DC power and voltage to AC power using Sandia's Grid-Connected PV Inverter model

  Determine the AC power output of an inverter given the DC voltage, DC
  power, and appropriate Sandia Grid-Connected Photovoltaic Inverter
  Model parameters. The output, ACPower, is clipped at the maximum power
  output, and gives a negative power during low-input power conditions,
  but does NOT account for maximum power point tracking voltage windows
  nor maximum current or voltage limits on the inverter. 

  Parameters
  ----------

  Inverter : DataFrame

           A DataFrame defining the inverter to be used, giving the
           inverter performance parameters according to the Sandia
           Grid-Connected Photovoltaic Inverter Model (SAND 2007-5036) [1]. A set of
           inverter performance parameters are provided with PV_LIB, or may be
           generated from a System Advisor Model (SAM) [2] library using pvl_retreivesam. 
           
            Required DataFrame components are:

           =============   ==============================================================================================================================================================================================
           Field            DataFrame
           =============   ==============================================================================================================================================================================================
           Inverter.Pac0   AC-power output from inverter based on input power and voltage, (W) 
           Inverter.Pdc0   DC-power input to inverter, typically assumed to be equal to the PV array maximum power, (W)
           Inverter.Vdc0   DC-voltage level at which the AC-power rating is achieved at the reference operating condition, (V)
           Inverter.Ps0    DC-power required to start the inversion process, or self-consumption by inverter, strongly influences inverter efficiency at low power levels, (W)
           Inverter.C0     Parameter defining the curvature (parabolic) of the relationship between ac-power and dc-power at the reference operating condition, default value of zero gives a linear relationship, (1/W)
           Inverter.C1     Empirical coefficient allowing Pdco to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.C2     empirical coefficient allowing Pso to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.C3     empirical coefficient allowing Co to vary linearly with dc-voltage input, default value is zero, (1/V)
           Inverter.Pnt    ac-power consumed by inverter at night (night tare) to maintain circuitry required to sense PV array voltage, (W)
           =============   ==============================================================================================================================================================================================
  
  Vdc : float or DataFrame
          DC voltages, in volts, which are provided as input to the inverter. Vdc must be >= 0.
  Pdc : float or DataFrame

          A scalar or DataFrame of DC powers, in watts, which are provided
           as input to the inverter. Pdc must be >= 0.

  Returns
  -------

  ACPower : float or DataFrame

           Mdeled AC power output given the input 
           DC voltage, Vdc, and input DC power, Pdc. When ACPower would be 
           greater than Pac0, it is set to Pac0 to represent inverter 
           "clipping". When ACPower would be less than Ps0 (startup power
           required), then ACPower is set to -1*abs(Pnt) to represent nightly 
           power losses. ACPower is not adjusted for maximum power point
           tracking (MPPT) voltage windows or maximum current limits of the
           inverter.

  References
  ----------

  [1] (SAND2007-5036, "Performance Model for Grid-Connected Photovoltaic 
  Inverters by D. King, S. Gonzalez, G. Galbraith, W. Boyson)

  [2] System Advisor Model web page. https://sam.nrel.gov.

  See also
  --------

  pvl_sapm
  pvl_samlibrary
  pvl_singlediode

  '''

  Vars=locals()
  Expect={'Inverter':(''),
      'Vmp':'',
      'Pmp':''}

  var=pvl_tools.Parse(Vars,Expect)

  Paco=var.Inverter['Paco']
  Pdco=var.Inverter['Pdco']
  Vdco=var.Inverter['Vdco']
  Pso=var.Inverter['Pso']
  C0=var.Inverter['C0']
  C1=var.Inverter['C1']
  C2=var.Inverter['C2']
  C3=var.Inverter['C3']
  Pnt=var.Inverter['Pnt']


  A=Pdco*((1 + C1*((var.Vmp - Vdco))))
  B=Pso*((1 + C2*((var.Vmp - Vdco))))
  C=C0*((1 + C3*((var.Vmp - Vdco))))
  ACPower=((Paco / (A - B)) - C*((A - B)))*((var.Pmp - B)) + C*((var.Pmp - B) ** 2)
  ACPower[ACPower > Paco]=Paco
  ACPower[ACPower < Pso]=- 1.0 * abs(Pnt)

  return ACPower



