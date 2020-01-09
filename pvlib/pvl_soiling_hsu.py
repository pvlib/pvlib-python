import argparse
import time, datetime
import numpy as np
from scipy import integrate, special

def accumarray(Indx, value):
	n = np.max(Indx)+1
	if(np.isscalar(value)):
		value = np.repeat(value, len(Indx))

	A = np.zeros((n,))
	for i in range(n):
		A[i]=np.sum(value[Indx[:]==i]) 
	return A

def pvl_soiling_hsu(Time, Rain, RainThresh, Tilt, PM2_5, PM10, ModelType = 2, RainAccPeriod = 1, LUC = 8, WindSpeed = 2, Temperature = 12):
'''
    PVL_SOILING_HSU Calculates soiling rate over time given particulate and rain data

    Parameters
    ----------
    
    Time : Time_Structure
        Time values for the soiling function do not need to be 
        regularly spaced, although large gaps in timing are discouraged. (datetime)
    
    Rain : numeric
        Rainfall values should be in mm of rainfall. Programmatically, rain
        is accumulated over a given time period, and cleaning is applied
        immediately after a time period where the cleaning threshold is
        reached. (mm)
    
    RainThresh : numeric
        RainThresh is a scalar for the amount of rain, in mm, in an accumulation
        period needed to clean the PV modules. In periods where the 
        accumulated rain meets or exceeds RainThresh, the panels are assumed 
        to be cleaned immediately after the accumulation period 
        [1] suggests a good RainThresh could be 1mm, but the time period is
        not specified. Rain accumulation period length can be adjusted in the
        optional input RainAccPeriod. (mm)
    
    Tilt : numeric
        Tilt is a scalar or vector for the tilt of the PV panels. Changing tilt
        angles (e.g. in tracking cases) can be accomodated, and tilt angles
        are correlated with the entries in Time.(degree)

    PM2_5 : numeric
        PM2_5 is the concentration of airborne particulate matter (PM) with diameter less 
        than 2.5 microns. (g/m^3)

    PM10 : numeric
        PM10 is the concentration of airborne particulate matter (PM) with diameter less than
        10 microns, in g/m^3.

    ModelType : numeric
        ModelType is an optional input to the function to determine the 
        the model type to be used in the soiling model, see [1]. A
        value of "1" indicates that the Variable Deposition Velocity model
        shall be used, a value of "2" indicates that the Fixed Settling
        Velocity model shall be used, and a value of "3" indicates that the
        Fixed Deposition Velocity model shall be used. [1] indicates that the
        Fixed Settling Velocity model performs best under a wide range of
        conditions, and thus "2" is the default ModelType if ModelType is omitted. 
        Validation efforts by Sandia National Laboratories
        confirm these findings. If an incorrect ModelType is provided, the
        Fixed Settling Velocity (type 2) will be used (with a warning).

    RainAccPeriod : numeric
        RainAccPeriod is an optional input that specifies the period, in hours,
        over which to accumulate rainfall totals before checking against the
        rain cleaning threshold. For example, if the rain threshold is
        0.5 mm per hour, then RainThresh should be 0.5 and RainAccPeriod
        should be 1. If the threshold is 1 mm per hour, then the values
        should be 1 and 1, respectively. The minimum RainAccPeriod is 1
        hour. The default value is 1, indicating hourly rain accumulation.
        Accumulation periods exceeding 24 (daily accumulation) are not
        recommended. (mm per hour)
    
    LUC : numeric
        LUC is an optional input to the function, but it is required for the
        Variable Deposition Model. LUC is the Land Use Category as specified
        in Table 19.2 of [2]. LUC must be a numeric scalar with value 1, 4,
        6, 8, or 10, corresponding to land with evergreen trees, deciduous
        trees, grass, desert, or shrubs with interrupted woodlands. If 
        omitted, the default value of 8 (desert) is used. 
    WindSpeed : numeric    
        WindSpeed is an optional input to the function, but is required for the
        Variable Deposition Model. WindSpeed is a scalar or vector value with
        the same number of elements as Time, and must be in meters per
        second. If WindSpeed is omitted, the value of 2 m/s is used as
        default. (m/s)
    
    Temperature : numeric    
        Temperature is an optional input to the function, but is required for
        the Variable Deposition Model. Temperature is a scalar or vector
        value with the same number of Elements as Time and must be in degrees
        C. If Temperature is omitted, the value of 12 C is used as default. (Celcius)

    Returns
    -------
    SR : numeric
        The soiling ratio (SR) of a tilted PV panel, this is a number
        between 0 and 1. SR is a time series where each element of SR
        correlates with the accumulated soiling and rain cleaning at the times
        specified in Time.

    Notes
    ------
    The following default values

    ============================   ================
    Parameter                      Value
    ============================   ================
    ModelType                      2
    Temperature at zero altitude   288.15 K
    RainAccPeriod                  1 mm per hour
    LUC                            2
    WindSpeed                      2 m/s
    Temperature                    12 C
    ============================   ================

    References
    -----------
    .. [1] M. Coello and L. Boyle, "Simple Model For Predicting Time Series 
       Soiling of Photovoltaic Panels," in IEEE Journal of Photovoltaics.
       doi: 10.1109/JPHOTOV.2019.2919628
    .. [2] Atmospheric Chemistry and Physics: From Air Pollution to Climate
       Change. J. Seinfeld and S. Pandis. Wiley and Sons 2001.   

    '''

	# Time is datetime structure  
	TimeAsDatenum = time.mktime(Time.timetuple())

	RainAccAsDatenum = np.floor(TimeAsDatenum * 24/ RainAccPeriod)

	##### Doubt ######

	[RainAccTimes, UnqRainAccFrstVal, UnqRainAccIndx] = np.unique(RainAccAsDatenum,return_index=True, return_inverse=True)

	RainAtAccTimes = accumarray(UnqRainAccIndx, Rain);
	##### Doubt ######

	AccumRain = np.zeros_like(Rain)
	AccumRain[UnqRainAccFrstVal[1:]-1] = RainAtAccTimes[1:-1]
	AccumRain[-1] = RainAtAccTimes[-1]

	vd_switch = {
	    
	        1:depo_veloc(Temperature, WindSpeed, LUC), # case 1  Variable Deposition Velocity
	    
	        2:np.array([0.0009,0.004]), # case 2 % Fixed Settling Velocity in m/s
	    
	    	3:np.array([0.0015,0.0917]) # case 3 % Fixed Deposition Velcoity in m/s

	        }

	try:
		vd = vd_switch[ModelType]
	except:
		print("Unknown MoodelType, assuming MoodelType to 2.")
		MoodelType = 2
		vd = vd_switch[ModelType]



	PMConcentration=np.zeros(len(np.ravel(TimeAsDatenum)),2) # pre-allocate with NAN


	PMConcentration[:,0] = PM2_5 # fill PM2.5 data in column 1
	PMConcentration[:,1] = PM10 - PM2_5 # fill in PM2.5-PM10 data in column 2

	PMConcentration[PM10 - PM2_5 < 0 , 1] = 0 

	PMConcentration = PMConcentration * 10**-6;

	F = PMConcentration * vd  # g * m^-2 * s^-1, by particulate size
	HorizontalTotalMassRate = F[:,0]+ F[:,2] # g * m^-2 * s^-1, total

	TiltedMassRate = HorizontalTotalMassRate * np.cosd(mp.pi*Tilt/180)

	TiltedMassNoRain = integrate.cumtrapz(TimeAsDatenum*86400, TiltedMassRate)

	TiltedMass = TiltedMassNoRain;

	for cntr1 in range(0,len(RainAtAccTimes)):
	    if (RainAtAccTimes[cntr1] >= RainThresh):
	        TiltedMass[UnqRainAccFrstVal[cntr1+1]:end] = TiltedMass[UnqRainAccFrstVal[cntr1+1]:end]-TiltedMass[UnqRainAccFrstVal[cntr1+1]-1]
	    

	SoilingRate = 34.37 * special.erf(0.17*TiltedMass**0.8473)

	SR = (100 - SoilingRate)/100


def depo_velocity(T, WindSpeed, LUC):

	# convert temperature into Kelvin 
	T = T + 273.15

	# save wind data
	if(np.isscalar(WindSpeed)):
		u = np.array([WindSpeed])
	else:
		u = WindSpeed

	g=9.81         #gravity in m/s^2
	Na=6.022*10**23 #avagadros number
	R=8.314        #Universal gas consant in m3Pa/Kmol
	k=1.38*10**-23  #Boltzmann's constant in m^2kg/sK
	P=101300       #pressure in Pa
	rhoair= 1.2041 #density of air in kg/m3
	z0=1
	rhop=1500      #Assume density of particle in kg/m^3

	switcher={
		1:0.56,
		4:0.56,
		6:0.54,
		8:0.54,
		10:0.54,
	}

	try:
		gamma = switcher[LUC]
	except:
		print("Unknown Land Use Category (LUC), assuming LUC 8.")
		LUC = 8
		gamma = switcher[LUC]


	# Diameter of particle in um
	Dpum=np.array([2.5,10])
	Dpm=Dpum*10**-6   # Diameter of particle in m

	# Calculations
	mu=1.8*10**-5*(T/298)**0.85      #viscosity of air in kg/m s
	nu=mu/rhoair
	lambda1=2*mu/(P*(8.*0.0288/(np.pi*R*T))**(0.5))   #mean free path
	ll = np.array([lambda1,lambda1])
	Cc = 1+2*ll/Dpm*(1.257+0.4*np.exp(-1.1*Dpm/(ll*2))) #slip correction coefficient

	###########################################################################
	# Calculate vs
	###########################################################################
	vs = rhop*Dpm**2*(g*Cc/(mu*18)) # particle settling velocity

	###########################################################################
	# Calculate rb
	###########################################################################
	ustar = np.zeros_like(u, dtype=float) # pre-allocate ustar
	# Equation 11.66 in Ramaswami (and 16.67 and Sienfeld &Pandis)
	ustar[u > 0] = 0.4 * u[u > 0]/np.log(10/z0)
	ustar[u<=0] = 0.001

	D=k*T*(Cc/(3*np.pi*mu*Dpm))

	Sc=nu/D
	# gamma=0.56      #for urban
	# alpha=1.5     #for urban      
	EB=Sc**(-1 * gamma)
	St = vs*(ustar ** 2)/ (g*nu)

	EIM=10.0**(-3.0/St)   #For smooth surfaces
	#EIM=((St)./(0.82+St)).^2

	R1=np.exp(-St**(0.5))  # percentage of particles that stick

	rb=1/(3*(EB+EIM)*ustar*R1)

	###########################################################################
	# Calculate ra
	###########################################################################
	a=np.array([-0.096,-0.037,-0.002,0,0.004,0.035])
	b=np.array([0.029,0.029,0.018,0,-0.018,-0.036])

	# For wind speeds <= 3, use a = -0.037 and b = 0.029
	# For wind speeds >3 and <=5, use a = -.002, b = 0.018
	# For wind speeds > 5, use a = 0, b = 0
	avals = a[1]*np.ones_like(u, dtype=float)
	avals[u>3] = a[2]
	avals[u>5] = a[3]

	bvals = b[1]*np.ones_like(u, dtype=float)
	bvals[u>3] = b[2]
	bvals[u>5] = b[3]

	L = 1/(avals + bvals* np.log(z0))

	zeta0=z0/L
	zeta=10.0/L
	eta = ((1-15*zeta)**(0.25))
	eta0 = ((1-15*zeta0)**(0.25))

	ra = np.zeros_like(zeta, dtype=float) # Preallocate memory
	ra[zeta == 0] = (1/ (0.4* ustar[zeta == 0]))* np.log(10.0/z0)
	ra[zeta > 0] = (1/(0.4*ustar[zeta > 0]))*(np.log(10.0/z0) + 4.7*(zeta[zeta > 0]-zeta0[zeta > 0]))
	ra[zeta < 0] = (1 / (0.4 * ustar[zeta < 0]))* (np.log(10.0/ z0) + np.log((eta0[zeta < 0]**2 + 1) * (eta0[zeta < 0]+1)**2 / ((eta[zeta < 0]**2 + 1) * (eta[zeta < 0]+1)**2)) + 2 * (np.arctan(eta[zeta < 0])-np.arctan(eta0[zeta < 0])))

	###########################################################################
	# Calculate vd and mass flux
	###########################################################################

	vd=1/(ra+rb)+vs

	return vd

