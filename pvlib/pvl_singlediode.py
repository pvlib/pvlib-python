

import numpy as np
import pvl_tools
import pandas as pd
import scipy 
from scipy.special import lambertw
import time


def pvl_singlediode(Module,IL,I0,Rs,Rsh,nNsVth,**kwargs):
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

    argW = Rs*I0*Rsh*np.exp(Rsh*(Rs*(IL+I0)+V)/(nNsVth*(Rs+Rsh)))/(nNsVth*(Rs + Rsh))
    inputterm =lambertw(argW)

    # Eqn. 4 in Jain and Kapoor, 2004
    I = -V/(Rs + Rsh) - (nNsVth/Rs) * inputterm + Rsh*(IL + I0)/(Rs + Rsh)
    

    return I.real