
import pvl_tools
import numpy as np

def pvl_physicaliam(K,L,n,theta):

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
