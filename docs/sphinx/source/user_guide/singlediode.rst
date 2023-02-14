.. _singlediode:

Single Diode Equation
=====================

This section reviews the solutions to the single diode equation used in
pvlib-python to generate an IV curve of a PV module.

The single diode equation describes the current-voltage characteristic of
an ideal single-junction PV device:

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{V + I R_s}{n Ns V_{th}} \right) - 1 \right)
       - \frac{V + I R_s}{R_{sh}}

where

* :math:`I` is electrical current in Amps
* :math:`V` is voltage in Amps
* :math:`I_L` is the photocurrent in Amps
* :math:`I_0` is the dark, or saturation current, in Amps
* :math:`R_{sh}` is the shunt resistance in Ohm
* :math:`R_s` is the series resistance in Ohm
* :math:`n` is the diode (ideality) factor (unitless)
* :math:`Ns` is the number of cells in series. Cells are assumed to be identical.
* :math:`V_{th}` is the thermal voltage at each cell's junction, given by :math:`V_{th} = \frac{k}{q} T_K`,
  where :math:`k` is the Boltzmann constant (J/K), :math:`q` is the elementary charge (Couloumb) and :math:`T_k`
  is the cell temperature in K.

pvlib-python supports two ways to solve the single diode equation:

1. Using the Lambert W function
2. Bishop's algorithm

The :func:`pvlib.pvsystem.singlediode` function's ``method`` keyword allows the user to choose the solution method.

Lambert W-Function
------------------
When ``method='lambertw'``, the Lambert W function is used to find solutions :math:`(V, I)`.
The Lambert W function is the converse relation of the function :math:`f \left( w \right) = w \exp \left( w \right)`,
that is, if :math:`y \exp \left( y \right) = x`, then :math:`y = W(x)`.
As previously shown by Jain, Kapoor [1, 2] and Hansen [3], solutions to the single diode equation
may be written in terms of :math:`W`. Define a variable :math:`\theta` as 

.. math::

   \theta = \frac{R_s I_0}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}} \right)} \exp \left(
       \frac{R_s \left( I_L + I_0 \right) + V}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}}\right)}
       \right)

Then the module current can be written as a function of voltage, using the Lambert W-function,
:math:`W \left(z \right)`.

.. math::

   I = \frac{I_L + I_0 - \frac{V}{R_{sh}}}{1 + \frac{R_s}{R_{sh}}}
       - \frac{n Ns V_{th}}{R_s} W \left(\theta \right)


Similarly, the voltage can be written as a function of current by defining a variable :math:`\psi` as

.. math::

   \psi = \frac{I_0 R_{sh}}{n Ns V_{th}} \exp \left(\frac{\left(I_L + I_0 - I\right) R_{sh}}{n Ns V_{th}} \right)

Then

.. math::

   V = \left(I_L + I_0 - I\right) R_sh - I R_s - n Ns V_th W\left( \psi \right)

When computing :math:`V = V\left( I \right)`, care must be taken to avoid overflow errors because the argument
of the exponential function in :math:`\psi` can become large.

The pvlib function :func:`pvlib.pvsystem.singlediode` uses these expressions :math:`I = I\left(V\right)` and
:math:`V = V\left( I \right)` to calculate :math:`I_{sc}` and :math:`V_{oc}` respectively.

To calculate the maximum power point :math:`\left( V_{mp}, I_{mp} \right)` a root-finding method is used. At the
maximum power point, the derivative of power with respect to current (or voltage) is zero. Differentiating
the equation :math:`P = V I` and evaluating at the maximum power current

.. math::

   0 = \frac{dP}{dI} \Bigr|_{I=I_{mp}} = \left(V + \frac{dV}{dI}\right) \Bigr|_{I=I_{mp}}

obtains

.. math::

   \frac{dV}{dI}\Bigr|_{I=I_{mp}} = \frac{-V_{mp}}{I_{mp}}

Differentiating :math:`V = V(I)` with respect to current, and applying the identity
:math:`\frac{dW\left( x \right)}{dx} = \frac{W\left( x \right)}{x \left( 1 + W \left( x \right) \right)}` obtains

.. math::

   \frac{dV}{dI}\Bigr|_{I=I_{mp}} = -\left(R_s + \frac{R_{sh}}{1 + W\left( \psi \right)} \right)\Bigr|_{I=I_{mp}}

Equating the two expressions for :math:`\frac{dV}{dI}\Bigr|_{I=I_{mp}}` and rearranging yields

.. math::

   \frac{\left(I_L + I_0 - I\right) R_{sh} - I R_s - n Ns V_{th} W\left( \psi \right)}{R_s + \frac{R_{sh}}{1 + W\left( \psi \right)}}\Bigr|_{I=I_{mp}} - I_{mp} = 0.

The above equation is solved for :math:`I_{mp}` using Newton's method, and then :math:`V_{mp} = V \left( I_{mp} \right)` is computed.


Bishop's Algorithm
------------------
The function :func:`pvlib.singlediode.bishop88` uses an explicit solution [4]
that finds points on the IV curve by first solving for pairs :math:`(V_d, I)`
where :math:`V_d` is the diode voltage :math:`V_d = V + I R_s`. Then the voltage
is backed out from :math:`V_d`. Points with specific voltage or current are located
using either Newton's or Brent's method, ``method='newton'`` or ``method='brentq'``,
respectvely.

For example, to find the open circuit voltage, we start with an estimate given by

.. math::

   V_{oc, est} = n Ns V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)

We know that :math:`V_d = 0` corresponds to a voltage less than zero, and
we can also show that when :math:`V_d = V_{oc, est}`, the resulting
current is also negative, meaning that the corresponding voltage must be
in the 4th quadrant and therefore greater than the open circuit voltage
(see proof below). Therefore the entire forward-bias 1st quadrant IV-curve
is bounded because :math:`V_{oc} < V_{oc, est}`, and so a bisection search
between 0 and :math:`V_{oc, est}` will always find any desired condition in the
1st quadrant including :math:`V_{oc}`.

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{V_{oc, est}}{n Ns V_{th}} \right) - 1 \right)
       - \frac{V_{oc, est}}{R_{sh}} \newline

   I = I_L - I_0 \left(\exp \left(\frac{n Ns V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{n Ns V_{th}} \right) - 1 \right)
       - \frac{n Ns V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\exp \left(\log \left(\frac{I_L}{I_0} + 1 \right) \right)  - 1 \right)
       - \frac{n Ns V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\frac{I_L}{I_0} + 1  - 1 \right)
       - \frac{n Ns V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\frac{I_L}{I_0} \right)
       - \frac{n Ns V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_L - \frac{n Ns V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = - \frac{n Ns V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)}{R_{sh}}

References
----------
[1] "Exact analytical solutions of the parameters of real solar cells using
Lambert W-function," A. Jain, A. Kapoor, Solar Energy Materials and Solar Cells,
81, (2004) pp 269-277.
:doi:`10.1016/j.solmat.2003.11.018`

[2] "A new method to determine the diode ideality factor of real solar cell
using Lambert W-function," A. Jain, A. Kapoor, Solar Energy Materials and Solar
Cells, 85, (2005) 391-396.
:doi:`10.1016/j.solmat.2004.05.022`

[3] "Parameter Estimation for Single Diode Models of Photovoltaic Modules,"
Clifford W. Hansen, Sandia `Report SAND2015-2065
<https://prod.sandia.gov/techlib-noauth/access-control.cgi/2015/152065.pdf>`_,
2015 :doi:`10.13140/RG.2.1.4336.7842`

[4] "Computer simulation of the effects of electrical mismatches in
photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
:doi:`10.1016/0379-6787(88)90059-2`