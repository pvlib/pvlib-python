.. _singlediode:

Single Diode Equation
=====================

This section reviews the solutions to the single diode equation used in
pvlib-python to generate an IV curve of a PV module.

pvlib-python supports two ways to solve the single diode equation:

1. Lambert W-Function
2. Bishop's Algorithm

The :func:`pvlib.pvsystem.singlediode` function allows the user to choose the
method using the ``method`` keyword.

Lambert W-Function
------------------
When ``method='lambertw'``, the Lambert W-function is used as previously shown
by Jain, Kapoor [1, 2] and Hansen [3]. The following algorithm can be found on
`Wikipedia: Theory of Solar Cells
<https://en.wikipedia.org/wiki/Theory_of_solar_cells>`_, given the basic single
diode model equation.

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{V + I R_s}{n Ns V_{th}} \right) - 1 \right)
       - \frac{V + I R_s}{R_{sh}}

Lambert W-function is the inverse of the function
:math:`f \left( w \right) = w \exp \left( w \right)` or
:math:`w = f^{-1} \left( w \exp \left( w \right) \right)` also given as
:math:`w = W \left( w \exp \left( w \right) \right)`. Defining the following
parameter, :math:`z`, is necessary to transform the single diode equation into
a form that can be expressed as a Lambert W-function.

.. math::

   z = \frac{R_s I_0}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}} \right)} \exp \left(
       \frac{R_s \left( I_L + I_0 \right) + V}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}}\right)}
       \right)

Then the module current can be solved using the Lambert W-function,
:math:`W \left(z \right)`.

.. math::

   I = \frac{I_L + I_0 - \frac{V}{R_{sh}}}{1 + \frac{R_s}{R_{sh}}}
       - \frac{n Ns V_{th}}{R_s} W \left(z \right)


Bishop's Algorithm
------------------
The function :func:`pvlib.singlediode.bishop88` uses an explicit solution [4]
that finds points on the IV curve by first solving for pairs :math:`(V_d, I)`
where :math:`V_d` is the diode voltage :math:`V_d = V + I*Rs`. Then the voltage
is backed out from :math:`V_d`. Points with specific voltage, such as open
circuit, are located using the bisection search method, ``brentq``, bounded
by a zero diode voltage and an estimate of open circuit voltage given by

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