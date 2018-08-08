.. _singlediode:

Single Diode Model
==================

This section reviews the solutions to the single diode model used in
pvlib-python to generate an IV curve of a PV module.

pvlib-python supports two ways to solve the single diode model, by passing the
a ``method`` keyword to the :func:`pvlib.pvsystem.singlediode` function:

1. Lambert W-Function
2. Bishop's Algorithm

Lambert W-Function
------------------
When ``method='lambertw'``, the Lambert W-Function is used as previously shown
by Jain and Kapoor [1]. The following algorithm can be found on
`Wikipedia: Theory of Solar Cells
<https://en.wikipedia.org/wiki/Theory_of_solar_cells>`_ given the basic single
diode model equation.

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{\left(V + I R_s \right)}{n Ns V_{th}} \right) - 1 \right)
       - \frac{\left(V + I R_s \right)}{R_{sh}}

Lambert W-function is the inverse of the function
:math:`f \left( w \right) = w \exp \left( w \right)` or
:math:`w = f^{-1} \left( w \exp \left( w \right) \right)` also given as
:math:`w = W \left( w \exp \left( w \right) \right)`. Rearranging the single
diode equation above with a Lambert W-function yields the following.

.. math::

   z = \frac{R_s I_0}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}} \right)} \exp \left(
       \frac{R_s \left( I_L + I_0 \right) + V}{n Ns V_{th} \left(1 + \frac{R_s}{R_{sh}}\right)}
       \right)

The the module current can be solved using the Lambert W-function.

.. math::

   I = \frac{I_L + I_0 - \frac{V}{R_{sh}}}{1 + \frac{R_s}{R_{sh}}}
       - \frac{n Ns V_{th}}{R_s} W(z)


Bishop's Algorithm
------------------
The function :func:`pvlib.singlediode.bishop88` uses an explicit solution [2]
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
is bounded, and a bisection search within these points will always find
desired condition.

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
[1] A. Jain, A. Kapoor, "Exact analytical solutions of the
parameters of real solar cells using Lambert W-function", Solar
Energy Materials and Solar Cells, 81 (2004) 269-277.

[2] "Computer simulation of the effects of electrical mismatches in
photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
https://doi.org/10.1016/0379-6787(88)90059-2