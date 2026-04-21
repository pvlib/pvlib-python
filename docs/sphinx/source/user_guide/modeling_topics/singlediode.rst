.. _singlediode:

Single diode models
===================

Single-diode models are a popular means of simulating the electrical output
of a PV module under any given irradiance and temperature conditions.
A single-diode model (SDM) pairs the single-diode equation (SDE) with a set of
auxiliary equations that predict the SDE parameters at any given irradiance
and temperature.  All SDMs use the SDE, but their auxiliary equations differ.
For more background on SDMs, see the `PVPMC website
<https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/single-diode-equivalent-circuit-models/>`_.

Three SDMs are currently available in pvlib: the CEC SDM, the PVsyst SDM,
and the De Soto SDM.  pvlib splits these models into two steps.  The first
is to compute the auxiliary equations using one of the following functions:

* CEC SDM: :py:func:`~pvlib.pvsystem.calcparams_cec`
* PVsyst SDM: :py:func:`~pvlib.pvsystem.calcparams_pvsyst`
* De Soto SDM: :py:func:`~pvlib.pvsystem.calcparams_desoto`

The second step is to use the output of these functions to compute points on
the SDE's I-V curve. Three points on the SDE I-V curve are typically of special
interest for PV modeling: the maximum power (MP), open circuit (OC), and
short circuit (SC) points. The most convenient function for computing these
points is :py:func:`pvlib.pvsystem.singlediode`. It provides several methods
for solving the SDE:

+------------------+------------+-----------+-------------------------+
| Method           | Type       | Speed     | Guaranteed convergence? |
+==================+============+===========+=========================+
| ``newton``       | iterative  | fast      | no                      |
+------------------+------------+-----------+-------------------------+
| ``brentq``       | iterative  | slow      | yes                     |
+------------------+------------+-----------+-------------------------+
| ``chandrupatla`` | iterative  | fast      | yes                     |
+------------------+------------+-----------+-------------------------+
| ``lambertw``     | explicit   | medium    | yes                     |
+------------------+------------+-----------+-------------------------+



Computing full I-V curves
-------------------------

Full I-V curves can be computed using
:py:func:`pvlib.pvsystem.i_from_v` and :py:func:`pvlib.pvsystem.v_from_i`, which
calculate either current or voltage from the other, with the methods listed
above.  It is often useful to
first compute the open-circuit or short-circuit values using
:py:func:`pvlib.pvsystem.singlediode` and then compute a range
of voltages/currents from zero to those extreme points.  This range can then
be used with the above functions to compute the I-V curve.


IV curves in reverse bias
-------------------------

The standard SDE does not account for diode breakdown at reverse bias. The
following functions can optionally include an extra term for modeling it:
:py:func:`pvlib.pvsystem.max_power_point`,
:py:func:`pvlib.singlediode.bishop88_i_from_v`,
and :py:func:`pvlib.singlediode.bishop88_v_from_i`. 


Recombination current for thin film cells
-----------------------------------------

The PVsyst SDM optionally modifies the SDE to better represent recombination
current in CdTe and a-Si modules. The modified SDE requires two additional
parameters. pvlib functions can compute the key points or full I-V curves using
the modified SDE:
:py:func:`pvlib.pvsystem.max_power_point`,
:py:func:`pvlib.singlediode.bishop88_i_from_v`,
and :py:func:`pvlib.singlediode.bishop88_v_from_i`.

Model parameter values
----------------------

Despite some models having parameters with similar names, parameter values are
specific to each model and thus must be produced with the intended model in mind.
For some models, sets of parameter values can be read from external sources,
for example:

* CEC SDM parameter database can be read using :py:func:`~pvlib.pvsystem.retrieve_sam`
* PAN files, which can be read using :py:func:`~pvlib.iotools.read_panond`

pvlib also provides a set of functions that can estimate SDM parameter values
from various datasources:

+---------------------------------------------------------------+---------+--------------------+
| Function                                                      | SDM     | Inputs             |
+===============================================================+=========+====================+
| :py:func:`~pvlib.ivtools.sdm.fit_cec_sam`                     | CEC     | datasheet          |
+---------------------------------------------------------------+---------+--------------------+
| :py:func:`~pvlib.ivtools.sdm.fit_desoto`                      | De Soto | datasheet          |
+---------------------------------------------------------------+---------+--------------------+
| :py:func:`~pvlib.ivtools.sdm.fit_desoto_sandia`               | De Soto | I-V curves         |
+---------------------------------------------------------------+---------+--------------------+
| :py:func:`~pvlib.ivtools.sdm.fit_pvsyst_sandia`               | PVsyst  | I-V curves         |
+---------------------------------------------------------------+---------+--------------------+
| :py:func:`~pvlib.ivtools.sdm.fit_pvsyst_iec61853_sandia_2025` | PVsyst  | IEC 61853-1 matrix |
+---------------------------------------------------------------+---------+--------------------+


Single-diode equation
---------------------

This section reviews the solutions to the single diode equation used in
pvlib-python to generate an IV curve of a PV module.

pvlib-python supports two ways to solve the single diode equation:

1. Lambert W-Function
2. Bishop's Algorithm

The :func:`pvlib.pvsystem.singlediode` function allows the user to choose the
method using the ``method`` keyword.

Lambert W-Function
******************
When ``method='lambertw'``, the Lambert W-function is used as previously shown
by Jain, Kapoor [1, 2] and Hansen [3]. The following algorithm can be found on
`Wikipedia: Theory of Solar Cells
<https://en.wikipedia.org/wiki/Theory_of_solar_cells>`_, given the basic single
diode model equation.

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{V + I R_s}{n N_s V_{th}} \right) - 1 \right)
       - \frac{V + I R_s}{R_{sh}}

Lambert W-function is the inverse of the function
:math:`f \left( w \right) = w \exp \left( w \right)` or
:math:`w = f^{-1} \left( w \exp \left( w \right) \right)` also given as
:math:`w = W \left( w \exp \left( w \right) \right)`. Defining the following
parameter, :math:`z`, is necessary to transform the single diode equation into
a form that can be expressed as a Lambert W-function.

.. math::

   z = \frac{R_s I_0}{n N_s V_{th} \left(1 + \frac{R_s}{R_{sh}} \right)} \exp \left(
       \frac{R_s \left( I_L + I_0 \right) + V}{n N_s V_{th} \left(1 + \frac{R_s}{R_{sh}}\right)}
       \right)

Then the module current can be solved using the Lambert W-function,
:math:`W \left(z \right)`.

.. math::

   I = \frac{I_L + I_0 - \frac{V}{R_{sh}}}{1 + \frac{R_s}{R_{sh}}}
       - \frac{n N_s V_{th}}{R_s} W \left(z \right)


Bishop's Algorithm
******************
The function :func:`pvlib.singlediode.bishop88` uses an explicit solution [4]
that finds points on the IV curve by first solving for pairs :math:`(V_d, I)`
where :math:`V_d` is the diode voltage :math:`V_d = V + I*Rs`. Then the voltage
is backed out from :math:`V_d`. Points with specific voltage, such as open
circuit, are located using the bisection search method, ``brentq``, bounded
by a zero diode voltage and an estimate of open circuit voltage given by

.. math::

   V_{oc, est} = n N_s V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)

We know that :math:`V_d = 0` corresponds to a voltage less than zero, and
we can also show that when :math:`V_d = V_{oc, est}`, the resulting
current is also negative, meaning that the corresponding voltage must be
in the 4th quadrant and therefore greater than the open circuit voltage
(see proof below). Therefore the entire forward-bias 1st quadrant IV-curve
is bounded because :math:`V_{oc} < V_{oc, est}`, and so a bisection search
between 0 and :math:`V_{oc, est}` will always find any desired condition in the
1st quadrant including :math:`V_{oc}`.

.. math::

   I = I_L - I_0 \left(\exp \left(\frac{V_{oc, est}}{n N_s V_{th}} \right) - 1 \right)
       - \frac{V_{oc, est}}{R_{sh}} \newline

   I = I_L - I_0 \left(\exp \left(\frac{n N_s V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{n N_s V_{th}} \right) - 1 \right)
       - \frac{n N_s V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\exp \left(\log \left(\frac{I_L}{I_0} + 1 \right) \right)  - 1 \right)
       - \frac{n N_s V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\frac{I_L}{I_0} + 1  - 1 \right)
       - \frac{n N_s V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_0 \left(\frac{I_L}{I_0} \right)
       - \frac{n N_s V_{th} \log \left(\frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = I_L - I_L - \frac{n N_s V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)}{R_{sh}} \newline

   I = - \frac{n N_s V_{th} \log \left( \frac{I_L}{I_0} + 1 \right)}{R_{sh}}

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
