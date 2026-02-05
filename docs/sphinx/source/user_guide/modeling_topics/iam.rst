.. _iam:


Incidence angle modifier
========================

Some fraction of the light incident on a PV module surface is reflected away or
absorbed before it reaches the PV cell.  This irradiance reduction depends
on the angle at which the light strikes the module (the angle of incidence,
:term:`aoi`) and the optical properties of the module.

Some reduction occurs at all angles of incidence, even normal incidence.
However, because PV module testing is performed at normal incidence,
the reduction at normal incidence is implicit in the PV module's power rating
and does not need to be accounted for separately in a performance model.
Therefore, only the extra reduction at non-normal incidence should be modeled.

This is done using incidence angle modififer (:term:`IAM`) models.  IAM is
the fraction of incident light that is transmitted to the PV cell, normalized
to the value at normal incidence:

.. math::

   IAM(\theta) = \frac{T(\theta)}{T(0)},

where $T(\theta)$ represents the transmitted light fraction at AOI $\theta$.
IAM equals (by definition) 1.0 when AOI is zero and typically approaches zero
as AOI approaches 90 degrees.  The shape of the IAM profile at intermediate AOI
is nonlinear and depends on the module's optical properties.

In pvlib, IAM is a unitless quantity (values from 0–1) and IAM functions take
input angles in degrees.


Types of models
---------------

IAM is sometimes applied only to the beam component of in-plane irradiance, as
the beam component is often the largest contributor to total in-plane
irradiance and has a highly variable AOI across the day and year.  However,
in principle IAM should be applied to diffuse irradiance as well.  Modeling IAM
for diffuse irradiance is complicated by the light coming from a range
of angles.  IAM for direct irradiance is comparatively straightforward
because the entire component has a single angle of incidence.

The IAM models currently available in pvlib are summarized in the
following table:

+-------------------------------------------+---------+-------------------------------------------+
| Model                                     | Type    | Notes                                     |
+===========================================+=========+===========================================+
| :py:func:`~pvlib.iam.ashrae`              | direct  | Once common, now less used                |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.martin_ruiz`         | direct  | Used in the IEC 61853 standard            |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.martin_ruiz_diffuse` | diffuse | Used in the IEC 61853 standard            |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.physical`            | direct  | Physics-based; optional AR coating        |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.sapm`                | direct  | Can exhibit "wiggles" in the curve        |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.schlick`             | direct  | Does not take module-specific parameters  |
+-------------------------------------------+---------+-------------------------------------------+
| :py:func:`~pvlib.iam.schlick_diffuse`     | diffuse | Does not take module-specific parmaeters  |
+-------------------------------------------+---------+-------------------------------------------+

In addition to the core models above, pvlib provides several other functions
for IAM modeling:

- :py:func:`~pvlib.iam.interp`: interpolate between points on a measured IAM profile 
- :py:func:`~pvlib.iam.marion_diffuse` and :py:func:`~pvlib.iam.marion_integrate`:
  numerically integrate any IAM model across AOI to compute sky, horizon, and ground IAMs


Model parameters
----------------

Some IAM model functions provide default values for their parameters.
However, these generic values may not be suitable for all modules.
It should be noted that using the default parameter values for each 
model generally leads to different IAM profiles.

Module-specific values can be obtained via testing.  For example, IEC 61853-2
testing produces measured IAM values across the range of AOI and a corresponding
parameter value for the Martin-Ruiz model.  Parameters for other models can
be determined using :py:func:`pvlib.iam.fit`.  Parameters can be (approximately)
converted between models using :py:func:`pvlib.iam.convert`.
