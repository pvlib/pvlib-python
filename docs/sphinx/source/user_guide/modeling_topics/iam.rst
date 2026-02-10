.. _iam:


Incidence angle modifier
========================

Some fraction of the light incident on a PV module surface is reflected away or
absorbed before it reaches the PV cell.  This irradiance reduction depends
on the angle at which the light strikes the module (the angle of incidence,
:term:`AOI <aoi>`) and the optical properties of the module.

Some reduction occurs at all angles of incidence, even normal incidence.
However, because PV modules are rated with irradiance at normal incidence,
the reduction at normal incidence is implicit in the PV module's power rating
and does not need to be accounted for separately in a performance model.
Therefore, only the extra reduction at non-normal incidence should be modeled.

This is done using incidence angle modififer (:term:`IAM`) models.
Conceptually, IAM is the fraction of incident light that is
transmitted to the PV cell, normalized to the value at normal incidence:

.. math::

   IAM(\theta) = \frac{T(\theta)}{T(0)},

where :math:`T(\theta)` represents the transmitted light fraction at AOI :math:`\theta`.
IAM equals (by definition) 1.0 when AOI is zero and typically approaches zero
as AOI approaches 90°.  The shape of the IAM profile at intermediate AOI
is nonlinear and depends on the module's optical properties.

IAM may also depend on the wavelength of the light, the polarization of the light,
and which side of the module the light comes from.  However, IAM models usually
neglect these minor effects.

In pvlib, IAM is a unitless quantity (values from 0–1) and IAM functions take
input angles in degrees.


Types of models
---------------

Because total in-plane irradiance is the combination of light from many
directions, IAM values are computed for each component separately:

- *direct IAM*: IAM computed for the AOI of direct irradiance
- *circumsolar IAM*: typically approximated as equal to the direct IAM
- *diffuse IAM*: IAM integrated across the ranges of AOI spanning the sky and/or
  ground surfaces

Because diffuse light can be thought of as a field of many small beams of
direct light, diffuse IAM can then be understood as the IAM averaged across
those individual beams.  This averaging can be done explicitly or empirically.

In principle, IAM should be applied to all components of incident irradiance.
In practice, IAM is sometimes applied only to the direct component of in-plane
irradiance, as the direct component is often the largest contributor to total
in-plane irradiance and has a highly variable AOI across the day and year.

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
| :py:func:`~pvlib.iam.sapm`                | direct  | Can be non-monotonic and exceed 1.0       |
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
However, these generic values may not be suitable for all PV modules.
It should be noted that using the default parameter values for each 
model generally leads to different IAM profiles.

Module-specific values can be obtained via testing.  For example, IEC 61853-2
testing produces measured IAM values across the range of AOI and a corresponding
parameter value for the Martin-Ruiz model.  Parameter values for other models can
be determined using :py:func:`pvlib.iam.fit`.  Parameter values can also be approximately
converted between models using :py:func:`pvlib.iam.convert`.
