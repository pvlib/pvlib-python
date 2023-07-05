.. _bifacial:

Bifacial modeling
=================

This section reviews the bifacial modeling capabilities of
pvlib-python.

A bifacial module accepts light on both surfaces. Bifacial modules usually have
a front and back surface, with the back surface intended to face away from
the primary source of light. The primary challenge in modeling a PV system
with bifacial modules is estimating the irradiance on the front and back
surfaces.

pvlib-python provides two groups of functions for estimating front and back
irradiance:

1. a wrapper for convenient use of the pvfactors model:
:py:func:`~pvlib.bifacial.pvfactors.pvfactors_timeseries`

2. the infinite sheds bifacial model:
:py:func:`~pvlib.bifacial.infinite_sheds.get_irradiance`
:py:func:`~pvlib.bifacial.infinite_sheds.get_irradiance_poa`


pvfactors
---------

The pvfactors model calculates
incident irradiance on the front and back surfaces of an array. pvfactors uses
a 2D geometry which assumes that the array is made up of long, regular rows.
Irradiance is calculated in the middle of a row; end-of-row effects are not
included. pvfactors can model arrays in fixed racking or on single-axis
trackers with a user-configurable number of rows.

Prior to pvlib version 0.10.1, pvlib used the original SunPower implementation
of the model via the `pvfactors <https://github.com/sunpower/pvfactors>`_
package.  Starting in version 0.10.1, pvlib instead uses
`solarfactors <https://github.com/pvlib/solarfactors>`_, a drop-in
replacement implementation maintained by the pvlib community.
This switch was made when the original ``pvfactors`` package became
difficult to install in modern python environments.
``solarfactors`` implements the same model as ``pvfactors`` and is kept
up to date and working over time.  Note that "solarfactors" is only the name
on PyPI (meaning it is installed via ``pip install solarfactors``);
after installation, Python code still accesses it as "pvfactors"
(e.g. ``import pvfactors``).


Infinite Sheds
--------------

The "infinite sheds" model [1] is a 2-dimensional model of irradiance on the
front and rear surfaces of a PV array. The model assumes that the array
comprises parallel, equally spaced rows (sheds) and calculates irradiance in
the middle of a shed which is far from the front and back rows of the array.
Sheds are assumed to be long enough that end-of-row effects can be
neglected. Rows can be at fixed tilt or on single-axis trackers. The ground
is assumed to be horizontal and level, and the array is mounted at a fixed
height above the ground.

The infinite sheds model accounts for the following effects:

    - limited view from the row surfaces to the sky due to blocking of the
      sky by nearby rows;
    - reduction of irradiance reaching the ground due to shadows cast by
      rows and due to blocking of the sky by nearby rows.

The model operates in the following steps:

1. Find the fraction of unshaded ground between rows, ``f_gnd_beam`` where
   both direct and diffuse irradiance is received. The model assumes that
   there is no direct irradiance in the shaded fraction ``1 - f_gnd_beam``.
2. Calculate the view factor, ``fz_sky``, from the ground to the sky accounting
   for the parts of the sky that are blocked from view by the array's rows.
   The view factor is multiplied by the sky diffuse irradiance to calculate
   the diffuse irradiance reaching the ground. Sky diffuse irradiance is thus
   assumed to be isotropic.
3. Calculate the view factor from the row surface to the ground which
   determines the fraction of ground-reflected irradiance that reaches the row
   surface.
4. Find the fraction of the row surface that is shaded from direct irradiance.
   Only sky and ground-reflected irradiance reach the the shaded fraction of
   the row surface.
5. For the front and rear surfaces, apply the incidence angle modifier to
   the direct irradiance and sum the diffuse sky, diffuse ground, and direct
   irradiance to compute the plane-of-array (POA) irradiance on each surface.
6. Apply the bifaciality factor, shade factor and transmission factor to
   the rear surface POA irradiance and add the result to the front surface
   POA irradiance to calculate the total POA irradiance on the row.

Array geometry is defined by the following:

    - ground coverage ratio (GCR), ``gcr``, the ratio of row slant height to
      the spacing between rows (pitch).
    - height of row center above ground, ``height``.
    - tilt of the row from horizontal, ``surface_tilt``.
    - azimuth of the row's normal vector, ``surface_azimuth``.

View factors from the ground to the sky are calculated at points spaced along
a one-dimensional axis on the ground, with the origin under the center of a
row and the positive direction toward the right. The positive direction is
considered to be towards the "front" of the array. Array height differs in this
code from the description in [1], where array height is described at the row's
lower edge.

If ``model='isotropic'`` (the default), ``dhi`` is assumed to be isotropically
distributed across the sky dome as in [1]_. This implementation provides an
optional extension to [1]_ to model sky anisotropy: if ``model='haydavies'``,
the input ``dhi`` is decomposed into circumsolar and isotropic components using
:py:func:`~pvlib.irradiance.haydavies`, with the circumsolar component treated
as additional ``dni`` for transposition and shading purposes.

This model is influenced by the 2D model published by Marion, *et al.* in [2].


References
----------
.. [1] Mikofski, M., Darawali, R., Hamer, M., Neubert, A., and Newmiller,
   J. "Bifacial Performance Modeling in Large Arrays". 2019 IEEE 46th
   Photovoltaic Specialists Conference (PVSC), 2019, pp. 1282-1287.
   doi: 10.1109/PVSC40753.2019.8980572.
.. [2] Marion. B., MacAlpine, S., Deline, C., Asgharzadeh, A., Toor, F.,
   Riley, D., Stein, J. and Hansen, C. "A Practical Irradiance Model for
   Bifacial PV Modules".2017 IEEE 44th Photovoltaic Specialists Conference
   (PVSC), 2017, pp. 1537-1543. doi: 10.1109/PVSC.2017.8366263

