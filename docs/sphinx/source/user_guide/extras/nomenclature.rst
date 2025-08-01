.. _nomenclature:

Nomenclature
============

There is a convention on consistent variable names throughout the library:

.. glossary::

    airmass
        Airmass
    
    airmass_absolute
        Absolute airmass
    
    airmass_relative
        Relative airmass
    
    albedo
        Ratio of reflected solar irradiance to global horizontal irradiance
        [unitless]
    
    aoi
        Angle of incidence. Angle between the surface normal vector and the
        vector pointing towards the sun's center. [°]
    
    aoi_projection
        cos(aoi). When the sun is behind the surface, the value is negative.
        For many uses, negative values must be set to zero.

    ape
        Average photon energy

    apparent_zenith
        Refraction-corrected solar zenith angle. The solar
        zenith angle describes the position of the sun relative to the vertical and is
        defined as the angle between a vector pointed straight up and a vector pointed
        at the sun, from the observer. [°]

    apparent_elevation
        Refraction-corrected solar elevation angle. This is the complement of
        :term:`apparent_zenith` (90 - apparent_zenith). [°]

    bhi
        Beam/direct horizontal irradiance

    dhi
        Diffuse horizontal irradiance

    dni
        Direct normal irradiance [Wm⁻²]. Irradiance received per unit area by a
        surface perpendicular (normal) to the sun's rays that propagate in a
        straight line from the sun.

    dni_clear
        Clear sky direct normal irradiance

    dni_extra
        Direct normal irradiance at top of atmosphere (extraterrestrial)

    effective_irradiance
        Effective irradiance

    eta_inv
        Inverter efficiency

    eta_inv_nom
        Nominal inverter efficiency

    eta_inv_ref
        Reference inverter efficiency

    g_poa_effective
        Broadband plane of array effective irradiance

    gamma_pdc
        Module temperature coefficient. Typically in units of 1/C.

    ghi
        Global horizontal irradiance

    ghi_clear
        Clearsky global horizontal irradiance [Wm⁻²]

    ghi_extra
        Horizontal irradiance at top of atmosphere (extraterrestrial)

    gri
        Ground-reflected irradiance

    i_sc
        Short circuit module current

    i_x, i_xx
        Sandia Array Performance Model IV curve parameters

    latitude
        Latitude in decimal degrees. Positive north of equator, negative to south.

    longitude
        Longitude in decimal degrees. Positive east of prime meridian, negative to west.

    pac, ac
        AC power

    pdc, dc
        DC power

    pdc0
        Nameplate DC rating

    photocurrent
        Photocurrent

    poa_diffuse
        Total diffuse irradiance in plane [Wm⁻²]. Sum of ground and sky diffuse
        components of global irradiance.

    poa_direct
        Direct/beam irradiance in plane [Wm⁻²].

    poa_global
        Global irradiance in plane.  Sum of diffuse and beam projection [Wm⁻²].

    poa_ground_diffuse
        In plane ground reflected irradiance [Wm⁻²].

    poa_sky_diffuse
        Diffuse irradiance in plane from scattered light in the atmosphere
        (without ground reflected irradiance) [Wm⁻²].

    precipitable_water
        Total precipitable water contained in a column of unit cross section
        from earth to top of atmosphere

    pressure
        Atmospheric pressure

    relative_humidity
        Relative humidity

    resistance_series
        Series resistance

    resistance_shunt
        Shunt resistance

    saturation_current
        Diode saturation current

    solar_azimuth
        Azimuth angle of the sun in degrees East of North. The solar azimuth angle
        describes the sun’s position along the horizon relative to the observer.
        The pvlib-python convention is defined as degrees East of North, so
        North = 0°, East = 90°, South = 180°, West = 270°.

    solar_zenith
        Zenith angle of the sun in degrees. This is the angle between is between a
        vector pointed straight up and a vector pointed at the sun, from the observer.
        This is the complement of solar elevation (90 - elevation). [°]

    spectra
    spectra_components
        Spectral irradiance (components) [Wm⁻²nm⁻¹]. The amount of energy
        incident on a unit area per unit time and per unit
        wavelength. As with broadband irradiance, global spectral irradiance
        is composed of direct and diffuse components.
    
    surface_azimuth
        Azimuth angle of the surface in degrees East of North. This angle describes the
        horizontal projection of the normal vector from the surface. The pvlib-python
        convention is defined as degrees East (clockwise) of North, so North = 0°,
        East = 90°, South = 180°, West = 270°.

    surface_tilt
        Tilt from horizontal [°]. The surface tilt angle 
        is defined as degrees from the horizontal
        such that a surface facing up would have a surface tilt of 0°, and one facing
        the horizon would be 90°.  [°]

    temp_air
        Temperature of the air

    temp_cell
        Temperature of the cell

    temp_dew
        Dewpoint temperature

    temp_module
        Temperature of the module

    tz
        Timezone

    v_mp, i_mp, p_mp
        Module voltage, current, power at the maximum power point

    v_oc
        Open circuit module voltage

    wind_direction
        Wind direction

    wind_speed
        Wind speed


For further explanation of the variables, common symbols, and
units, refer to the following sources from `SoDa Service <http://www.soda-pro.com/home>`_:

   * `Acronyms, Terminology and Units <https://www.soda-pro.com/help/general/acronyms-terminology-and-units>`_
   * `Plane orientations and radiation components <https://www.soda-pro.com/help/general/plane-orientations-and-radiation-components>`_
   * `Time references <https://www.soda-pro.com/help/general/time-references>`_

.. note:: These further references might not use the same terminology as
          *pvlib*. But the physical process referred to is the same.
