.. _spectrum_user_guide:
.. currentmodule:: pvlib.spectrum


Spectrum
========

The spectrum functionality of pvlib-python includes simulating clear sky
spectral irradiance curves, calculating the spectral mismatch factor for
a range of single-junction PV cell technologies, and other calculations
such as converting between spectral response and EQE, and computing average
photon energy values from spectral irradiance data.

This user guide page summarizes some of pvlib-python's spectrum-related
capabilities, starting with a summary of spectral mismatch estimation models
available in pvlib-python.


Spectral mismatch models
------------------------
The spectral mismatch factor is the ratio of a PV device's response under a
given spectrum to its response under a reference spectrum, typically the
AM1.5G spectrum. It represents the relative difference in the performance of
a PV device under a spectrum different from the reference spectrum, and can be
used to correct the measured power output of a PV system for spectral effects.

pvlib-python contains several models to estimate the spectral mismatch factor
using atmospheric variables such as air mass, or calculate it exactly using
system and meteorological data such as spectral response and spectral
irradiance. Examples demonstrating the application of several spectral
mismatch models using pvlib-python are also available:
:ref:`sphx_glr_gallery_spectrum_spectral_factor.py` and Reference [1]_, the
latter of which also contains downloadable spectral response and spectral
irradiance data.

The table below summarizes the models currently available in pvlib, their
required inputs, cell technologies for which model coefficients have been
published, and references. Note that while most models are validated for
specific cell technologies, the Sandia Array Performance Model (SAPM) is
validated for a range of commercial modules. An extended review of a
wider range of models available in the published literature may be found in
Reference [2]_.


+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| Model                                               | Inputs                      | Default parameter availability                      | Reference |
+                                                     +                             +---------+---------+------+------+------+------------+           +
|                                                     |                             | mono-Si | poly-Si | CdTe | CIGS | a-Si | perovskite |           |
+=====================================================+=============================+=========+=========+======+======+======+============+===========+
| :py:func:`Caballero <spectral_factor_caballero>`    | :term:`airmass_absolute`,   |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |         |      |      |      |            |           |
|                                                     |:term:`precipitable_water`,  |   ✓     |    ✓    |  ✓   |   ✓  |  ✓   |     ✓      |   [2]_    |
|                                                     +-----------------------------+         |         |      |      |      |            |           |
|                                                     | aod                         |         |         |      |      |      |            |           |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| :py:func:`First Solar <spectral_factor_firstsolar>` | :term:`airmass_absolute`,   |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |    ✓    |  ✓   |      |      |            |   [3]_    |
|                                                     | :term:`precipitable_water`  |         |         |      |      |      |            |           |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| :py:func:`JRC <spectral_factor_jrc>`                | :term:`airmass_relative`,   |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |    ✓    |  ✓   |      |      |            +   [4]_    |
|                                                     | clearsky_index              |         |         |      |      |      |            |           |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| :py:func:`Polo <spectral_factor_polo>`              | :term:`precipitable_water`, |         |         |      |      |      |            |           |
|                                                     +-----------------------------+   ✓     |         |  ✓   |  ✓   |  ✓   |            +   [5]_    |
|                                                     | :term:`airmass_absolute`,   |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |         |      |      |      |            |           |
|                                                     | aod500,                     |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |         |      |      |      |            |           |
|                                                     | :term:`aoi`,                |         |         |      |      |      |            |           |
|                                                     +-----------------------------+         |         |      |      |      |            |           |
|                                                     | :term:`pressure`            |         |         |      |      |      |            |           |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| :py:func:`PVSPEC <spectral_factor_pvspec>`          | :term:`airmass_absolute`,   |         |         |      |      |      |            |           |
|                                                     +-----------------------------+   ✓     |    ✓    |  ✓   |  ✓   |  ✓   |            |   [6]_    |
|                                                     | clearsky_index              |         |         |      |      |      |            |           |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+
| :py:func:`SAPM <spectral_factor_sapm>`              | :term:`airmass_absolute`    |         |         |      |      |      |            |   [7]_    |
+-----------------------------------------------------+-----------------------------+---------+---------+------+------+------+------------+-----------+


References
----------
.. [1] A. Driesse, J. S. Stein, and M. Theristis, "Global horizontal spectral
       irradiance and module spectral response measurements: an open dataset
       for PV research Sandia National Laboratories, ALbuquerque, NM, USA, Rep.
       SAND2023-02045, 2023. Available:
       https://datahub.duramat.org/dataset/module-sr-library
.. [2] R. Daxini and Y. Wu, "Review of methods to account for the solar
       spectral influence on photovoltaic device performance," Energy, 
       vol. 286, p. 129461, Jan. 2024. :doi:`10.1016/j.energy.2023.129461`
.. [3] J. A. Caballero, E. Fernández, M. Theristis, F. Almonacid, and
       G. Nofuentes, "Spectral Corrections Based on Air Mass, Aerosol Optical
       Depth and Precipitable Water for PV Performance Modeling," IEEE Journal
       of Photovoltaics, vol. 8, no. 2, pp. 552–558, Mar. 2018. 
       :doi:`10.1109/JPHOTOV.2017.2787019`
.. [4] S. Pelland, J. Remund, and J. Kleissl, "Development and Testing of the
       PVSPEC Model of Photovoltaic Spectral Mismatch Factor," in Proc. 2020
       IEEE 47th Photovoltaic Specialists Conference (PVSC), Calgary, AB,
       Canada, 2020, pp. 1–6. :doi:`10.1109/PVSC45281.2020.9300932`
.. [5] J. Polo and C. Sanz-Saiz, 'Development of spectral mismatch models
       for BIPV applications in building façades', Renewable Energy, vol. 245,
       p. 122820, Jun. 2025, :doi:`10.1016/j.renene.2025.122820`
.. [6] D. L. King, W. E. Boyson, and J. A. Kratochvil, Photovoltaic Array
       Performance Model, Sandia National Laboratories, Albuquerque, NM, USA,
       Tech. Rep. SAND2004-3535, Aug. 2004. :doi:`10.2172/919131`
.. [7] M. Lee and A. Panchula, "Spectral Correction for Photovoltaic Module
       Performance Based on Air Mass and Precipitable Water," 2016 IEEE 43rd
       Photovoltaic Specialists Conference (PVSC), Portland, OR, USA, 2016,
       pp. 3696-3699. :doi:`10.1109/PVSC.2016.7749836`
.. [8] T. Huld, T. Sample, and E. Dunlop, "A Simple Model for Estimating the
       Influence of Spectrum Variations on PV Performance," pp. 3385–3389, Nov.
       2009, :doi:`10.4229/24THEUPVSEC2009-4AV.3.27`
.. [9] IEC 60904-7:2019, Photovoltaic devices — Part 7: Computation of the
       spectral mismatch correction for measurements of photovoltaic devices, 
       International Electrotechnical Commission, Geneva, Switzerland, 2019.