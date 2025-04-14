.. _spectrum_user_guide:

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

pvlib-python contains several models to estimate the spectral mismatch factor
using atmospheric variables such as air mass, or system and meteorological
data such as spectral response and spectral irradiance. An example
demonstrating the application of three pvlib-python spectral mismatch models
is also available: :ref:`sphx_glr_gallery_spectrum_spectral_factor.py`. Here,
a comparison of all models available in pvlib-python is presented. An extended
review of a wider range of models available in the published literature may be
found in Reference [1]_.

The table below summarises the models currently available in pvlib, the inputs
required, cell technologies for which model coefficients have been published, 
and references. Note that while most models are validated for specific cell
technologies, the Sandia Array Performance Model (SAPM) and spectral mismatch
calculation are not specific to cell type; the former is validated for a range
of commerical module products.

+---------------------------------------------------------+----------------------------+-----------------------------+
| Model                                                   | Inputs                     | Cell technology | Reference |
+=========================================================+============================+=============================+
| :py:func:`~pvlib.spectrum.spectral_factor_caballero`    | :term:`absolute airmass`,  | CdTe,           |           |
|                                                         | :term:`precipitable water`,| mono-Si,        |           |
|                                                         | aerosol optical depth      | poly-Si,        | [2]_      |
|                                                         |                            | aSi,            |           |
|                                                         |                            | CIGS,           |           |
|                                                         |                            | Perovskite      |           |
+---------------------------------------------------------+----------------------------+-----------------------------+
| :py:func:`~pvlib.spectrum.spectral_factor_firstsolar`   | :term:`absolute airmass`,  | CdTe,           |           |
|                                                         | :term:`precipitable water` | poly-Si         | [3]_      |
+---------------------------------------------------------+----------------------------+-----------------------------+
| :py:func:`~pvlib.spectrum.spectral_factor_sapm`         | :term:`absolute airmass`   | Multiple        | [4]_      |
+---------------------------------------------------------+----------------------------+-----------------------------+
| :py:func:`~pvlib.spectrum.spectral_factor_pvspec`       | :term:`absolute airmass`,  | CdTe,           |           |
|                                                         | clearsky index             | poly-Si,        |           |
|                                                         |                            | mono-Si,        |           |
|                                                         |                            | CIGS,           | [5]_      |
|                                                         |                            | aSi             |           |
+---------------------------------------------------------+----------------------------+-----------------------------+
| :py:func:`~pvlib.spectrum.spectral_factor_jrc`          | :term:`absolute airmass`,  | CdTe,           |           |
|                                                         | clearsky index             | poly-Si         | [6]_      |
+---------------------------------------------------------+----------------------------+-----------------------------+
| :py:func:`~pvlib.spectrum.calc_spectral_mismatch_field` | spectral response,         | -               |           |
|                                                         | :term:`spectra`            |                 |           |
+---------------------------------------------------------+----------------------------+-----------------------------+


References
----------
.. [1] R. Daxini and Y. Wu, "Review of methods to account for the solar
       spectral influence on photovoltaic device performance," Energy, 
       vol. 286, p. 129461, Jan. 2024. :doi:`10.1016/j.energy.2023.129461`
.. [2] J. A. Caballero, E. Fernández, M. Theristis, F. Almonacid, and
       G. Nofuentes, "Spectral Corrections Based on Air Mass, Aerosol Optical
       Depth and Precipitable Water for PV Performance Modeling," IEEE Journal
       of Photovoltaics, vol. 8, no. 2, pp. 552–558, Mar. 2018. 
       :doi:`10.1109/JPHOTOV.2017.2787019`
.. [3] M. Lee and A. Panchula, "Spectral Correction for Photovoltaic Module
       Performance Based on Air Mass and Precipitable Water," 2016 IEEE 43rd
       Photovoltaic Specialists Conference (PVSC), Portland, OR, USA, 2016,
       pp. 3696-3699. :doi:`10.1109/PVSC.2016.7749836`
.. [4] D. L. King, W. E. Boyson, and J. A. Kratochvil, Photovoltaic Array
       Performance Model, Sandia National Laboratories, Albuquerque, NM, USA,
       Tech. Rep. SAND2004-3535, Aug. 2004. :doi:`10.2172/919131`
.. [5] S. Pelland, J. Remund, and J. Kleissl, "Development and Testing of the
       PVSPEC Model of Photovoltaic Spectral Mismatch Factor," in Proc. 2020
       IEEE 47th Photovoltaic Specialists Conference (PVSC), Calgary, AB,
       Canada, 2020, pp. 1–6. :doi:`10.1109/PVSC45281.2020.9300932`
.. [6] H. Thomas, S. Tony, and D. Ewan, “A Simple Model for Estimating the
       Influence of Spectrum Variations on PV Performance,” pp. 3385–3389, Nov.
       2009, :doi:10.4229/24THEUPVSEC2009-4AV.3.27
.. [7] IEC 60904-7:2019, Photovoltaic devices — Part 7: Computation of the
       spectral mismatch correction for measurements of photovoltaic devices, 
       International Electrotechnical Commission, Geneva, Switzerland, 2019.