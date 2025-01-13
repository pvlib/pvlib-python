.. _modelcomparison:

Model comparison tables
=======================

Intro?

Spectral mismatch models
------------------------

pvlib-python contains several models to estimate the spectral mismatch factor
using atmospheric variables such as air mass, or system and meteorological
data such as spectral response and spectral irradiance. An example
demonstrating the application of three pvlib-python spectral mismatch models
is also available: :ref:`sphx_glr_gallery_spectrum_spectral_factor.py. Here,
a comparison of all models available in pvlib-python is presented. An extended
review of a wider range of models available in the published literature may be
found in Reference [X].

The table below summarises the models currently available in pvlib, the inputs
required, cell technologies for which model coefficients have been published, 
source of data used for model development and validation, and references.

+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| Model                                                   | Inputs                                                       | Cell technology | Data source       | Reference |
+=========================================================+==============================================================+=================+===================+===========+
| :py:func:`~pvlib.spectrum.spectral_factor_caballero`    | absolute airmass, precipitable water, aerosol optical depth  | Multiple        | SMARTS, measured  |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| :py:func:`~pvlib.spectrum.spectral_factor_firstsolar`   | absolute airmass, precipitable water                         | mSi, CdTe       | SMARTS+TMY, field |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| :py:func:`~pvlib.spectrum.spectral_factor_sapm`         | absolute airmass                                             | Multiple        | Field             |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| :py:func:`~pvlib.spectrum.spectral_factor_pvspec`       | absolute airmass, clearsky index                             | Multiple        | Field             |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| :py:func:`~pvlib.spectrum.spectral_factor_jrc`          | absolute airmass, clearsky index                             | Multiple        | Field             |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+
| :py:func:`~pvlib.spectrum.calc_spectral_mismatch_field` | spectral response, spectral irradiance                       |       -         |     -             |   [X]     |
+---------------------------------------------------------+--------------------------------------------------------------+-----------------+-------------------+-----------+


