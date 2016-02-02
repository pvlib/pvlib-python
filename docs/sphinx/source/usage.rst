.. _usage:

Examples on Usage
=================


Tutorials
-----------

These tutorials aim at exemplying the usage of the core library for various uses.

* generals
    * imports
    * locations

* atmosphere.py tutorial -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/atmosphere.ipynb
    *  
* solarposition.py tutorial -- source
    *  Setup
    *  SPA output
        * sunset / sun rise?
    *  Speed tests
* irradiance.py tutorial -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/irradiance.ipynb
    * Extraterrestrial radiation
    * Clear sky models
    * Diffuse ground
    * Diffuse sky
        *  Isotropic
        *  Klucher
        *  Reindl
        *  Hay-Davies
        *  Perez
    *   Angle of incidence
    *   total_irrad (on plane of arry)
* pvsystem tutorial -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/pvsystem.ipynb
    *  systemdef
    *  Angle of Incidence Modifiers
    *  Sandia Cell Temp correction
    *  Sandia Inverter Model
    *  Sandia Array Performance Model
    *  SAPM IV curves
    *  DeSoto Model
    *  Single Diode Model
* Tracking -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/tracking.ipynb
    * Setup
    * Define input parameters.
    * Transform solar position to South facing coordinate system.
    * Transform solar position to panel coordinate system.
    * Determine the ideal tracking angle when ignoring backtracking.
    * Correct the tracking angle to account for backtracking.
    * Calculate the panel normal vector based on tracking angle.
    * Calculate the solar angle of incidence.
    * Calculate the panel tilt and azimuth.
* TMY tutorial -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/tmy.ipynb
    *  Import modules
*  TMY data and diffuse irradiance model -- source: http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/tmy_and_diffuse_irrad_models.ipynb
    * Setup
    * Diffuse irradiance models
        * Perez
        * HayDavies
        * Isotropic
        * King Diffuse model
        * Klucher Model
        * Reindl
    * Plot Results
* TMY to Power Tutorial -- http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/tmy_to_power.ipynb
    * Setup
    * Load TMY data
    * Calculate modeling intermediates
    * DC power using SAPM
    * AC power using SAPM
* Modeling multiple systems with system losses -- source: http://nbviewer.jupyter.org/github/jforbess/pvlib-python/blob/Issue84/docs/tutorials/system_loss_modeling.ipynb



Links to Notebooks
------------------------

Papers and Publications
_________________________________

* `PVSC 2015 <http://nbviewer.jupyter.org/github/pvlib/pvsc2015/blob/master/paper.ipynb#PVSC-2015>`_
    * Single axis tracker -- Simulation of single axis tracker output near Albuquerque
    * SAPM -- Some simulations using the Sandia Array Performance Model.
    * IV curves -- Make some IV curves based on this data
* 

Temporary edit hints
----------------------------

List of all notebooks:

* atmosphere.ipynb
* irradiance.ipynb
* package_overview.ipynb
* pvsystem.ipynb
* solarposition.ipynb
* tmy.ipynb
* tmy_and_diffuse_irrad_models.ipynb
* tmy_to_power.ipynb
* tracking.ipynb

Further tutorials

* `Modeling multiple systems with system losses <http://nbviewer.jupyter.org/github/jforbess/pvlib-python/blob/Issue84/docs/tutorials/system_loss_modeling.ipynb#Modeling-multiple-systems-with-system-losses>`_ -- discussion at: https://github.com/pvlib/pvlib-python/issues/84
    * Import site configuration data
    * Define key system model method
    * Load environmental data
    * Run performance simulation for each array
    * Plot plane of array irradiance vs output energy