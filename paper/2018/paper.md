---
title: 'pvlib python: a python package for modeling solar energy systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - renewable energy
authors:
  - name: William F. Holmgren
    orcid: 0000-0001-6218-9767
    affiliation: 1
  - name: Clifford W. Hansen
    orcid: 0000-0002-8620-5378
    affiliation: 2
  - name: Mark A. Mikofski
    orcid: 0000-0001-8001-8582
    affiliation: 3
affiliations:
 - name: Department of Hydrology and Atmospheric Sciences, University of Arizona
   index: 1
 - name: Sandia National Laboratories
   index: 2
 - name: DNV-GL
   index: 3
date: 2 August 2018
bibliography: paper.bib
---

# Summary

pvlib python is a community-supported open source tool that provides a
set of functions and classes for simulating the performance of
photovoltaic energy systems. pvlib python aims to provide reference
implementations of models relevant to solar energy, including for
example algorithms for solar position, clear sky irradiance, irradiance
transposition, DC power, and DC-to-AC power conversion. pvlib python is
an important component of a growing ecosystem of open source tools for
solar energy [@Holmgren2018].

pvlib python is developed on GitHub by contributors from academia,
national laboratories, and private industry. pvlib python is released
with a BSD 3-clause license allowing permissive use with attribution.
pvlib python is extensively tested for functional and algorithm
consistency. Continuous integration services check each pull request on
multiple platforms and Python versions. The pvlib python API is
thoroughly documented and detailed tutorials are provided for many
features. The documentation includes help for installation and
guidelines for contributions. The documentation is hosted at
readthedocs.org as of this writing. A Google group and StackOverflow tag
provide venues for user discussion and help.

The pvlib python API was designed to serve the various needs of the many
subfields of solar power research and engineering. It is implemented in
three layers: core functions, the ``Location`` and ``PVSystem`` classes,
and the ``ModelChain`` class. The core API consists of a collection of
functions that implement algorithms. These algorithms are typically
implementations of models described in peer-reviewed publications. The
functions provide maximum user flexibility, however many of the function
arguments require an unwieldy number of parameters. The next API level
contains the ``Location`` and ``PVSystem`` classes. These abstractions
provide simple methods that wrap the core function API layer. The method
API simplification is achieved by separating the data that represents
the object (object attributes) from the data that the object methods
operate on (method arguments). For example, a ``Location`` is
represented by a latitude, longitude, elevation, timezone, and name,
which are ``Location`` object attributes. Then a ``Location`` object
method operates on a ``datetime`` to get the corresponding solar
position. The methods combine these data sources when calling the
function layer, then return the results to the user. The final level of
API is the ``ModelChain`` class, designed to simplify and standardize
the process of stitching together the many modeling steps necessary to
convert a time series of weather data to AC solar power generation,
given a PV system and a location.

pvlib python was ported from the PVLib MATLAB toolbox in 2014
[@Stein2012, @Andrews2014]. Efforts to make the project more pythonic
were undertaken in 2015 [@Holmgren2015]. Additional features continue to
be added, see, for example [@Stein2016, @Holmgren2016] and the
documentation's "What's New" section.

pvlib python has been used in numerous studies, for example, of solar
power forecasting [@Gagne2017, @Holmgren2017], development of solar
irradiance models [@Polo2016], and estimation of photovoltaic energy
potential [@Louwen2017]. Mikofski et. al. used pvlib python to study
the accuracy of clear sky models with different aerosol optical depth
and precipitable water data sources [@Mikofski2017] and to determine the
effects of spectral mismatch on different PV devices [@Mikofski2016].
pvlib python is a foundational piece of an award, "An Open Source
Evaluation Framework for Solar Forecasting," made under the Department
of Energy Solar Forecasting 2 program [@DOESF2].

Plans for pvlib python development includes the implementation of new
and existing models, addition of functionality to assist with
input/output, and improvements to API consistency.

The source code for each pvlib python version is archived with Zenodo
[@pvlibZenodo].

# Acknowledgements

The authors acknowledge and thank the code, documentation, and
discussion contributors to the project.

WH acknowledges support from the Department of Energy's Energy
Efficiency and Renewable Energy Postdoctoral Fellowship Program
(2014-2016), Tucson Electric Power, Arizona Public Service, and Public
Service Company of New Mexico (2016-2018), and University of Arizona
Institute for Energy Solutions (2017-2018).

CH acknowledges support from the U.S. Department of Energy's Solar
Energy Technology Office.

WH and CH acknowledge support from the Department of Energy Solar
Forecasting 2 program.

MM acknowledges support from SunPower Corporation (2016-2017).

Sandia National Laboratories is a multi-mission laboratory managed and
operated by National Technology and Engineering Solutions of Sandia,
LLC., a wholly owned subsidiary of Honeywell International, Inc., for
the U.S. Department of Energy's National Nuclear Security Administration
under contract DE-NA-0003525.

# References
