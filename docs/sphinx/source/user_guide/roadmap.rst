.. _roadmap:

pvlib Roadmap
=============

**Last revision: October 2023**

This document outlines a "wish list" for advancing pvlib-python, focusing
on gaps in functionality and other code-related areas of improvement.

Achieving these improvements will depend on community contributions.
Therefore, it's not reasonable to set forth a timeline for this roadmap
to be completed.
For a more concrete plan of short-term additions to pvlib, please see
the `milestone page <https://github.com/pvlib/pvlib-python/milestones>`_
on GitHub.

Additionally, an idea being listed here is no guarantee that it
will ever be added to pvlib, nor is omission from this page an
indication that an idea is not of interest.  This roadmap is an update
of previous roadmaps based on requests from users:

* `2023 pvlib-python user meeting notes
  <https://github.com/pvlib/pvlib-python/wiki/2023-pvlib-python-user-meeting-notes>`_
* `2022 pvlib development roadmap
  <https://github.com/pvlib/pvlib-python/discussions/1581>`_
* `2019 pvlib-python development roadmap
  <https://github.com/pvlib/pvlib-python/wiki/2019-pvlib-python-Development-Roadmap>`_


Core modeling capabilities
--------------------------
The core functionality of pvlib is its library of model functions,
organized by the various conceptual steps of a PV system performance
simulation.  pvlib has robust modeling capability in many of these topic
areas, but coverage in some areas is poor or nonexistent.  Here is a
partial list of modeling areas that could be improved:

* Improved inverter performance models (power factor, off-MPP operation,
  thermal derating, MPPT voltage range)
* DC-DC optimizer performance models
* Transformer models
* Electrical mismatch functionality
* Improved snow loss models, in particular for bifacial and tracked systems
* Shade models
* Degradation
* More sophisticated single-axis tracker models
* Bifacial irradiance models
* Updated parameter libraries, e.g. the CEC PV Module database

Note that published models may not exist for some of these enhancements,
models would need to be developed and published before pvlib can
provide a reference implementation.


Auxiliary modeling capabilities
-------------------------------
Increasingly, pvlib is gaining functionality outside of the typical
modeling workflow, for example parameter estimation/fitting, converting
parameters between similar models, and clear-sky detection.
These kinds of methods are not usually implemented in other
performance modeling software, meaning providing pvlib implementations
is of significant value.  However, making progress here is also somewhat
constrained by the small number of existing published methods that can be
implemented.


System simulation
-----------------
pvlib's :py:class:`~pvlib.pvsystem.PVSystem` and
:py:class:`~pvlib.modelchain.ModelChain` classes connect the individual
model functions together to provide coherent system simulations.
However, not all of pvlib's models are accessible with these classes.
For example:

* Bifacial irradiance models
* Internal, external, and horizon shading
* Soiling loss models

In addition to more comprehensive integration of pvlib's model functions,
the scope of these high-level capabilities could be expanded:

* Multi-inverter systems
* Geographic and electrical coordinate systems, to enable more detailed modeling
* "out-of-code" model specification (e.g. configuration files)


Data I/O
--------
:py:mod:`pvlib.iotools` provides functions to import
 weather datasets, horizon profiles, component specifications,
and other external data.  Expanding this capability by
adding functions for additional data sources is of significant interest.
:py:mod:`pvlib.iotools` would also benefit from further standardization
(``map_variables``, etc) and more documentation.

See :discuss:`1528` for a detailed plan for :py:mod:`pvlib.iotools`.


Speed improvements & benchmarking
---------------------------------
The value of improved computation speed is growing over time
as pvlib is increasingly used in commercial APIs, large-scale
research studies, and high-resolution timeseries simulations.
However, benchmarking and improving the runtime of pvlib's codebase
has received only sporadic and unfocused attention so far.
Establishing a comprehensive set of speed (and potentially memory)
benchmarks would be a good place to start.  Identified bottlenecks
should be reviewed and made more efficient if possible.  If not,
then alternative models could be implemented for cases where
speed is the priority.

As accelerator tools like Numba continue to mature, they should be
evaluated for potential use in pvlib.  Although fast runtime speed is
desirable, it must be weighed against code maintainability and portability.

For modeling topics where runtime can be significant (e.g. solar position),
the relevant User's Guide documentation section could include speed
comparisons.


Documentation
-------------
pvlib's documentation is overdue for a revision, with a focus on
strategic organization.  See :issue:`329` for a specific proposal.

Within that framework, better "getting started" tutorials are needed.
However, the scope should probably stick to pvlib usage.  PV modeling in
general is better left to other projects like the
`PVPMC website <https://pvpmc.sandia.gov/>`_ or the
`conference tutorials <https://pvsc-python-tutorials.github.io/>`_.

The gallery of examples could be significantly expanded as well.
An aspirational goal could be for every public function to be used
in at least one gallery page.

Additionally, it would be nice to have a project website (pvlib.org?)
with scope going beyond code documentation.


API cleanup
-----------
Some parts of pvlib's API should be cleaned up before version 1.0
is eventually released.  Examples of such cleanup include migrating some code
from :py:mod:`pvlib.pvsystem` to :py:mod:`pvlib.pvarray` and renaming
parameters with inconsistent names.

Taking stock of changes needed before 1.0 is the first step.  Then,
actually implementing the changes will likely require deprecation periods.
