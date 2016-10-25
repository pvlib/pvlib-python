.. currentmodule:: pvlib

#############
API reference
#############


Classes
=======

pvlib-python provides a collection of classes
for users that prefer object-oriented programming.
These classes can help users keep track of data in a more organized way,
and can help to simplify the modeling process.
The classes do not add any functionality beyond the procedural code.
Most of the object methods are simple wrappers around the
corresponding procedural code.

.. autosummary::
   :toctree: generated/

   pvlib.location.Location
   pvlib.pvsystem.PVSystem
   pvlib.tracking.SingleAxisTracker
   pvlib.modelchain.ModelChain
   pvlib.pvsystem.LocalizedPVSystem
   pvlib.tracking.LocalizedSingleAxisTracker


Solar Position
==============

.. autosummary::
   :toctree: generated/

   pvlib.location.Location.get_solarposition
   pvlib.solarposition
