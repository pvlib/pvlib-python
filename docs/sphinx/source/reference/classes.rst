.. currentmodule:: pvlib

Classes
=======

pvlib-python provides a collection of classes for users that prefer
object-oriented programming. These classes can help users keep track of
data in a more organized way, and can help to simplify the modeling
process. The classes do not add any functionality beyond the procedural
code. Most of the object methods are simple wrappers around the
corresponding procedural code. For examples of using these classes, see
the :ref:`pvsystemdoc` and :ref:`modelchaindoc` pages.

.. autosummary::
   :toctree: generated/

   location.Location
   pvsystem.PVSystem
   pvsystem.Array
   pvsystem.FixedMount
   pvsystem.SingleAxisTrackerMount
   modelchain.ModelChain
   modelchain.ModelChainResult
