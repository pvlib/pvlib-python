.. _devops:

Development Operations
======================

This page provides information on specific development needs found in the pvlib-python ecosystem. Some specific Python concepts may be used in this section.

Deprecations
------------
Let's start by what is a deprecation: sometimes, a feature in the library is no longer needed because it has been superceded by better altenatives, or because better practices are considred beneficial. In this case, just doing that change (a removal, a rename) will probably be a **breaking change** for a number of users. Both developers and users desire to not get their code broken after a release in normal circumstances. There are a number of approaches to make these changes gradually, so at least there is some time in-between to warn about upcoming changes and to allow users to adapt their code.

There are two ways to warn about upcoming changes:

- Passively, by expecting users to read whatsnew entries, new version announcements on the mailing list, or by keeping a close eye on the repo activity.
- Actively, via raising warnings with specific instructions when any of these deprecated features are used.

While the pros for the latter are almost obvious, there is a main weakness; it imposes a number of extra steps to take and more code to maintain by the developers. This guide strives to close that gap.

pvlib's submodule :py:mod:`pvlib._deprecation` has some utilities to ease the implementation of deprecations.

Deprecation Warnings and Messages
---------------------------------
This is about the ``Exception`` that gets raised, but quickly dismished by the interpreter after logging. They automatically leave a text trace in the output buffer (console) so it can be seen by the user. In code terms, the following line raises a warning:

.. code-block::

    import warnings

    warnings.warn("This feature is deprecated!")

As a general rule, try to be concise on what the problem is and how to fix that. By default, Python will automatically inform about where the issue was found (although that can be modified, again in code terms, by setting a custom ``stacklevel`` in the warning factory).

List of pvlib deprecation helpers
---------------------------------

.. py:module:: pvlib._deprecation

.. currentmodule:: pvlib

.. autosummary::
   :toctree: generated/

   _deprecation.deprecated
   _deprecation.renamed_kwarg_warning
   _deprecation.renamed_key_items_warning

Know your deprecation helper
----------------------------
Remember to import the submodule.

.. code-block::

    from pvlib import _deprecation

.. contents:: Table of Contents
    :local:

Deprecate a function
~~~~~~~~~~~~~~~~~~~~
See :py:func:`pvlib._deprecation.deprecated`.

Rename keyword parameters
~~~~~~~~~~~~~~~~~~~~~~~~~
Applies both to *positional-or-keyword* parameters and *keyword-only* parameters.
You can check out the differences at the `Python docs glossary <https://docs.python.org/3/glossary.html#term-parameter>`_.

See :py:func:`pvlib._deprecation.renamed_kwarg_warning`.

Rename an item from a collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For example, the key an item uses in a dictionary, the column in a ``pandas.DataFrame`` or any key-indexed object. Intended for objects returned by pvlib functions in the public API.

See :py:func:`pvlib._deprecation.renamed_key_items_warning`
