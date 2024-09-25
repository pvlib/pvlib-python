.. _documentation-and-style-guide:

Documentation and style guide
=============================

.. _code-style:

Code style
~~~~~~~~~~

pvlib python generally follows the `PEP 8 -- Style Guide for Python Code
<https://www.python.org/dev/peps/pep-0008/>`_. Maximum line length for code
is 79 characters.

pvlib python uses a mix of full and abbreviated variable names. See
:ref:`variables_style_rules`. We could be better about consistency.
Prefer full names for new contributions. This is especially important
for the API. Abbreviations can be used within a function to improve the
readability of formulae.

Set your editor to strip extra whitespace from line endings. This
prevents the git commit history from becoming cluttered with whitespace
changes.

Please see :ref:`Documentation` for information specific to documentation
style.

Remove any ``logging`` calls and ``print`` statements that you added
during development. ``warning`` is ok.

We typically use GitHub's
"`squash and merge <https://help.github.com/articles/about-pull-request-merges/#squash-and-merge-your-pull-request-commits>`_"
feature to merge your pull request into pvlib. GitHub will condense the
commit history of your branch into a single commit when merging into
pvlib-python/main (the commit history on your branch remains
unchanged). Therefore, you are free to make commits that are as big or
small as you'd like while developing your pull request.


.. _documentation:

Documentation
~~~~~~~~~~~~~

.. _documentation-style:

Documentation style
-------------------

Documentation must be written in
`numpydoc format <https://numpydoc.readthedocs.io/>`_ format which is rendered
using the `Sphinx Napoleon extension
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

The numpydoc format includes a specification for the allowable input
types. Python's `duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_
allows for multiple input types to work for many parameters. pvlib uses
the following generic descriptors as short-hand to indicate which
specific types may be used:

* dict-like : dict, OrderedDict, pd.Series
* numeric : scalar, np.array, pd.Series. Typically int or float dtype.
* array-like : np.array, pd.Series. Typically int or float dtype.

Parameters that specify a specific type require that specific input type.

Read the Docs will automatically build the documentation for each pull
request. Please confirm the documentation renders correctly by following
the ``docs/readthedocs.org:pvlib-python`` link within the checks
status box at the bottom of the pull request.


.. _building-the-documentation:

Building the documentation
--------------------------

Building the documentation locally is useful for testing out changes to the
documentation's source code without having to repeatedly update a PR and have
Read the Docs build it for you.  Building the docs locally requires installing
pvlib python as an editable library (see :ref:`installation` for instructions).
First, install the ``doc`` dependencies specified in the
``EXTRAS_REQUIRE`` section of
`setup.py <https://github.com/pvlib/pvlib-python/blob/main/setup.py>`_.
An easy way to do this is with::

    pip install pvlib[doc]    # on Mac:  pip install "pvlib[doc]"

Note: Anaconda users may have trouble using the above command to update an
older version of docutils. If that happens, you can update it with ``conda``
(e.g. ``conda install docutils=0.15.2``) and run the above command again.

Once the ``doc`` dependencies are installed, navigate to ``/docs/sphinx`` and
execute::

    make html

Be sure to skim through the output of this command because Sphinx might emit
helpful warnings about problems with the documentation source code.
If the build succeeds, it will make a new directory ``docs/sphinx/build``
with the documentation's homepage located at ``build/html/index.html``.
This file can be opened with a web browser to view the local version
like any other website. Other output formats are available; run ``make help``
for more information.

Note that Windows users need not have the ``make`` utility installed as pvlib
includes a ``make.bat`` batch file that emulates its interface.


.. _example-gallery:

Example Gallery
---------------

The example gallery uses `sphinx-gallery <https://sphinx-gallery.github.io/>`_
and is generated from script files in the
`docs/examples <https://github.com/pvlib/pvlib-python/tree/main/docs/examples>`_
directory.  sphinx-gallery will execute example files that start with
``plot_`` and capture the output.

Here is a starter template for new examples:

.. code-block:: python

    """
    Page Title
    ==========

    A sentence describing the example.
    """

    # %%
    # Explanatory text about the example, what it does, why it does it, etc.
    # Text in the comment block before the first line of code `import pvlib`
    # will be printed to the example's webpage.

    import pvlib
    import matplotlib.pyplot as plt

    plt.scatter([1, 2, 3], [4, 5, 6])
    plt.show()

For more details, see the sphinx-gallery
`docs <https://sphinx-gallery.github.io/stable/syntax.html#embedding-rst>`_.
