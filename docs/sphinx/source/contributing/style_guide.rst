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
(e.g. ``conda install docutils=0.21``) and run the above command again.

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


.. _example-docstring:

Example Docstring
-----------------

Here is a template for a function docstring that encapsulates the main
features that may be used in any docstring. Note that not all sections are
required for every function.

.. code-block:: python

    def example_function(poa_global, exponents, degree_symbol, time_ref='UT',
                         optional_arg=None):
        r"""
        Brief summary of the function (no citations).

        A longer description of the function. This can include citations
        (references) to literature [1]_, websites [2]_, and other code elements
        such as functions (:py:func:`pvlib.location.lookup_altitude`) and
        classes (:py:class:`pvlib.location.Location`).

        .. versionadded:: 0.0.1
        There are many more purpose-specific directives, admonitions and such
        available at `this link <admonitions>`_. E.g.: ``.. versionchanged::``,
        ``.. deprecated::``,  ``.. note::`` and ``.. warning::``.

        .. _admonitions: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#admonitions-messages-and-warnings

        Parameters
        ----------
        poa_global : numeric
            Plane-of-array global irradiance, see :term:`poa_global`. [Wm⁻²].
        exponents : array-like
            A list of exponents. [x⁰¹²³⁴⁵⁶⁷⁸⁹⁻].
        degree_symbol : pandas.Series or pandas.DataFrame
            It's different from superscript zero. [°].
        time_ref : ``'UT'`` or ``'TST'``, default: ``'UT'``
            ``'UT'`` (universal time) or ``'TST'`` (True Solar Time).
        optional_arg : integer, optional
            A description of ``optional_arg``. [Unitless].

        Returns
        -------
        name : numeric
            A description of the return value.

        Raises
        ------
        ValueError
            If ``poa_global`` is negative.
        KeyError
            If ``time_ref`` does not exist.

        Notes
        -----
        This section can include additional information about the function.

        For example, an equation using LaTeX markup:

        .. math::

            a = \left(\frac{b}{c}\right)^2

        where :math:`a` is the result of the equation, and :math:`b` and :math:`c`
        are inputs.

        Or a figure with a caption:

        .. figure:: ../../_images/pvlib_logo_horiz.png
            :scale: 10%
            :alt: alternate text
            :align: center

            Figure caption.

        See Also
        --------
        pvlib.location.lookup_altitude, pvlib.location.Location

        Examples
        --------
        >>> example_function(1, 1, 1)
        'Something'

        References
        ----------
        A IEEE citation to a relevant reference. You may use an automatic
        citation generator to format the citation correctly.

        .. [1] Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M.,
           and Driesse, A. "pvlib python: 2023 project update." Journal of Open
           Source Software, 8(92), 5994, (2023). :doi:`10.21105/joss.05994`.
        .. [2] J. Smith and J. Doe. "Obama inaugurated as President." CNN.com.
           Accessed: Feb. 1, 2009. [Online.]
           Available: http://www.cnn.com/POLITICS/01/21/obama_inaugurated/index.html
        """
        return "Something"

A preview of how this docstring would render in the documentation can be seen in the
following file: :download:`Example docstring<../_images/example_function_screenshot.png>`.

Remember that to show the docstring in the documentation, you must list
the function in the appropriate ``.rst`` file in the ``docs/sphinx/source/reference`` file.

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
