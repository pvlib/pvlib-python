.. _contributing_new_code:

How to contribute new code
==========================

The basics
~~~~~~~~~~

Contributors to pvlib-python use GitHub's pull requests to add/modify
its source code. The GitHub pull request process can be intimidating for
new users, but you'll find that it becomes straightforward once you use
it a few times. Please let us know if you get stuck at any point in the
process. Here's an outline of the process:

#. Create a GitHub issue and get initial feedback from users and
   maintainers. If the issue is a bug report, please include the
   code needed to reproduce the problem.
#. Obtain the latest version of pvlib-python: Fork the pvlib-python
   project to your GitHub account, ``git clone`` your fork to your computer.
#. Make some or all of your changes/additions and ``git commit`` them to
   your local repository.
#. Share your changes with us via a pull request: ``git push`` your
   local changes to your GitHub fork, then go to GitHub make a pull
   request.

The Pandas project maintains an excellent `contributing page
<http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_ that goes
into detail on each of these steps. Also see GitHub's `Set Up Git
<https://help.github.com/articles/set-up-git/>`_ and `Using Pull
Requests <https://help.github.com/articles/using-pull-requests/>`_.

You must include documentation and unit tests for any new or improved
code. We can provide help and advice on this after you start the pull
request. See the :ref:`documentation` and :ref:`testing` sections for more
information on these aspects.


.. _pull-requests:

Pull requests (PRs)
~~~~~~~~~~~~~~~~~~~

.. _pull-request-scope:

Scope
-----

This section can be summed up as "less is more".

A pull request can quickly become unmanageable if too many lines are
added or changed. "Too many" is hard to define, but as a rule of thumb,
we encourage contributions that contain less than 50 lines of primary code.
50 lines of primary code will typically need at least 250 lines
of documentation and testing. This is about the limit of what the
maintainers can review on a regular basis.

A pull request can also quickly become unmanageable if it proposes
changes to the API in order to implement another feature. Consider
clearly and concisely documenting all proposed API changes before
implementing any code. Modifying
`api.rst <https://github.com/pvlib/pvlib-python/tree/main/docs/sphinx/source/reference>`_
and/or the latest `whatsnew file <https://github.com/pvlib/pvlib-python/tree/main/docs/sphinx/source/whatsnew>`_
can help formalize this process.

Questions about related issues frequently come up in the process of
addressing implementing code for a pull request. Please try to avoid
expanding the scope of your pull request (this also applies to
reviewers!). We'd rather see small, well-documented additions to the
project's technical debt than see a pull request languish because its
scope expanded beyond what the reviewer community is capable of
processing.

Of course, sometimes it is necessary to make a large pull request. We
only ask that you take a few minutes to consider how to break it into
smaller chunks before proceeding.

pvlib-python contains :ref:`3 "layers" of code <modeling-paradigms>`:
functions, PVSystem/Location, and ModelChain. We recommend that
contributors focus their work on only one or two of those layers in a
single pull request. New models are *not* required to be available to
the higher-level API!


When should I submit a pull request?
------------------------------------

The short answer: anytime.

The long answer: it depends. If in doubt, go ahead and submit. You do
not need to make all of your changes before creating a pull request.
Your pull requests will automatically be updated when you commit new
changes and push them to GitHub.

There are pros and cons to submitting incomplete pull-requests. On the
plus side, it gives everybody an easy way to comment on the code and can
make the process more efficient. On the minus side, it's easy for an
incomplete pull request to grow into a multi-month saga that leaves
everyone unhappy. If you submit an incomplete pull request, please be
very clear about what you would like feedback on and what we should
ignore. Alternatives to incomplete pull requests include creating a
`gist <https://gist.github.com>`_ or experimental branch and linking to
it in the corresponding issue.

The best way to ensure that a pull request will be reviewed and merged in
a timely manner is to:

#. Start by creating an issue. The issue should be well-defined and
   actionable.
#. Ask the `maintainers <https://github.com/orgs/pvlib/people>`_ to tag
   the issue with the appropriate milestone.
#. Make a limited-scope pull request. It can be a lot of work to check all of
   the boxes in `pull request guidelines
   <https://github.com/pvlib/pvlib-python/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_,
   especially for pull requests with a lot of new primary code.
   See :ref:`pull-request-scope`.
#. Tag pvlib community members or ``@pvlib`` when the pull
   request is ready for review. (see :ref:`pull-request-reviews`)


.. _pull-request-reviews:

Pull request reviews
--------------------

The pvlib community and maintainers will review your pull request in a
timely fashion. Please "ping" ``@pvlib`` if it seems that
your pull request has been forgotten at any point in the pull request
process.

Keep in mind that the PV modeling community is diverse and each pvlib
community member brings a different perspective when reviewing code.
Some reviewers bring years of expertise in the sub-field that your code
contributes to and will focus on the details of the algorithm. Other
reviewers will be more focused on integrating your code with the rest of
pvlib, ensuring that it is feasible to maintain, that it meets the
:ref:`code style <code-style>` guidelines, and that it is
:ref:`comprehensively tested <testing>`. Limiting the scope of the pull
request makes it much more likely that all of these reviews can be
conducted and any issues can be resolved in a timely fashion.

Sometimes it's hard for reviewers to be immediately available, so the
right amount of patience is to be expected. That said, interested
reviewers should do their best to not wait until the last minute to put
in their two cents.


.. _virtual-environments:

Virtual environments
~~~~~~~~~~~~~~~~~~~~

We strongly recommend using virtual environments for development.
Virtual environments make it easier to switch between different
versions of software. This `scientific-python.org guide
<https://learn.scientific-python.org/development/tutorials/dev-environment/>`_
is a good reference for virtual environments. The pvlib-python
:ref:`installation guide <setupenvironment>` also provides instructions on
setting up a virtual environment. If this is your first pull request, don't
worry about using a virtual environment.
