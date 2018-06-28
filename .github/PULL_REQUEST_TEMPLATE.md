pvlib python pull request guidelines
====================================

Thank you for your contribution to pvlib python! You may delete all of these instructions except for the list below.

You may submit a pull request with your code at any stage of completion.

The following items must be addressed before the code can be merged. Please don't hesitate to ask for help if you're unsure of how to accomplish any of the items below:

 - [ ] Closes #xxxx
 - [ ] I am familiar with the [contributing guidelines](http://pvlib-python.readthedocs.io/en/latest/contributing.html).
 - [ ] Fully tested. Added and/or modified tests to ensure correct behavior for all reasonable inputs. Tests (usually) must pass on the TravisCI and Appveyor testing services.
 - [ ] Updates entries to `docs/sphinx/source/api.rst` for API changes.
 - [ ] Adds description and name entries in the appropriate `docs/sphinx/source/whatsnew` file for all changes.
 - [ ] Code quality and style is sufficient. Passes ``git diff upstream/master -u -- "*.py" | flake8 --diff``
 - [ ] New code is fully documented. Includes sphinx/numpydoc compliant docstrings and comments in the code where necessary.
 - [ ] Pull request is nearly complete and ready for detailed review.

Brief description of the problem and proposed solution (if not already fully described in the issue linked to above):
