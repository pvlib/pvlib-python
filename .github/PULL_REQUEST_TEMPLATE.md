pvlib python pull request guidelines
====================================

Thank you for your contribution to pvlib python!

You may submit a pull request with your code at any stage of completion, however, before the code can be merged the following items must be addressed:

 - [ ] Closes issue #xxxx
 - [ ] Fully tested. Added and/or modified tests to ensure correct behavior for all reasonable inputs. Tests must pass on the TravisCI and Appveyor testing services.
 - [ ] Code quality and style is sufficient. Passes ``git diff upstream/master -u -- "*.py" | flake8 --diff`` and/or landscape.io linting service.
 - [ ] New code is fully documented. Includes sphinx/numpydoc compliant docstrings and comments in the code where necessary.
 - [ ] Updates entries to `docs/sphinx/source/api.rst` for API changes.
 - [ ] Adds description and name entries in the appropriate `docs/sphinx/source/whatsnew` file for all changes.

Please don't hesitate to ask for help if you're unsure of how to accomplish any of the above.
