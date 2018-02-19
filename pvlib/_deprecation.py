"""Matplotlib license for the deprecation module.

License agreement for matplotlib versions 1.3.0 and later
=========================================================

1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and
otherwise using matplotlib software in source or binary form and its
associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that MDT's
License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012- Matplotlib Development Team; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib .

4. MDT is making matplotlib available to Licensee on an "AS
IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between MDT and
Licensee.  This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib ,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.

License agreement for matplotlib versions prior to 1.3.0
========================================================

1. This LICENSE AGREEMENT is between John D. Hunter ("JDH"), and the
Individual or Organization ("Licensee") accessing and otherwise using
matplotlib software in source or binary form and its associated
documentation.

2. Subject to the terms and conditions of this License Agreement, JDH
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that JDH's
License Agreement and JDH's notice of copyright, i.e., "Copyright (c)
2002-2011 John D. Hunter; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib  or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib.

4. JDH is making matplotlib  available to Licensee on an "AS
IS" basis.  JDH MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, JDH MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. JDH SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between JDH and
Licensee.  This License Agreement does not grant permission to use JDH
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.
"""

import functools
import warnings


class PVLibDeprecationWarning(UserWarning):
    """A class for issuing deprecation warnings for pvlib users.

    In light of the fact that Python builtin DeprecationWarnings are ignored
    by default as of Python 2.7 (see link below), this class was put in to
    allow for the signaling of deprecation, but via UserWarnings which are not
    ignored by default.

    https://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x
    """

    pass


pvlibDeprecation = PVLibDeprecationWarning


def _generate_deprecation_message(since, message='', name='',
                                  alternative='', pending=False,
                                  obj_type='attribute',
                                  addendum=''):

    if not message:

        if pending:
            message = (
                'The {} {} will be deprecated in a '
                'future version.'.format(name, obj_type))
        else:
            message = (
                'The {} {} was deprecated in version '
                '{}.'.format(name, obj_type, since))

    altmessage = ''
    if alternative:
        altmessage = ' Use {} instead.'.format(alternative)

    message = message + altmessage

    if addendum:
        message += addendum

    return message


def warn_deprecated(since, message='', name='', alternative='', pending=False,
                    obj_type='attribute', addendum=''):
    """Display deprecation warning in a standard way.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.

    message : str, optional
        Override the default deprecation message.  The format
        specifier `%(name)s` may be used for the name of the function,
        and `%(alternative)s` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.  `%(obj_type)s` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated object.

    alternative : str, optional
        An alternative function that the user may use in place of the
        deprecated function.  The deprecation warning will tell the user
        about this alternative if provided.

    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.

    obj_type : str, optional
        The object type being deprecated.

    addendum : str, optional
        Additional text appended directly to the final message.

    Examples
    --------
        Basic example::

            # To warn of the deprecation of "pvlib.name_of_module"
            warn_deprecated('0.6.0', name='pvlib.name_of_module',
                            obj_type='module')

    """
    message = _generate_deprecation_message(since, message, name, alternative,
                                            pending, obj_type)

    warnings.warn(message, pvlibDeprecation, stacklevel=1)


def deprecated(since, message='', name='', alternative='', pending=False,
               obj_type=None, addendum=''):
    """Mark a function or a class as deprecated.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is
        required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier `%(name)s` may be used for the name of the object,
        and `%(alternative)s` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        object.  `%(obj_type)s` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated object; if not provided the name
        is automatically determined from the passed in object,
        though this is useful in the case of renamed functions, where
        the new function is just assigned to the name of the
        deprecated function.  For example::

            def new_function():
                ...
            oldFunction = new_function

    alternative : str, optional
        An alternative object that the user may use in place of the
        deprecated object.  The deprecation warning will tell the user
        about this alternative if provided.

    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.

    addendum : str, optional
        Additional text appended directly to the final message.

    Examples
    --------
        Basic example::

            @deprecated('1.4.0')
            def the_function_to_deprecate():
                pass

    """
    def deprecate(obj, message=message, name=name, alternative=alternative,
                  pending=pending, addendum=addendum):
        import textwrap

        if not name:
            name = obj.__name__

        if isinstance(obj, type):
            obj_type = 'class'
            old_doc = obj.__doc__
            func = obj.__init__

            def finalize(wrapper, new_doc):
                try:
                    pass
                    # obj.__doc = new_doc
                except (AttributeError, TypeError):
                    # cls.__doc__ is not writeable on Py2.
                    # TypeError occurs on PyPy
                    pass
                obj.__init__ = wrapper
                return obj
        else:
            obj_type = 'function'
            if isinstance(obj, classmethod):
                func = obj.__func__
                old_doc = func.__doc__

                def finalize(wrapper, new_doc):
                    wrapper = functools.wraps(func)(wrapper)
                    # wrapper.__doc__ = new_doc
                    return classmethod(wrapper)
            else:
                func = obj
                old_doc = func.__doc__

                def finalize(wrapper, new_doc):
                    wrapper = functools.wraps(func)(wrapper)
                    # wrapper.__doc__ = new_doc
                    return wrapper

        message = _generate_deprecation_message(since, message, name,
                                                alternative, pending,
                                                obj_type, addendum)

        def wrapper(*args, **kwargs):
            warnings.warn(message, pvlibDeprecation, stacklevel=2)
            return func(*args, **kwargs)

        old_doc = textwrap.dedent(old_doc or '').strip('\n')
        message = message.strip()
        new_doc = ('\n.. deprecated:: {}'
                   '\n    {}\n\n'.format(since, message) + old_doc)
        if not old_doc:
            # This is to prevent a spurious 'unexected unindent' warning from
            # docutils when the original docstring was blank.
            new_doc += r'\ '

        return finalize(wrapper, new_doc)

    return deprecate
