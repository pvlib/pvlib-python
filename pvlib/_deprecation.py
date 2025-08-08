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

# modified from Matplotlib b97cd2d (post 2.2.2) in the following ways:
# 1. use module-level _projectName = 'pvlib' and
#    _projectWarning = 'pvlibDeprecationWarning' in place of MPL specific
#    string/Class.
# 2. remove keyword only argument requirement for removal
# 3. remove deprecated obj_type from deprecated function
# 4. if removal is empty, say 'soon' instead of assuming two minor releases
#    later.

import functools
import textwrap
import warnings


class pvlibDeprecationWarning(UserWarning):
    """A class for issuing deprecation warnings for pvlib users.

    In light of the fact that Python builtin DeprecationWarnings are ignored
    by default as of Python 2.7 (see link below), this class was put in to
    allow for the signaling of deprecation, but via UserWarnings which are not
    ignored by default.

    https://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x
    """

    pass


# make it easier for others to copy paste this code into their projects
_projectName = 'pvlib'
_projectWarning = pvlibDeprecationWarning


def _generate_deprecation_message(
        since, message='', name='', alternative='', pending=False,
        obj_type='attribute', addendum='', removal=''):

    if removal == "":
        removal = "soon"
    elif removal:
        if pending:
            raise ValueError(
                "A pending deprecation cannot have a scheduled removal")
        removal = "in {}".format(removal)

    if not message:
        message = (
            "The %(name)s %(obj_type)s"
            + (" will be deprecated in a future version"
               if pending else
               (" was deprecated in %(projectName)s %(since)s"
                + (" and will be removed %(removal)s"
                   if removal else
                   "")))
            + "."
            + (" Use %(alternative)s instead." if alternative else "")
            + (" %(addendum)s" if addendum else ""))

    return message % dict(
        func=name, name=name, obj_type=obj_type, since=since, removal=removal,
        alternative=alternative, addendum=addendum, projectName=_projectName)


def warn_deprecated(
        since, message='', name='', alternative='', pending=False,
        obj_type='attribute', addendum='', removal=''):
    """
    Used to display deprecation in a standard way.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.
    message : str, optional
        Override the default deprecation message.  The format
        specifier ``%(name)s`` may be used for the name of the function,
        and ``%(alternative)s`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.  `%(obj_type)s` may be used to insert a friendly name
        for the type of object being deprecated.
    name : str, optional
        The name of the deprecated object.
    alternative : str, optional
        An alternative API that the user may use in place of the deprecated
        API.  The deprecation warning will tell the user about this alternative
        if provided.
    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.  Cannot be used together with *removal*.
    removal : str, optional
        The expected removal version.  With the default (an empty string), a
        removal version is automatically computed from *since*.  Set to other
        Falsy values to not schedule a removal date.  Cannot be used together
        with *pending*.
    obj_type : str, optional
        The object type being deprecated.
    addendum : str, optional
        Additional text appended directly to the final message.

    Examples
    --------
    Basic example:

    >>> # To warn of the deprecation of "pvlib.name_of_module"
    >>> warn_deprecated('1.4.0', name='pvlib.name_of_module',
    >>>                 obj_type='module')
    """
    message = '\n' + _generate_deprecation_message(
        since, message, name, alternative, pending, obj_type, addendum,
        removal=removal)
    category = (PendingDeprecationWarning if pending
                else _projectWarning)
    warnings.warn(message, category, stacklevel=2)


def deprecated(since, message='', name='', alternative='', pending=False,
               addendum='', removal=''):
    """
    Decorator to mark a function or a class as deprecated.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is
        required.
    message : str, optional
        Override the default deprecation message.  The format
        specifier ``%(name)s`` may be used for the name of the object,
        and ``%(alternative)s`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        object.
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
        An alternative API that the user may use in place of the deprecated
        API.  The deprecation warning will tell the user about this alternative
        if provided.
    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.  Cannot be used together with *removal*.
    removal : str, optional
        The expected removal version.  With the default (an empty string), a
        removal version is automatically computed from *since*.  Set to other
        Falsy values to not schedule a removal date.  Cannot be used together
        with *pending*.
    addendum : str, optional
        Additional text appended directly to the final message.

    Examples
    --------
    Basic example:

    >>> @deprecated('1.4.0')
    >>> def the_function_to_deprecate():
    >>>     pass
    """

    def deprecate(obj, message=message, name=name, alternative=alternative,
                  pending=pending, addendum=addendum):

        if not name:
            name = obj.__name__

        if isinstance(obj, type):
            obj_type = "class"
            old_doc = obj.__doc__
            func = obj.__init__

            def finalize(wrapper, new_doc):
                obj.__doc__ = new_doc
                obj.__init__ = wrapper
                return obj
        else:
            obj_type = "function"
            if isinstance(obj, classmethod):
                func = obj.__func__
                old_doc = func.__doc__

                def finalize(wrapper, new_doc):
                    wrapper = functools.wraps(func)(wrapper)
                    wrapper.__doc__ = new_doc
                    return classmethod(wrapper)
            else:
                func = obj
                old_doc = func.__doc__

                def finalize(wrapper, new_doc):
                    wrapper = functools.wraps(func)(wrapper)
                    wrapper.__doc__ = new_doc
                    return wrapper

        message = _generate_deprecation_message(
            since, message, name, alternative, pending, obj_type, addendum,
            removal=removal)
        category = (PendingDeprecationWarning if pending
                    else _projectWarning)

        def wrapper(*args, **kwargs):
            warnings.warn(message, category, stacklevel=2)
            return func(*args, **kwargs)

        old_doc = textwrap.dedent(old_doc or '').strip('\n')
        message = message.strip()
        new_doc = (('\n.. deprecated:: %(since)s'
                    '\n    %(message)s\n\n' %
                    {'since': since, 'message': message}) + old_doc)
        if not old_doc:
            # This is to prevent a spurious 'unexected unindent' warning from
            # docutils when the original docstring was blank.
            new_doc += r'\ '

        return finalize(wrapper, new_doc)

    return deprecate


def renamed_kwarg_warning(since, old_param_name, new_param_name, removal=""):
    """
    Decorator to mark a possible keyword argument as deprecated and replaced
    with other name.

    Raises a warning when the deprecated argument is used, and replaces the
    call with the new argument name. Does not modify the function signature.

    .. warning::
        Ensure ``removal`` date with a ``fail_on_pvlib_version`` decorator in
        the test suite.

    .. note::
        Not compatible with positional-only arguments.

    .. note::
        Documentation for the function may updated to reflect the new parameter
        name; it is suggested to add a ``.. versionchanged::`` directive.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.
    old_param_name : str
        The name of the deprecated parameter.
    new_param_name : str
        The name of the new parameter.
    removal : str, optional
        The expected removal version, in order to compose the Warning message.

    Examples
    --------
    >>> @renamed_kwarg_warning("1.4.0", "old_name", "new_name", "1.6.0")
    >>> def some_function(new_name=None):
    >>>     pass
    >>> some_function(old_name=1)
    Parameter 'old_name' has been renamed since 1.4.0. and
    will be removed in 1.6.0. Please use 'new_name' instead.

    >>> @renamed_kwarg_warning("1.4.0", "old_name", "new_name")
    >>> def some_function(new_name=None):
    >>>     pass
    >>> some_function(old_name=1)
    Parameter 'old_name' has been renamed since 1.4.0. and
    will be removed soon. Please use 'new_name' instead.
    """

    def deprecate(func, old=old_param_name, new=new_param_name, since=since):
        def wrapper(*args, **kwargs):
            if old in kwargs:
                if new in kwargs:
                    raise ValueError(
                        f"{func.__name__} received both '{old}' and '{new}', "
                        "which are mutually exclusive since they refer to the "
                        f"same parameter. Please remove deprecated '{old}'."
                    )
                warnings.warn(
                    f"Parameter '{old}' has been renamed since {since}. "
                    f"and will be removed "
                    + (f"in {removal}" if removal else "soon")
                    + f". Please use '{new}' instead.",
                    _projectWarning,
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        wrapper = functools.wraps(func)(wrapper)
        return wrapper

    return deprecate


def renamed_key_items_warning(since, old_to_new_keys_map, removal=""):
    """
    Decorator to mark a possible key item (e.g. ``df["key"]``) of an object as
    deprecated and replaced with other attribute.

    Raises a warning when the deprecated attribute is used, and uses the new
    attribute instead, by wrapping the ``__getattr__`` method of the object.
    See [1]_.

    While this implementation is decorator-like, Python syntax won't allow
    ``@decorator`` for applying it. Two sets of parenthesis are required:
    the first one configures the wrapper and the second one applies it.
    This leaves room for reusability too.

    Code is inspired by [2]_, thou it has been generalized to arbitrary data
    types.

    .. warning::
        Ensure ``removal`` date with a ``fail_on_pvlib_version`` decorator in
        the test suite.

    .. note::
        This works for any object that implements a ``__getitem__`` method,
        such as dictionaries, DataFrames, and other collections.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.
    old_to_new_keys_map : dict
        A dictionary mapping old keys to new keys.
    removal : str, optional
        The expected removal version, in order to compose the Warning message.

    Returns
    -------
    object
        A new object that behaves like the original, but raises a warning
        when accessing deprecated keys and returns the value of the new key.

    Examples
    --------
    >>> dict_obj = {"new_key": "renamed_value", "another_key": "another_value"}
    >>> dict_obj = renamed_key_items_warning(
    ...     "1.4.0", {"old_key": "new_key"}
    ... )(dict_obj)
    >>> dict_obj["old_key"]
    pvlibDeprecationWarning: Please use `new_key` instead of `old_key`. \
    Deprecated since 1.4.0 and will be removed soon.
    'renamed_value'
    >>> isinstance(d, dict)
    True
    >>> type(dict_obj)
    <class 'pvlib._deprecation.DeprecatedKeyItems'>

    >>> dict_obj = {"new_key": "renamed_value", "new_key2": "another_value"}
    >>> dict_obj = renamed_key_items_warning(
    ...     "1.4.0", {"old_key": "new_key", "old_key2": "new_key2"}, "1.6.0"
    ... )(dict_obj)
    >>> dict_obj["old_key2"]
    pvlibDeprecationWarning: Please use `new_key2` instead of `old_key2`. \
    Deprecated since 1.4.0 and will be removed in 1.6.0.
    'another_value'

    You can even chain the decorator to rename multiple keys at once:

    >>> dict_obj = {"new_key1": "value1", "new_key2": "value2"}
    >>> dict_obj = renamed_key_items_warning(
    ...     "0.1.0", {"old_key1": "new_key1"}, "0.2.0"
    ... )(dict_obj)
    >>> dict_obj = renamed_key_items_warning(
    ...     "0.3.0", {"old_key2": "new_key2"}, "0.4.0"
    ... )(dict_obj)
    >>> dict_obj["old_key1"]
    pvlibDeprecationWarning: Please use `new_key1` instead of `old_key1`. \
    Deprecated since 0.1.0 and will be removed in 0.4.0.
    'value1'
    >>> dict_obj["old_key2"]
    pvlibDeprecationWarning: Please use `new_key2` instead of `old_key2`. \
    Deprecated since 0.3.0 and will be removed in 0.4.0.
    'value2'

    Reusing the object wrapper factory:

    >>> dict_obj1 = {"new_key": "renamed_value", "another_key": "another_value"}
    >>> dict_obj2 = {"new_key": "just_another", "yet_another_key": "yet_another_value"}
    >>> wrapper_renames_old_key_to_new_key = renamed_key_items_warning("1.4.0", {"old_key": "new_key"}, "2.0.0")
    >>> new_dict_obj1 = wrapper_renames_old_key_to_new_key(dict_obj1)
    >>> new_dict_obj2 = wrapper_renames_old_key_to_new_key(dict_obj2)
    >>> new_dict_obj1["old_key"]
    <stdin>:1: pvlibDeprecationWarning: Please use `new_key` instead of `old_key`. Deprecated since 1.4.0 and will be removed in 2.0.0.
    'renamed_value'
    >>> new_dict_obj2["old_key"]
    <stdin>:1: pvlibDeprecationWarning: Please use `new_key` instead of `old_key`. Deprecated since 1.4.0 and will be removed in 2.0.0.
    'just_another'

    Notes
    -----
    This decorator does not modify the way you access methods on the original
    type. For example, dictionaries can only be accessed with bracketed
    indexes, ``dictionary["key"]``. After decoration, ``"old_key"`` can only
    be used as follows: ``dictionary["old_key"]``. Both ``dictionary.key`` and
    ``dictionary.old_key`` won't become available after wrapping.

    >>> from pvlib._deprecation import renamed_key_items_warning
    >>> dict_base = {"a": [1]}
    >>> dict_depre = renamed_key_items_warning("0.0.1", {"b": "a"})(dict_base)
    >>> dict_depre["a"]
    [1]
    >>> dict_depre["b"]
    <stdin>:1: pvlibDeprecationWarning: Please use `a` instead of `b`. \
    Deprecated since 0.0.1 and will be removed soon.
    [1]
    >>> dict_depre.a
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: 'DeprecatedKeyItems' object has no attribute 'a'
    >>> dict_depre.b
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: 'DeprecatedKeyItems' object has no attribute 'b'

    On the other hand, ``pandas.DataFrame`` and other types may also expose
    indexes as attributes on the object instance. In a ``DataFrame`` you can
    either use ``df.a`` or ``df["a"]``. An old key ``b`` that maps to ``a``
    through the decorator, can either be accessed with ``df.b`` or ``df["b"]``.

    >>> from pvlib._deprecation import renamed_key_items_warning
    >>> import pandas as pd
    >>> df_base = pd.DataFrame({"a": [1]})
    >>> df_base.a
    0    1
    Name: a, dtype: int64
    >>> df_depre = renamed_key_items_warning("0.0.1", {"b": "a"})(df_base)
    >>> df_depre.a
    0    1
    Name: a, dtype: int64
    >>> df_depre.b
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "...", line 6299, in __getattr__
        return object.__getattribute__(self, name)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AttributeError: 'DeprecatedKeyItems' object has no attribute 'b'

    References
    ----------
    .. [1] `Python docs on __getitem__
       <https://docs.python.org/3/reference/datamodel.html#object.__getitem__>`_
    .. [2] `StackOverflow thread on deprecating dict keys
       <https://stackoverflow.com/questions/54095279/how-to-make-a-dict-key-deprecated>`_
    """  # noqa: E501

    def deprecated(obj, old_to_new_keys_map=old_to_new_keys_map, since=since):
        obj_type = type(obj)

        class DeprecatedKeyItems(obj_type):
            """Handles deprecated key-indexed elements in a collection."""

            def __getitem__(self, old_key):
                if old_key in old_to_new_keys_map:
                    new_key = old_to_new_keys_map[old_key]
                    msg = (
                        f"Please use `{new_key}` instead of `{old_key}`. "
                        f"Deprecated since {since} and will be removed "
                        + (f"in {removal}." if removal else "soon.")
                    )
                    with warnings.catch_warnings():
                        # by default, only first ocurrence is shown
                        # remove limitation to show on multiple uses
                        warnings.simplefilter("always")
                        warnings.warn(
                            msg, category=_projectWarning, stacklevel=2
                        )
                    old_key = new_key
                return super().__getitem__(old_key)

        wrapped_obj = DeprecatedKeyItems(obj)

        wrapped_obj.__class__ = type(
            wrapped_obj.__class__.__name__,
            (DeprecatedKeyItems, obj.__class__),
            {},
        )

        return wrapped_obj

    return deprecated
