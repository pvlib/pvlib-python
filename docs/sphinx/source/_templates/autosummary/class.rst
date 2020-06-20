{{ fullname }}
{{ underline }}====================

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}

{# not sure how to get the minigallery directive to not render empty #}
{# galleries, so just use the old `include` style instead #}
{# .. minigallery:: {{ fullname }} #}

.. include:: gallery_backreferences/{{fullname}}.examples
