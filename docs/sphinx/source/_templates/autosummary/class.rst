{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{# not sure how to get the minigallery directive to not render empty #}
{# galleries, so just use the old `include` style instead #}
{# .. minigallery:: {{ fullname }} #}

.. include:: gallery_backreferences/{{fullname}}.examples
