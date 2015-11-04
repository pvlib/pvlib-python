{%- extends 'python.tpl' -%}
{% block in_prompt %}{% endblock in_prompt %}
{% block input %}{{ cell.source | ipython2python }}
{% endblock input %}
{% block header %}# coding: utf-8
{% endblock header %}
