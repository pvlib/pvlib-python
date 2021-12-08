"use strict";

// Source:
// https://github.com/anymail/django-anymail/blob/4c443f5515d1d5269a95cb54cf75057c56a3b150/docs/_static/version-alert.js
// via:
// https://github.com/qucontrol/krotov/blob/969fc980346e6411903de854118c48c51208a810/docs/_static/version-alert.js
// following instructions here:
// https://michaelgoerz.net/notes/showing-a-warning-for-the-latest-documentation-on-readthedocs.html

function warnOnLatestVersion() {

  // The warning text and link is really specific to RTD hosting,
  // so we can just check their global to determine version:
  if (!window.READTHEDOCS_DATA || window.READTHEDOCS_DATA.version !== "latest") {
    return;  // not latest, or not on RTD
  }

  var warning = document.createElement('div');
  warning.setAttribute('class', 'admonition danger');
  warning.innerHTML = "<p class='first admonition-title'>Note</p> " +
    "<p class='last'> " +
    "This document is for an <strong>unreleased development version</strong>. " +
    "Documentation is available for the <a href='/en/stable/'>current stable release</a>, " +
    "or for older versions through the &ldquo;v:&rdquo; menu at bottom left." +
    "</p>";
  warning.querySelector('a').href = window.location.pathname.replace('/latest', '/stable');

  // modified from original to work better w/ pydata sphinx theme
  var parent = document.querySelector('main') || document.body;
  parent.insertBefore(warning, parent.firstChild);
}

document.addEventListener('DOMContentLoaded', warnOnLatestVersion);
