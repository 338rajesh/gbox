# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from os.path import dirname

project = 'gbox'
copyright = '2023, Rajesh Nakka'
author = 'Rajesh Nakka'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []


sys.path.insert(0, dirname(dirname(dirname(__file__))))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autodoc_member_order = 'bysource'  # to avoid the sorting of methods in the alphabetical order

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

html_theme = 'pydata_sphinx_theme'
# Available themes
# sphinx_rtd_theme, alabaster, nature, scrolls, pyramid, haiku, pydata_sphinx_theme
html_static_path = []

html_show_sourcelink = False

# pydata sphinx theme options
html_theme_options = {
    "logo": {
        "text": "gbox",
        # "image_dark": "_static/logo-dark.svg",
    }
}

#
autosummary_generate = True
