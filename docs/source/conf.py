# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
from recommonmark.transform import AutoStructify
import os
import sys

# Path for autodocs
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'clip_bbox'
copyright = '2023, Grace Su'
author = 'Grace Su'
release = "0.3.0"

master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['recommonmark', 'sphinx.ext.autodoc', 'sphinx.ext.githubpages', 'sphinx.ext.napoleon']
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = []


def setup(app):
    app.add_config_value(
        'recommonmark_config',
        {
            'auto_toc_tree_section': 'Contents',
        },
        True,
    )
    app.add_transform(AutoStructify)
