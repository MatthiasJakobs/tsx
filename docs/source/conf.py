# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import tsx

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tsx'
copyright = '2023, Matthias Jakobs'
author = 'Matthias Jakobs'
release = '0.2'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store'
]

add_module_names = True
# autodoc_mock_imports = [
#     'numpy',
#     '_c_utils',
#     'sklearn',
#     'portion',
#     'igraph'
# ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'tsx'
html_static_path = ['_static']
html_theme_options = {
    'logo': {
        'image_light': '_static/tsx_logo_light.svg',
        'image_dark': '_static/tsx_logo_dark.svg',
        'text': 'tsx',
    }
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
