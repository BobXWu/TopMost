# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.append("../")
sys.path.append("../../")

project = 'TopMost'
copyright = '2023, Xiaobao Wu'
author = 'Xiaobao Wu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',  # this one is really important
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    #'sphinxcontrib.napoleon',
    'sphinx.ext.autosectionlabel',  # allows referring sections its title, affects `ref`
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    # 'recommonmark',
    # 'sphinx_markdown_tables'
    # 'sphinx.ext.imgconverter',  # for svg image to pdf
    # 'sphinxcontrib.inkscapeconverter',
]

autoapi_type = 'python'
autoapi_dirs = ['../../topmost']
autoapi_template_dir = '_autoapi_templates'
autoapi_python_class_content = 'both' 

add_module_names = False  # makes Sphinx render package.module.Class as Class


# Add more mapping for 'sphinx.ext.intersphinx'
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'PyTorch': ('http://pytorch.org/docs/master/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/dev/', None)}

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True


# for 'sphinxcontrib.bibtex' extension
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'

autodoc_mock_imports = ["numpy", "torch", "torchvision", "pandas"]
autoclass_content = 'both'


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = "furo"
html_favicon = "./_static/topmost-logo.png"
html_favicon_width = '20px'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = {
    # "announcement": """
    #     <a style=\"text-decoration: none; color: white;\"
    #        href=\"https://github.com/sponsors/urllib3\">
    #        <img src=\"/en/latest/_static/favicon.png\"/> Support urllib3 on GitHub Sponsors
    #     </a>
    # """,
    "sidebar_hide_name": True,
    "light_logo": "topmost-logo.png",
    "dark_logo": "topmost-logo.png",
}

# html_logo = "topmost-logo.png"
# html_logo_width = '20px'
