# @Author: Felix Kramer <felix>
# @Date:   2022-06-30T14:21:38+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: conf.py
# @Last modified by:   kramer
# @Last modified time: 08-07-2022

import pathlib
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# locPath = '/home/felix/Documents/Git'
locPath = '/home/kramer/Documents/GitHub'
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(locPath,'go-with-the-flow')))
sys.path.insert(0, os.path.abspath(os.path.join(locPath,'kirchhoff-circuits')))
sys.path.insert(0, os.path.abspath(os.path.join(locPath,'kirchhoff-hydro')))
sys.path.insert(0, os.path.abspath('/home/kramer/anaconda3/lib/python3.7/site-packages/'))
# sys.path.insert(0, os.path.abspath('/home/felix/anaconda3/lib/python3.9/site-packages/'))
# -- Project information -----------------------------------------------------

project = 'kirchhoff-hydro'
copyright = '2022, Felix Kramer'
author = 'Felix Kramer'

master_doc = 'index'
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'recommonmark'
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'
# html_theme = 'sphinxdoc'
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'bizstyle'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# The readme that already exists
readme_path = pathlib.Path('../..').parent.resolve().parent / "README.md"
# We copy a modified version here
readme_target = pathlib.Path('../').parent / "intro.md"

with readme_target.open("w") as outf:
    # Change the title to
    outf.write(
        "\n".join(
            [
                "About hailhydro: ",
            ]
        )
    )
    lines = []
    for line in readme_path.read_text().split("\n"):
        if line.startswith("# "):
            # Skip title, because we now use "Readme"
            # Could also simply exclude first line for the same effect
            continue
        lines.append(line)
    outf.write("\n".join(lines))
