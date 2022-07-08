# @Author:  Felix Kramer
# @Date:   2021-05-23T23:25:28+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:   kramer
# @Last modified time: 08-07-2022
# @License: MIT



import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hailhydro", # Replace with your own username
    version="0.1.1",
    author="felixk1990",
    author_email="felixuwekramer@protonmail.com",
    description="Collection of tools for computation of flows and and fluxes on Kirchhoff circuits. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felixk1990/kirchhoff-hydro",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
