#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='simone',
    packages=find_packages(),
    requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'argh', 'openpyxl']
)

