#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='geomt',
    version='0.0.5',
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'basemt>=0.0.2',
    ],
)
