#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='geomt',
    version='0.0.1',
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages(),
    install_requires=[
        'numpy'
        'basemt',
    ],
    dependency_links=[
        'https://github.com/inteplus/basemt/tarball/master#egg=0.0.2',
    ]
)
