#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='geomt',
    version='0.1.0',
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages(),
    package_data={
        'geomt': ['*.pyx'],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'cython',
        'basemt>=0.0.2',
    ],
)
