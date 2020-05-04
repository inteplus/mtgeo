#!/usr/bin/env python3

from setuptools import setup, find_packages, find_namespace_packages
from mt.geo.version import VERSION

setup(
    name='geomt',
    version=VERSION,
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages() + find_namespace_packages(include=['mt.*']),
    package_data={
        'mt.geo': ['*.pyx'],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'cython',
        'basemt>=0.3.0',
    ],
)
