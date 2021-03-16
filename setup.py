#!/usr/bin/env python3

from setuptools import setup, Extension, find_packages, find_namespace_packages
from Cython.Build import cythonize
from mt.geo.version import version

extensions = [
    Extension(
        name="mt.geo.linear2d",
        sources=["mt/geo/linear2d.pyx"],
        language_level=3,
    )
]

setup(
    name='mtgeo',
    version=version,
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages() + find_namespace_packages(include=['mt.*']),
    ext_modules=cythonize(extensions),
    package_data={
        'mt.geo': ['*.pyx'],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'mtbase>=1.0.8',
        #'shapely', # aarch64 does not have libgeos-dev (deb) or geos-devel (yum) prebuilt yet so we can't run shapely on tx2 for now
    ],
    url='https://github.com/inteplus/mtgeo',
    project_urls={
        'Documentation': 'https://mtdoc.readthedocs.io/en/latest/mt.geo/mt.geo.html',
        'Source Code': 'https://github.com/inteplus/mtgeo',
    }
)
