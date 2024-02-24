#!/usr/bin/env python3

from setuptools import setup, find_packages, find_namespace_packages
from mt.geo.version import version

setup(
    name="mtgeo",
    version=version,
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_packages() + find_namespace_packages(include=["mt.*"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "mtbase>=4.22",  # for basic functionalities
        "mtglm",  # for matrix operations in C++
        "mttf",  # some functions use tensorflow and tensorflow-graphics tensors
        #'shapely', # aarch64 does not have libgeos-dev (deb) or geos-devel (yum) prebuilt yet so we can't run shapely on tx2 for now
    ],
    url="https://github.com/inteplus/mtgeo",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.geo/mt.geo.html",
        "Source Code": "https://github.com/inteplus/mtgeo",
    },
)
