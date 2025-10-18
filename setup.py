#!/usr/bin/env python3

import os
from setuptools import setup, find_packages, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mtgeo",
    description="The most fundamental geometric modules in Python for Minh-Tri Pham",
    author="Minh-Tri Pham",
    packages=find_packages() + find_namespace_packages(include=["mt.*"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "mtbase>=4.33.2",  # for basic functionalities
        "mtglm>=0.1",  # to have svd2() and svd3()
        "mttf",  # some functions use tensorflow and tensorflow-graphics tensors
        #'shapely', # aarch64 does not have libgeos-dev (deb) or geos-devel (yum) prebuilt yet so we can't run shapely on tx2 for now
    ],
    url="https://github.com/inteplus/mtgeo",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.geo/mt.geo.html",
        "Source Code": "https://github.com/inteplus/mtgeo",
    },
    setup_requires=["setuptools-git-versioning<2"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)
