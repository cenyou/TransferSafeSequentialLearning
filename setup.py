from setuptools import find_packages, setup

from tssl import __version__

name = "tssl"
version = __version__
description = "Active Learning framework"

setup(
      name=name,
      version=version,
      packages=find_packages(exclude=["tests"]),
      description=description,
)