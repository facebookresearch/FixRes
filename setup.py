# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from setuptools import setup
from setuptools import find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="Fixing the train-test resolution discrepancy scripts",
    version="1.0",
    description="Script of models from https://arxiv.org/abs/1906.06423",
    author="Facebook AI Research",
    packages=find_packages(),
    install_requires=requirements,
)
