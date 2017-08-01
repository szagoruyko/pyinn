#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = "Manually fused PyTorch NN ops"

setup_info = dict(
    # Metadata
    name='pyinn',
    version=VERSION,
    author='Sergey Zagoruyko',
    author_email='sergey.zagoruyko@enpc.fr',
    url='https://github.com/szagoruyko/pyinn',
    description='Manually fused PyTorch NN ops',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'cupy',
        # 'scikit-cuda',
    ]
)

setup(**setup_info)
