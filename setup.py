#!/usr/bin/env python

from setuptools import setup

version = dict()
with open('giatools/version.py') as f:
    exec(f.read(), version)


setup(
    name='giatools',
    version=version['__version__'],
    description='Tools required for Galaxy Image Analysis',
    author='Leonid Kostrykin',
    author_email='leonid.kostrykin@bioquant.uni-heidelberg.de',
    url='https://kostrykin.com',
    license='MIT',
    packages=['giatools', 'giatools.io', 'giatools.io._backends'],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.18',
        'scikit-image>=0.18,<0.27',
        'typing-extensions',
        'tifffile',
        'attrs>=25.4',
    ],
    extras_require={
        'omezarr': ['ome-zarr>=0.12.2,<0.13'],
        'pandas': ['pandas>=1'],
    },
)
