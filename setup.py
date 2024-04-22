#!/usr/bin/env python

from distutils.core import setup

import giatools


setup(
    name = 'giatools',
    version = giatools.VERSION,
    description = 'Tools required for Galaxy Image Analysis',
    author = 'Leonid Kostrykin',
    author_email = 'leonid.kostrykin@bioquant.uni-heidelberg.de',
    url = 'https://kostrykin.com',
    license = 'MIT',
    packages = ['giatools'],
)
