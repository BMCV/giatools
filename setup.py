#!/usr/bin/env python

from distutils.core import setup

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
    packages=['giatools', 'giatools.io', 'giatools.io.backends'],
)
