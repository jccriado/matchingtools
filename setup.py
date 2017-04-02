from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='efttools',

    version='1.5.4',

    description='A library for Effective Field Theory calculations',
    long_description=long_description,

    url='https://github.com/jccriado/efttools',

    author='Juan Carlos Criado Alamo',
    author_email='jccriadoalamo@ugr.es',

    keywords=['effective field theory tree level integration'],

    packages=['efttools', 'efttools.extras']
)
