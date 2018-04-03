from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='matchingtools',

    version='0.2.10',

    description='A library for symbolic Effective Field Theory calculations',
    long_description=long_description,

    url='https://github.com/jccriado/matchingtools',

    author='Juan Carlos Criado Alamo',
    author_email='jccriadoalamo@ugr.es',

    license='MIT',

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'],

    keywords=['effective field theory symbolic tree matching integration'],

    packages=['matchingtools', 'matchingtools.extras']
)
