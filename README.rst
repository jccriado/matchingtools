*Effective* is a Python library for doing symbolic calculations in
effective field theory.

It provides the tools to create a Lagrangian and integrate out heavy
fields at the tree level. It also includes functions for applying
customizable transformations (for example, Fierz identities or
simplification using equations of motion of the light fields) to the
effective Lagrangian to simplify it or write it in terms of a chosen
effective operator basis.

Installation
============

To install Effective using `pip`_ do::

  pip install effective

The source can be downloaded from the `GitHub repository`_.
To install Effective from a local copy of the project do::

  python setup.py install

in the top directory.

.. _pip: https://pypi.python.org/pypi/pip/

.. _GitHub repository: https://github.com/jccriado/effective
  
Documentation
=============

Read the documentation at: http://effective.readthedocs.io/en/latest/.

Test
====

To test that Effective is working correctly, you can run the script
examples/simple_example.py. It should produce a text file with the
following content::

  Collected:
    O1phi:
      (2+0j) {MXi^-4.0}(24)kappa()kappa()

    O3phi:
      (-1+0j) {MXi^-4.0}(24)kappa()kappa()

    OD2phi:
      (0.5+0j) {MXi^-4.0}(24)kappa()kappa()

    OD2phic:
      (0.5+0j) {MXi^-4.0}(24)kappa()kappa()

    Ophi4:
      (0.5+0j) {MXi^-2.0}(10)kappa()kappa()

    Ophi6:
      (-1+0j) {MXi^-4.0}(21)lamb()kappa()kappa()

  Rest:


File list
=========

The file file_list.txt in the top directory of the project contains
a list of its files with a short description of each of them.

