#!/usr/bin/env python

import re
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


for line in open('pandas_sklearn/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()

setup(name='pandas2sklearn',
      version=__version__,
      description='An integration of pandas dataframes with scikit learn.',
      maintainer='Mourad Mourafiq',
      maintainer_email='mouradmourafiq@gmail.com',
      url='https://github.com/mouradmourafiq/pandas2sklearn',
      license='MIT',
      platforms='any',
      packages=['pandas_sklearn'],
      keywords=['scikit', 'sklearn', 'pandas'],
      install_requires=[
          'pandas>=0.15.0',
          'numpy>=1.9.2'
      ])
