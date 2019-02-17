"""
Setup of EE 106B Lab1 python codebase
Author: Chris Correa
"""
from setuptools import setup

requirements = []

setup(name='lab1_pkg',
      version='0.1.0',
      description='lab1 package for EE 106B',
      author='Chris Correa',
      author_email='chris.correa@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['paths', 'controllers', 'utils', 'joint_trajectory_action_server'],
      install_requires=requirements,
      test_suite='test'
     )