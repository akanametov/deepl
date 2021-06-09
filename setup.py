import os
import re

from setuptools import setup, find_packages

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

setup(
    name='deepl',
    packages=find_packages(),
    url='https://github.com/akanametov/deepl',
    author='Azamat Kanametov',
    author_email='akkanametov@gmail.com',
    install_requires=['numpy', 'scikit-learn'],
    version='0.1',
    license='MIT',
    description='An example of a python package from pre-existing code',
    include_package_data=True)
