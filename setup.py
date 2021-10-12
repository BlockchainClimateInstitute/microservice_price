import logging

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

import os

setup(
    name='bciavm',
    version='1.21.25',
    author='Mike Casale | Blockchain Climate Institute',
    author_email='mike.casale@blockchainclimate.org',
    description='bciAVM is a machine learning pipeline used to predict property prices.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/BlockchainClimateInstitute/microservice_price',
    download_url = 'https://github.com/BlockchainClimateInstitute/microservice_price/archive/bciavm-1.21.25.tar.gz',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'bciavm = bciavm.__main__:cli'
        ]
    },
    data_files=[]
)

