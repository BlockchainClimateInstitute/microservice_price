import logging

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

import os

# if os.uname().machine != 'arm64':
setup(
    name='bciavm',
    version='1.21.3',
    author='Mike Casale | Blockchain Climate Institute',
    author_email='mike.casale@blockchainclimate.org',
    description='bciAVM is a machine learning pipeline used to predict property prices.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gcode-ai/bciavm',
    download_url = 'https://github.com/gcode-ai/bciavm/archive/v1.21.3.tar.gz',
    install_requires=open('core-requirements.txt').readlines() + open('requirements.txt').readlines()[1:],
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
          'bciavm = bciavm.__main__:cli'
        ]
    },
    data_files=[
                # ('bciavm/data/lookup_table_parquet', [
                #                 'bciavm/data/lookup_table_parquet/lookup_table1.parquet',
                #                 'bciavm/data/lookup_table_parquet/lookup_table2.parquet',
                #                 'bciavm/data/lookup_table_parquet/lookup_table3.parquet',
                #                 ]),
                # ('bciavm/data/dfPricesEpc_parquet', [
                #                 'bciavm/data/dfPricesEpc_parquet/data1.parquet',
                #                 'bciavm/data/dfPricesEpc_parquet/data2.parquet',
                #                 'bciavm/data/dfPricesEpc_parquet/data3.parquet',
                #                 ]),
                ]
)

