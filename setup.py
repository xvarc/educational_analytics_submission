from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A simple ML model that predicts whether a student is likely to pass a test or not.',
    author='Xan',
    license='MIT',
    include_package_data=True,
    install_requires=requirements
)