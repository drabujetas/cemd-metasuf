from setuptools import setup, find_packages 
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cemd_metasurf',
    packages=find_packages(include=['cemd_metasurf','cemd_metasurf.depolarization_gf','cemd_metasurf.reflec_transm','cemd_metasurf.polarizability']),
    version='0.1.2',
    description='Optical properties of metasurfaces made of dipolar particles',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cemd-metasuf.readthedocs.io/",
    author='Diego Romero Abujetas',
    author_email="drabujetas@gmail.com",
    license='MIT',
    install_requires=['numpy'],
)
