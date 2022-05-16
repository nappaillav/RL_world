# setup.py
from setuptools import setup

setup(
    name='rl_world',
    version='0.1.0',
    packages=['rl_world'],
    license='MIT License',
    long_description=open('readme.md').read(),
    install_requires=open('my_requirements.txt').read()
)
