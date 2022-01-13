from setuptools import setup

setup(name='pytorch_pae',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='a pytorch package for training a probabilistic autoencoder',
      url='http://github.com/VMBoehm/PytorchPAE',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='Apache License v2',
      packages=['pytorch_pae']
      )
