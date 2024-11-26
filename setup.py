from setuptools import setup

setup(name='nuqloud',
      version='1.0',
      description='NUclear-based Quantification of Local Organisation via cellUlar Distributions (nuQLOUD) is a software for the generation of organisational features from point clouds. In short, this tool uses local density around individual cells and features of the 3D Voronoi diagram to characterise local neighbourhoods by the distributions of cells.',
      url='https://github.com/max-brambach/nuQLOUD',
      author='Max Brambach',
      author_email='max.brambach@gmail.com',
      license='MIT',
      packages=['nuqloud'],
      zip_safe=False)
