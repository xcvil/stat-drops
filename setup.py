from setuptools import setup, find_packages

setup(name='stat-drops',
      version='1.0',
      description='This package is to process urban rain drops data',
      long_description='Some long stuff.',
      classifiers=[
        'Development Status :: early stage',
        'Programming Language :: Python :: 3',
      ],
      install_requires=['numpy'],
      packages=find_packages(),
      keywords='stat-drops',
      author='Xiaochen Zheng',
      author_email='xzheng.ethz@gmail.com',
      license='Eawag')
