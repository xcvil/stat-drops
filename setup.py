from setuptools import setup, find_packages

setup(name='statdrops',
      version='0.0.1',
      description='This package is to process urban rain drops data',
      long_description='Some long stuff.',
      classifiers=[
        'Development Status :: early stage',
        'Programming Language :: Python :: 3',
      ],
      install_requires=['numpy', 'matplotlib', 'scipy'],
      packages=find_packages(),
      keywords='drop size distribution',
      author='Xiaochen Zheng',
      author_email='xzheng.ethz@gmail.com',
      license='Eawag')
