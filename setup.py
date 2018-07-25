from setuptools import setup, find_packages

setup(name='mla_sani',
      version='0.1',
      description='Machine Learning Algorithms - Simple and Naive Implementation',
      url='http://git.sacry.org/mla_sani',
      author='sacry',
      author_email='sacry@sacry.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'matplotlib',
          'seaborn',
          'numpy',
          'sklearn',
          'pandas',
          'scipy',
          'ipython',
          'jupyter',
      ],
      zip_safe=False)
