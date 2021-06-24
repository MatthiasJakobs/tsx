import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name='tsx',
      version='0.1',
      description='Library containing code for time series experiments regarding explainability',
      long_description=long_description,
      long_description_content_type='text/markdown',
      keywords=['machine learning', 'explainability', 'time-series', 'pytorch'],
      author='Matthias Jakobs',
      author_email='matthias.jakobs@tu-dortmund.de',
      url='https://github.com/MatthiasJakobs/tsx',
      license='GNU GPLv3',
      packages=setuptools.find_packages(),
      python_requires='>=3.5',)
