import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name='tsx',
      version='0.2',
      description='Library containing code for time series experiments regarding explainability',
      long_description=long_description,
      long_description_content_type='text/markdown',
      keywords=['machine learning', 'explainability', 'time-series', 'pytorch'],
      author='Matthias Jakobs',
      author_email='matthias.jakobs@tu-dortmund.de',
      url='https://github.com/MatthiasJakobs/tsx',
      license='GNU GPLv3',
      packages=setuptools.find_packages(),
      install_requires=[
        'torch>=1.13.1',
        'pandas>=1.5.0',
        'matplotlib>=3.6.1',
        'tqdm>=4.64.1',
        'scipy>=1.9.2',
        'shap @ git+https://github.com/MatthiasJakobs/shap',
        'seedpy>=0.3',
        'tslearn>=0.5.2',
        'fastdtw>=0.3.4',
        'h5py>=3.8.0',
        'skorch>=0.12.1'
      ],
      python_requires='>=3.8',
      extras_require={
          'docs': ['myst-parser', 'sphinx-rtd-theme'],
      }
)
