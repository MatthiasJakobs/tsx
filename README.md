# Time series explainability library (TSX)
Collection of methods used to explain different aspects and application settings for time series data.

# Installation
The latest release version of `tsx` can be installed via

```
pip install git+https://github.com/MatthiasJakobs/tsx.git
```

This will only install some dependencies.
Keep in mind that, for full support, you also need to run

```
pip install git+https://github.com/MatthiasJakobs/shap
pip install torch tslearn fastdtw h5py skorch

``

# Documentation
The documentation for the latest release version is available [here](https://matthiasjakobs.github.io/tsx). 
Notice that this code base is heavily work in progress and APIs might change and break.
We are actively maintaining the functionality documented above and try our best to not change the API.
Undocumented functionality, however, should not be relied upon.
If you like to use undocumented code, please make sure to check out a specific commit from Github instead of the latest `dev` version.

