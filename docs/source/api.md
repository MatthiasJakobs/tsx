# API 

This page documents the classes and methods contained in the `tsx` package.


## Models

```{eval-rst}
.. autoclass:: tsx.models.base.NeuralNetRegressor()
   :members:
.. autoclass:: tsx.models.base.NeuralNetClassifier()
   :members:
```

```{eval-rst}
.. autoclass:: tsx.models.sdt.SoftDecisionTreeClassifier()
   :members:
.. autoclass:: tsx.models.sdt.SoftEnsembleClassifier()
   :members:
.. autoclass:: tsx.models.sdt.SoftDecisionTreeRegressor()
   :members:
.. autoclass:: tsx.models.sdt.SoftEnsembleRegressor()
   :members:
```

```{eval-rst}
.. autoclass:: tsx.models.forecaster.ospgsm.OS_PGSM
   :members:
.. autoclass:: tsx.models.forecaster.ospgsm.OEP_ROC
   :members:
```

## Datasets

### Monash Forecasting Repository
```{eval-rst}
.. autofunction:: tsx.datasets.monash.possible_datasets
.. autofunction:: tsx.datasets.monash.load_monash
```

### Jena Climate Dataset
```{eval-rst}
.. autofunction:: tsx.datasets.jena.load_jena
```

### Utilities
```{eval-rst}
.. autofunction:: tsx.datasets.utils.windowing
.. autofunction:: tsx.datasets.utils.split_horizon
```

## Model selection and ensembling
```{eval-rst}
.. autoclass:: tsx.model_selection.ROC_Member
.. autofunction:: tsx.model_selection.roc_tools.find_best_forecaster
.. autofunction:: tsx.model_selection.roc_tools.find_closest_rocs
```

## Distances
```{eval-rst}
.. autofunction:: tsx.distances.dtw
.. autofunction:: tsx.distances.euclidean
```

## Utilities
```{eval-rst}
.. autofunction:: tsx.utils.to_random_state
```

<!-- ### `tsx.datasets.jena` -->
<!-- ```{eval-rst} -->
<!-- .. automodule:: tsx.datasets.jena -->
<!--    :members: -->
<!--    :undoc-members: -->
<!--    :show-inheritance: -->
<!-- ``` -->


<!-- ```{eval-rst} -->
<!-- .. automodule:: tsx.models.base -->
<!--    :members: -->
<!--    :inherited-members: -->
<!--    :undoc-members: -->
<!--    :show-inheritance: -->
<!-- ``` -->


