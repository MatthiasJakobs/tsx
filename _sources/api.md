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


