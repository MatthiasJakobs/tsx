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
.. autofunction:: tsx.datasets.utils.split_proportion
.. autofunction:: tsx.datasets.utils.global_subsample_train
```

## Model selection and ensembling
```{eval-rst}
.. autoclass:: tsx.model_selection.ROC_Member
    :members:
.. autofunction:: tsx.model_selection.roc_tools.find_best_forecaster
.. autofunction:: tsx.model_selection.roc_tools.find_closest_rocs
.. autoclass:: tsx.model_selection.ADE
    :members:
.. autoclass:: tsx.model_selection.DETS
    :members:
.. autoclass:: tsx.model_selection.KNNRoC
    :members:
.. autoclass:: tsx.model_selection.OMS_ROC
    :members:
```

## Concepts

### Base functions
```{eval-rst}
.. autofunction:: tsx.concepts.n_uniques
.. autofunction:: tsx.concepts.generate_unique_concepts
.. autofunction:: tsx.concepts.generate_all_concepts
.. autofunction:: tsx.concepts.generate_samples
.. autofunction:: tsx.concepts.find_closest_concepts
.. autofunction:: tsx.concepts.get_concept_distributions
```
### TCAV
```{eval-rst}
.. autofunction:: tsx.concepts.get_cavs
.. autofunction:: tsx.concepts.get_tcav
```

## Distances
```{eval-rst}
.. autofunction:: tsx.distances.dtw
.. autofunction:: tsx.distances.euclidean
```

## Metrics
```{eval-rst}
.. autofunction:: tsx.metrics.mase
.. autofunction:: tsx.metrics.entropy
```

## Utilities
```{eval-rst}
.. autofunction:: tsx.utils.to_random_state
.. autofunction:: tsx.utils.get_device
```
