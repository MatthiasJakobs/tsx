import numpy as np
import torch

from seedpy import fixedseed
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from tsx.distances import dtw, euclidean
from tsx.datasets import windowing
from tsx.model_selection.roc_tools import find_best_forecaster, roc_mean, roc_matrix, find_closest_rocs
from tsx.model_selection import ROC_Member

def _check_compatibility_pool(pool):
    for m in pool:
        is_pytorch = isinstance(m, torch.nn.Module)
        has_feature_extractor = hasattr(m, 'feature_extractor')
        has_forecaster = hasattr(m, 'forecaster')
        if not (is_pytorch and has_feature_extractor and has_forecaster):
            return False
    return True

def concept_drift(residuals, ts_length, R=1):
    if len(residuals) <= 1:
        return False
    
    residuals = np.array(residuals)

    epsilon = np.sqrt((R**2)*np.log(1/0.95) / (2*ts_length))

    if np.abs(residuals[-1]) <= np.abs(epsilon):
        return False
    else:
        return True


def _calc_losses_and_cams(x, pool, context_size, L, apply_roc_mean=False):
    losses = np.zeros((len(pool)))
    if apply_roc_mean:
        all_cams = np.zeros((len(pool), context_size)) 
    else:
        all_cams = np.zeros((len(pool), context_size - L, L))

    X, y = windowing(x, L=L, z=1, use_torch=True)
    for n_m, m in enumerate(pool):

        # New gradcam
        feats = m.feature_extractor(X.unsqueeze(1))
        J = torch.autograd.functional.jacobian(lambda _X: (m.forecaster(_X).squeeze() - y)**2, feats)
        J = torch.einsum('bbjk->bjk', J)
        feats = feats.detach()
        loss = np.sum(((m.forecaster(feats).squeeze() - y)**2).detach().numpy())
        w = torch.mean(J, axis=-1)
        r = torch.nn.functional.relu((feats * w[..., None]).sum(1)).squeeze().numpy()

        if apply_roc_mean:
            r = roc_mean(roc_matrix(r, z=1))

        losses[n_m] = loss
        all_cams[n_m] = r

    return losses, all_cams

def rocs_from_cams(x, y, cams, threshold, min_roc_size):
    rocs = []
    cams_i = cams 

    if len(cams_i.shape) == 1:
        cams_i = np.expand_dims(cams_i, 0)

    for offset, cam in enumerate(cams_i):
        # Normalize CAMs
        max_r = np.max(cam)
        if max_r == 0:
            continue
        normalized = cam / max_r

        # Extract all subseries divided by zeros
        after_threshold = normalized * (normalized > threshold)
        condition = len(np.nonzero(after_threshold)[0]) > 0

        if condition:
            indices = split_zero(after_threshold, min_size=min_roc_size)
            for (f, t) in indices:
                salient_indices = np.arange(f+offset, t+offset+1)
                r = ROC_Member(x.numpy(), y.numpy(), salient_indices)
                rocs.append(r)

    return rocs

def split_zero(arr, min_size=1):
    f, t = 0, 0
    splits = []

    for idx in range(len(arr)):
        if arr[idx] == 0:
            if (t-f) > min_size:
                splits.append((f, t-1))
            f = idx + 1
            t = idx + 1
        else:
            t += 1

    # If last salient segment extends to the end:
    if t == len(arr) and (t-f) > min_size:
        splits.append((f, t-1))

    return splits

class OS_PGSM:

    ''' Improved version of the OS-PGSM algorithm originally presented in Saadallah, A., Jakobs, M., Morik, K. (2021). Explainable Online Deep Neural Network Selection Using Adaptive Saliency Maps for Time Series Forecasting. In: Machine Learning and Knowledge Discovery in Databases. Research Track. ECML PKDD 2021. https://doi.org/10.1007/978-3-030-86486-6_25

    Args:
        pool (List of `nn.Module`): List of pretrained neural network models. Each model must contain a `feature_extractor` and `forecaster` submodule for OS-PGSM to work.
        L (int): The amount of lag used to train the pool models
        context_size (int): Size of the chunks from which Regions of Competence are created
        detect_concept_drift (bool): Whether or not to enable concept drift detection
        threshold (float): Minimum value indicating when a step is salient enough to be part of the Region of Competence
        min_roc_size (int): RoC member smaller than this value are not added to the RoC
        random_state (int): Random state seeding the `run` method
    
    '''
    def __init__(self, pool, L, context_size, detect_concept_drift=True, threshold=0.5, min_roc_size=2, random_state=0):

        if not _check_compatibility_pool(pool):
            raise RuntimeError('One of the models in the pool does not comply with the needed specifications for this method')

        self.pool = pool
        self.L = L
        self.context_size = context_size
        self.detect_concept_drift = detect_concept_drift
        self.random_state = random_state
        self.threshold = threshold
        self.min_roc_size = min_roc_size

    def run(self, X_val, X_test):
        ''' Main method for running the prediction
        
        Args:
            X_val (`torch.Tensor`): Univariate validation time series from which the Regions of Competence are created
            X_test (`torch.Tensor`): Univariate test time series which should be forecasted

        Returns:
            Prediction tensor with the same length as `X_test`
        '''
        with fixedseed(torch, seed=self.random_state):
            self.rebuild_rocs(X_val)
            forecast = self.adaptive_online_forecast(X_val, X_test)
            return forecast

    def adaptive_online_forecast(self, X_val, X_test):
        # Adaptive method from OS-PGSM
        self.test_forecasters = []
        self.drifts_detected = []
        val_start = 0
        val_stop = len(X_val) + self.L
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        residuals = []
        predictions = []
        means = [torch.mean(current_val).numpy()]

        for target_idx in range(self.L, len(X_test)):
            f_test = (target_idx-self.L)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 
  
            # TODO: Only sliding val, since default paramter
            val_start += 1
            val_stop += 1

            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())

            residuals.append(means[-1]-means[-2])

            if self.detect_concept_drift:
                if concept_drift(residuals, len(current_val)):
                    self.drifts_detected.append(target_idx)
                    val_start = val_stop - len(X_val) - self.L
                    current_val = X_complete[val_start:val_stop]
                    residuals = []
                    means = [torch.mean(current_val).numpy()]
                    self.rebuild_rocs(current_val)

            best_model = find_best_forecaster(x, self.rocs, self.pool, dtw)[0]
            self.test_forecasters.append(best_model)

            # TODO: Each model needs to reshape according to their needs. This will only be (batch, features)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            predictions.append(self.pool[best_model].predict(x).squeeze())

        return np.concatenate([X_test[:self.L].numpy(), np.array(predictions)])

    def rebuild_rocs(self, X):
        self.rocs = [ [] for _ in range(len(self.pool)) ]

        x_c, y_c =  windowing(X, L=self.context_size, z=self.context_size, use_torch=True)

        # Create RoCs
        all_models = []
        for x, y in zip(x_c, y_c):
            losses, cams = _calc_losses_and_cams(x, self.pool, self.context_size, self.L, apply_roc_mean=False)
            cams = cams.squeeze()
            best_model = np.argmin(losses)
            all_models.append(best_model)

            # Calculate ROCs from CAMs for best model
            rocs = rocs_from_cams(x, y, cams[best_model], self.threshold, self.min_roc_size) 
            if len(rocs) > 0:
                self.rocs[best_model].extend(rocs)

        # Sanity check
        if np.all([len(roc) == 0 for roc in self.rocs]):
            raise RuntimeError('All Regions of Competence are empty. Predictions will always be NaN')

    def forecast_on_test(self, x_test):
        self.test_forecasters = []
        predictions = np.zeros_like(x_test)

        x = x_test[:self.L]
        predictions[:self.L] = x

        for x_i in range(self.L, len(x_test)):
            x = x_test[x_i-self.L:x_i].unsqueeze(0)

            # Find top-m model who contain the smallest distances to x in their RoC
            best_models = find_best_forecaster(x, self.rocs, self.pool, dtw)

            self.test_forecasters.append([int(m) for m in best_models])
            for i in range(len(best_models)):
                predictions[x_i] += self.pool[best_models[i]].predict(x.unsqueeze(0))

            predictions[x_i] = predictions[x_i] / len(best_models)


        return np.array(predictions)

class OEP_ROC:

    ''' Improved version of the OEP-ROC algorithm originally presented in Saadallah, A., Jakobs, M. & Morik, K. Explainable online ensemble of deep neural network pruning for time series forecasting. Mach Learn 111, 3459–3487 (2022)

    Args:
        pool (List of `nn.Module`): List of pretrained neural network models. Each model must contain a `feature_extractor` and `forecaster` submodule for OS-PGSM to work.
        L (int): The amount of lag used to train the pool models
        context_size (int): Size of the chunks from which Regions of Competence are created
        context_step (int): Step size between the `context_size` chunks
        nr_clusters_ensemble (int): How many cluster centers to use
        dist_fn (callable): A distance function defined on two time series windows
        detect_concept_drift (bool): Whether or not to enable concept drift detection
        threshold (float): Minimum value indicating when a step is salient enough to be part of the Region of Competence
        min_roc_size (int): RoC member smaller than this value are not added to the RoC
        random_state (int): Random state seeding the `run` method
    
    '''
    def __init__(self, pool, L, context_size, context_step, nr_clusters_ensemble=15, dist_fn=euclidean, detect_concept_drift=True, threshold=0.1, min_roc_size=2, random_state=0):

        if not _check_compatibility_pool(pool):
            raise RuntimeError('One of the models in the pool does not comply with the needed specifications for this method')

        self.pool = pool
        self.L = L
        self.context_size = context_size
        self.min_roc_size = min_roc_size
        self.context_step = context_step
        self.detect_concept_drift = detect_concept_drift
        self.threshold = threshold
        self.nr_clusters_ensemble = nr_clusters_ensemble
        self.dist_fn = dist_fn

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def run(self, X_val, X_test):
        ''' Main method for running the prediction
        
        Args:
            X_val (`torch.Tensor`): Univariate validation time series from which the Regions of Competence are created
            X_test (`torch.Tensor`): Univariate test time series which should be forecasted

        Returns:
            Prediction tensor with the same length as `X_test`
        '''
        with fixedseed(torch, seed=self.random_state):
            self.rebuild_rocs(X_val)
            forecast = self.adaptive_monitor_min_distance(X_val, X_test)
            return forecast

    def rebuild_rocs(self, X):
        self.rocs = [ [] for _ in range(len(self.pool)) ]

        x_c, y_c =  windowing(X, L=self.context_size+1, z=self.context_step, use_torch=True)

        # Create RoCs
        all_models = []
        for x, y in zip(x_c, y_c):
            losses, cams = _calc_losses_and_cams(x, self.pool, self.context_size, self.L, apply_roc_mean=True)
            cams = cams.squeeze()
            best_model = np.argmin(losses)
            all_models.append(best_model)

            # Calculate all ROCs from all CAMs
            for i in range(len(cams)):
                rocs = rocs_from_cams(x, y, cams[i], self.threshold, self.min_roc_size) 
                if len(rocs) > 0:
                    # Reject all samples that are not equal to length self.L
                    self.rocs[i].extend([rm for rm in rocs if len(rm.r) == self.L])

        # Sanity check
        if np.all([len(roc) == 0 for roc in self.rocs]):
            raise RuntimeError('All Regions of Competence are empty. Predictions will always be NaN')

    def cluster_rocs(self, best_models, clostest_rocs, nr_desired_clusters, dist_fn):
        dist_fn_dict = {
           'euclidean': euclidean, 
           'dtw': dtw, 
        }

        if nr_desired_clusters == 1:
            return best_models, clostest_rocs

        new_closest_rocs = []

        # Cluster into the desired number of left-over models.
        tslearn_formatted = to_time_series_dataset([r.r for r in clostest_rocs])
        metric = [k for k,v in dist_fn_dict.items() if v == dist_fn][0]
        km = TimeSeriesKMeans(n_clusters=nr_desired_clusters, metric=metric, random_state=self.rng)
        C = km.fit_predict(tslearn_formatted)
        C_count = np.bincount(C)

        # Final model selection
        G = []

        for p in range(len(C_count)):
            # Under all cluster members, find the one maximizing distance to current point
            cluster_member_indices = np.where(C == p)[0]
            # Since the best_models (and closest_rocs) are sorted by distance to x (ascending), 
            # choosing the first one will always minimize distance
            if len(cluster_member_indices) > 0:
                idx = cluster_member_indices[0]
                G.append(best_models[idx])
                new_closest_rocs.append(clostest_rocs[idx])

        return G, new_closest_rocs

    def select_topm(self, models, rocs, x, upper_bound):
        # Select top-m until their distance is outside of the upper bounds
        topm_models = []
        topm_rocs = []
        distances_to_x = np.zeros((len(rocs)))
        for idx, r in enumerate(rocs):
            distance_to_x = self.dist_fn(r.r, x)
            distances_to_x[idx] = distance_to_x

            if distance_to_x <= upper_bound:
                topm_models.append(models[idx])
                topm_rocs.append(r)

        return topm_models, topm_rocs

    def recluster_and_reselect(self, x, iteration):
        # Find closest time series in each models RoC to x
        models, rocs = find_closest_rocs(x, self.rocs, dist_fn=self.dist_fn)

        # Cluster all RoCs into nr_clusters_ensemble clusters
        models, rocs = self.cluster_rocs(models, rocs, self.nr_clusters_ensemble, self.dist_fn)
        self.clustered_rocs.append((iteration, models, rocs))

        # Calculate upper and lower bound of delta (radius of circle)
        numpy_regions = [r.r for r in rocs]
        lower_bound = 0.5 * np.sqrt(np.sum((numpy_regions - np.mean(numpy_regions, axis=0))**2) / self.nr_clusters_ensemble)
        upper_bound = np.sqrt(np.sum((numpy_regions - x.numpy())**2) / self.nr_clusters_ensemble)
        assert lower_bound <= upper_bound, "Lower bound bigger than upper bound"

        # Select topm models according to upper bound
        selected_models, selected_rocs = self.select_topm(models, rocs, x, upper_bound)
        self.topm_selection.append((iteration, selected_models, selected_rocs))
        return selected_models, selected_rocs

    def adaptive_monitor_min_distance(self, X_val, X_test):
        self.test_forecasters = []
        self.clustered_rocs = []
        self.topm_selection = []
        self.drifts_type_1_detected = []
        self.drifts_type_2_detected = []

        predictions = []
        mean_residuals = []
        means = []
        ensemble_residuals = []
        topm_buffer = []

        val_start = 0
        val_stop = len(X_val) + self.L
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        means = [torch.mean(current_val).numpy()]

        # --- First iteration ---
        # -----------------------
        x = X_test[:self.L]
        x_unsqueezed = x.unsqueeze(0).unsqueeze(0)
        
        # Cluster RoCs and select top-m
        topm, _ = self.recluster_and_reselect(x, self.L)

        # Compute min distance to x from the all models needed for Drift Type II
        _, closest_rocs = find_closest_rocs(x, self.rocs, dist_fn=self.dist_fn)
        mu_d = np.min([self.dist_fn(r.r, x) for r in closest_rocs])

        # If new selection is empty, keep the old one
        if len(topm) != 0:
            topm_buffer = topm

        # In the first iteration, topm_buffer can be empty, so we might need to sample a start ensemble
        if len(topm_buffer) == 0:
            topm_buffer = self.rng.choice(len(self.pool), size=min(3, len(self.pool)), replace=False).tolist()

        # Do the actual prediction with the models in topm_buffer
        ensemble_prediction = np.mean([self.pool[i].predict(x_unsqueezed) for i in topm_buffer])
        predictions.append(ensemble_prediction)
        self.test_forecasters.append(topm_buffer)

        # --- Subsequent iterations ---
        for target_idx in range(self.L+1, len(X_test)):
            f_test = (target_idx-self.L)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 
            x_unsqueezed = x.unsqueeze(0).unsqueeze(0)
            val_start += 1
            val_stop += 1
            
            # Residuals for Drift Type I detection
            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())
            mean_residuals.append(means[-1]-means[-2])

            # Compute min distance to x from the all models needed for Drift Type II
            _, closest_rocs = find_closest_rocs(x, self.rocs, dist_fn=self.dist_fn)
            ensemble_residuals.append(mu_d - np.min([self.dist_fn(r.r, x) for r in closest_rocs]))

            drift_type_one = concept_drift(mean_residuals, len(current_val), R=1.5)
            drift_type_two = concept_drift(ensemble_residuals, len(current_val), R=100)

            if drift_type_one and self.detect_concept_drift:
                # If Drift Type I, recreate Regions of Competence
                self.drifts_type_1_detected.append(target_idx)
                val_start = val_stop - len(X_val) - self.L
                current_val = X_complete[val_start:val_stop]
                mean_residuals = []
                means = [torch.mean(current_val).numpy()]
                self.rebuild_rocs(current_val)

            if (drift_type_one or drift_type_two) and self.detect_concept_drift:
                # No matter which Drift Type triggered: Redo ensembling
                topm, _ = self.recluster_and_reselect(x, target_idx)
                
                # If Drift Type II, reset residuals and recompute mu_d
                if drift_type_two:
                    self.drifts_type_2_detected.append(target_idx)
                    _, closest_rocs = find_closest_rocs(x, self.rocs, dist_fn=self.dist_fn)
                    mu_d = np.min([self.dist_fn(r.r, x) for r in closest_rocs])
                    ensemble_residuals = []

                # If new selection is empty, keep the old one
                if len(topm) != 0:
                    topm_buffer = topm

            # Do the actual prediction with the models in topm_buffer
            ensemble_prediction = np.mean([self.pool[i].predict(x_unsqueezed) for i in topm_buffer])
            predictions.append(ensemble_prediction)
            self.test_forecasters.append(topm_buffer)

        return np.concatenate([X_test[:self.L].numpy(), np.array(predictions)])