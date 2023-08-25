import numpy as np
import torch

from seedpy import fixedseed
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from tsx.attribution import simple_gradcam as gradcam
from tsx.metrics import mse
from tsx.distances import dtw, euclidean
from tsx.datasets import windowing
from tsx.model_selection import find_best_forecaster, ROC_Member, roc_mean, roc_matrix, find_closest_rocs

def _check_compatibility_pool(pool):
    for m in pool:
        is_pytorch = isinstance(m, torch.nn.Module)
        has_feature_extractor = hasattr(m, 'feature_extractor')
        has_forecaster = hasattr(m, 'forecaster')
        if not (is_pytorch and has_feature_extractor and has_forecaster):
            return False
    return True

def concept_drift(residuals, ts_length, test_length, drift_type=None, R=1):
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

def split_array_at_zero(arr):
    indices = np.where(arr != 0)[0]
    splits = []
    i = 0
    while i+1 < len(indices):
        start = i
        stop = start
        j = i+1
        while j < len(indices):
            if indices[j] - indices[stop] == 1:
                stop = j
                j += 1
            else:
                break

        if start != stop:
            splits.append((indices[start], indices[stop]))
            i = stop
        else:
            i += 1

    return splits

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

    def __init__(self, models, config, random_state=0):
        self.models = models
        self.config = config
        self.lag = config.get("k", 5)
        self.topm = config.get("topm", 1)
        self.nr_clusters_single = config.get("nr_clusters_single", 1) # Default value: No clustering
        self.threshold = config.get("smoothing_threshold", 0.5)
        self.nr_clusters_ensemble = config.get("nr_clusters_ensemble", 1) # Default value: No clustering
        self.n_omega = config.get("n_omega", self.lag)
        self.z = config.get("z", 1)
        self.invert_relu = config.get("invert_relu", False)
        self.roc_take_only_best = config.get("roc_take_only_best", True)
        self.small_z = config.get("small_z", 1)
        self.delta = config.get("delta", 0.05)
        self.roc_mean = config.get("roc_mean", False)
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state
        self.concept_drift_detection = config.get("concept_drift_detection", None)
        self.drift_type = config.get("drift_type", "ospgsm")
        self.ambiguity_measure = config.get("ambiguity_measure", "euclidean")
        self.skip_drift_detection = config.get("skip_drift_detection", False)
        self.skip_topm = config.get("skip_topm", False)
        self.skip_clustering = config.get("skip_clustering", False)
        self.skip_type1 = config.get("skip_type1", False)
        self.skip_type2 = config.get("skip_type2", False)
        self.distance_measure = config.get("distance_measure", "euclidean")
        self.nr_select = config.get("nr_select", None)
        self.explanation_method = 'gradcam'

    def ensemble_predict(self, x, subset=None):
        if len(x.shape)==2:
            x = x.unsqueeze(0)
        if subset is None:
            predictions = [m.predict(x) for m in self.models]
        else:
            predictions = [self.models[i].predict(x) for i in subset]

        return np.mean(predictions)

    def shrink_rocs(self):
        # Make RoCs more concise by considering cluster centers instead of all RoCs
        if self.nr_clusters_single > 1:
            for i, single_roc in enumerate(self.rocs): 

                # Skip clustering if there would be no shrinking anyway
                if len(single_roc) <= self.nr_clusters_single:
                    continue

                tslearn_formatted = to_time_series_dataset(single_roc)
                km = TimeSeriesKMeans(n_clusters=self.nr_clusters_single, metric="dtw", random_state=self.rng)
                km.fit(tslearn_formatted)

                # Choose cluster centers as new RoCs
                new_roc = km.cluster_centers_.squeeze()
                self.rocs[i] = []
                for roc in new_roc:
                    self.rocs[i].append(torch.tensor(roc).float())

    def rebuild_rocs(self, X):
        self.rocs = [ [] for _ in range(len(self.models))]

        x_c, y_c = self.split_n_omega(X)
        # Create RoCs
        for x, y in zip(x_c, y_c):
            losses, cams = self.evaluate_on_validation(x, y) # Return-shapes: n_models, (n_models, blag-lag, lag)
            best_model = self.compute_ranking(losses) # Return: int [0, n_models]
            all_rocs = self.calculate_rocs(x, cams) # Return: List of vectors
            if self.roc_take_only_best:
                rocs_i = all_rocs[best_model]
                if rocs_i is not None and len(rocs_i) > 0:
                    self.rocs[best_model].extend(rocs_i)
            else:
                for i, rocs_i in enumerate(all_rocs):
                    if rocs_i is not None:
                        self.rocs[i].extend(rocs_i)

        # Sanity check
        if np.all([len(roc) == 0 for roc in self.rocs]):
            raise RuntimeError('All Regions of Competence are empty. Predictions will always be NaN')

    def detect_concept_drift(self, residuals, ts_length, test_length, drift_type=None, R=1):
        if self.skip_drift_detection:
            return False

        if self.concept_drift_detection is None:
            raise RuntimeError("Concept drift should be detected even though config does not specify method", self.concept_drift_detection)

        if self.concept_drift_detection == "periodic":
            if drift_type is None:
                return len(residuals) >= int(test_length / 10.0)
            else:
                return len(residuals) >= 20
        elif self.concept_drift_detection == "hoeffding":
            residuals = np.array(residuals)

            # Empirical range of residuals
            #R = 1
            #R = np.max(np.abs(residuals)) # R = 1 

            epsilon = np.sqrt((R**2)*np.log(1/self.delta) / (2*ts_length))

            if np.abs(residuals[-1]) <= np.abs(epsilon):
                return False
            else:
                return True

    def adaptive_online_roc_rebuild(self, X_val, X_test):
        # Adaptive method from OS-PGSM
        self.test_forecasters = []
        self.drifts_detected = []
        val_start = 0
        val_stop = len(X_val) + self.lag
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        residuals = []
        predictions = []
        means = [torch.mean(current_val).numpy()]

        for target_idx in range(self.lag, len(X_test)):
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 
  
            # TODO: Only sliding val, since default paramter
            val_start += 1
            val_stop += 1

            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())

            residuals.append(means[-1]-means[-2])

            if len(residuals) > 1: 
                if self.detect_concept_drift(residuals, len(current_val), len(X_test)):
                    self.drifts_detected.append(target_idx)
                    val_start = val_stop - len(X_val) - self.lag
                    current_val = X_complete[val_start:val_stop]
                    residuals = []
                    means = [torch.mean(current_val).numpy()]
                    self.rebuild_rocs(current_val)

            best_model = self.find_best_forecaster(x)
            try:
                self.test_forecasters.append(best_model.tolist())
            except:
                self.test_forecasters.append(best_model)

            # TODO: Each model needs to reshape according to their needs. This will only be (batch, features)
            ensemble_prediction = self.ensemble_predict(x.unsqueeze(0), subset=best_model)
            predictions.append(ensemble_prediction)

        return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

    def find_closest_rocs(self, x, rocs):
        closest_rocs = []
        closest_models = []

        for model in range(len(rocs)):
            rs = rocs[model]
            distances = [euclidean(x.squeeze(), r.squeeze()) for r in rs]
            if len(distances) != 0:
                closest_rocs.append(rs[np.argsort(distances)[0]])
                closest_models.append(model)
        return closest_models, closest_rocs

    def select_topm(self, models, rocs, x, upper_bound):
        # Select top-m until their distance is outside of the upper bounds
        topm_models = []
        topm_rocs = []
        distances_to_x = np.zeros((len(rocs)))
        for idx, r in enumerate(rocs):
            distance_to_x = euclidean(r, x)
            distances_to_x[idx] = distance_to_x

            if distance_to_x <= upper_bound:
                topm_models.append(models[idx])
                topm_rocs.append(r)

        return topm_models, topm_rocs

    def roc_rejection_sampling(self):
        for i, model_rocs in enumerate(self.rocs):
            roc_lengths = np.array([len(r) for r in model_rocs])
            length_sample = np.where(roc_lengths == self.lag)[0]
            if len(length_sample) > 0:
                self.rocs[i] = [r for j, r in enumerate(model_rocs) if j in length_sample]
            else:
                self.rocs[i] = []

    def recluster_and_reselect(self, x, iteration):
        # Find closest time series in each models RoC to x
        models, rocs = self.find_closest_rocs(x, self.rocs)

        # Cluster all RoCs into nr_clusters_ensemble clusters
        if not self.skip_clustering:
            models, rocs = self.cluster_rocs(models, rocs, self.nr_clusters_ensemble)
            self.clustered_rocs.append((iteration, models, rocs))

        # Calculate upper and lower bound of delta (radius of circle)
        numpy_regions = [r.numpy() for r in rocs]
        lower_bound = 0.5 * np.sqrt(np.sum((numpy_regions - np.mean(numpy_regions, axis=0))**2) / self.nr_clusters_ensemble)
        upper_bound = np.sqrt(np.sum((numpy_regions - x.numpy())**2) / self.nr_clusters_ensemble)
        assert lower_bound <= upper_bound, "Lower bound bigger than upper bound"

        # Select topm models according to upper bound
        if not self.skip_topm:
            selected_models, selected_rocs = self.select_topm(models, rocs, x, upper_bound)
            self.topm_selection.append((iteration, selected_models, selected_rocs))
            return selected_models, selected_rocs

        if self.nr_select is not None:
            selected_indices = np.argsort([euclidean(x, r) for r in rocs])[:self.nr_select]
            selected_models = np.array(models)[selected_indices]
            selected_rocs = [rocs[idx] for idx in selected_indices]
            self.topm_selection.append((iteration, selected_models, selected_rocs))
            return selected_models, selected_rocs
        return models, rocs 

    def adaptive_monitor_min_distance(self, X_val, X_test):
        def get_topm(buffer, new):
            if len(new) == 0:
                if len(buffer) == 0:
                    # Failsafe if the very first iteration results in a topm_models that is empty
                    return self.rng.choice(len(self.models), size=3, replace=False).tolist()
                return buffer
            return new

        self.length_of_best_roc = []
        self.test_forecasters = []
        self.drifts_type_1_detected = []
        self.drifts_type_2_detected = []

        self.random_selection = []
        self.topm_empty = []

        self.clustered_rocs = []
        self.topm_selection = []
        
        predictions = []
        mean_residuals = []
        means = []
        ensemble_residuals = []
        topm_buffer = []

        val_start = 0
        val_stop = len(X_val) + self.lag
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        # Remove all RoCs that have not length self.lag
        self.roc_rejection_sampling()

        means = [torch.mean(current_val).numpy()]

        # First iteration 
        x = X_test[:self.lag]
        x_unsqueezed = x.unsqueeze(0).unsqueeze(0)
        topm, topm_rocs = self.recluster_and_reselect(x, 5)

        # Compute min distance to x from the all models
        _, closest_rocs = self.find_closest_rocs(x, self.rocs)
        mu_d = np.min([euclidean(r, x) for r in closest_rocs])

        # topm_buffer contains the models used for prediction
        topm_buffer = get_topm(topm_buffer, topm)

        predictions.append(self.ensemble_predict(x_unsqueezed, subset=topm_buffer))
        self.test_forecasters.append(topm_buffer)

        # Subsequent iterations
        for target_idx in range(self.lag+1, len(X_test)):
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 
            x_unsqueezed = x.unsqueeze(0).unsqueeze(0)
            val_start += 1
            val_stop += 1
            
            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())
            mean_residuals.append(means[-1]-means[-2])

            _, closest_rocs = self.find_closest_rocs(x, self.rocs)
            ensemble_residuals.append(mu_d - np.min([euclidean(r, x) for r in closest_rocs]))

            if len(mean_residuals) > 1:
                drift_type_one = self.detect_concept_drift(mean_residuals, len(current_val), len(X_test), drift_type="type1", R=1.5)
            else:
                drift_type_one = False

            if len(ensemble_residuals) > 1:
                drift_type_two = self.detect_concept_drift(ensemble_residuals, len(current_val), len(X_test), drift_type="type2", R=100)
            else:
                drift_type_two = False

            # Allow for skipping either drift detection part
            # drift_type_one = drift_type_one or self.skip_type1
            # drift_type_two = drift_type_one or self.skip_type2

            if drift_type_one:
                self.drifts_type_1_detected.append(target_idx)
                val_start = val_stop - len(X_val) - self.lag
                current_val = X_complete[val_start:val_stop]
                mean_residuals = []
                means = [torch.mean(current_val).numpy()]
                self.rebuild_rocs(current_val)
                self.roc_rejection_sampling()

            if drift_type_one or drift_type_two:
                topm, topm_rocs = self.recluster_and_reselect(x, target_idx)
                if drift_type_two:
                    self.drifts_type_2_detected.append(target_idx)
                    _, closest_rocs = self.find_closest_rocs(x, self.rocs)
                    mu_d = np.min([euclidean(r, x) for r in closest_rocs])
                    ensemble_residuals = []
                topm_buffer = get_topm(topm_buffer, topm)

            predictions.append(self.ensemble_predict(x_unsqueezed, subset=topm_buffer))
            self.test_forecasters.append(topm_buffer)

        return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

    # TODO: No runtime reports
    def run(self, X_val, X_test, reuse_prediction=False):
        with fixedseed(torch, seed=self.random_state):
            self.rebuild_rocs(X_val)
            self.shrink_rocs()        

            self.drifts_detected = []

            if self.concept_drift_detection is None:
                forecast = self.forecast_on_test(X_test, reuse_prediction=reuse_prediction)
            else:
                if self.drift_type == "ospgsm":
                    forecast = self.adaptive_online_roc_rebuild(X_val, X_test)
                elif self.drift_type == "min_distance_change":
                    forecast = self.adaptive_monitor_min_distance(X_val, X_test)
                else:
                    raise NotImplementedError(f"Drift type {self.drift_type} not implemented")

            return forecast

    def cluster_rocs(self, best_models, clostest_rocs, nr_desired_clusters):
        if nr_desired_clusters == 1:
            return best_models, clostest_rocs

        new_closest_rocs = []

        # Cluster into the desired number of left-over models.
        tslearn_formatted = to_time_series_dataset(clostest_rocs)
        km = TimeSeriesKMeans(n_clusters=nr_desired_clusters, metric=self.distance_measure, random_state=self.rng)
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
                #idx = cluster_member_indices[-1]
                idx = cluster_member_indices[0]
                G.append(best_models[idx])
                new_closest_rocs.append(clostest_rocs[idx])

        return G, new_closest_rocs

    # TODO: Make faster
    def forecast_on_test(self, x_test, reuse_prediction=False):
        self.test_forecasters = []
        predictions = np.zeros_like(x_test)

        x = x_test[:self.lag]
        predictions[:self.lag] = x

        for x_i in range(self.lag, len(x_test)):
            if reuse_prediction:
                x = torch.from_numpy(predictions[x_i-self.lag:x_i]).unsqueeze(0)
            else:
                x = x_test[x_i-self.lag:x_i].unsqueeze(0)

            # Find top-m model who contain the smallest distances to x in their RoC
            best_models, closest_rocs = self.find_best_forecaster(x, return_closest_roc=True)

            # Further reduce number of best models by clustering
            best_models, _ = self.cluster_rocs(best_models, closest_rocs, self.nr_clusters_ensemble)

            self.test_forecasters.append([int(m) for m in best_models])
            for i in range(len(best_models)):
                predictions[x_i] += self.models[best_models[i]].predict(x.unsqueeze(0))

            predictions[x_i] = predictions[x_i] / len(best_models)


        return np.array(predictions)

    def compute_ranking(self, losses):
        assert len(losses) == len(self.models)
        return np.argmin(losses)

    def split_n_omega(self, X):
        if self.roc_mean:
            return windowing(X, self.n_omega+1, z=self.z, use_torch=True)
        else:
            return windowing(X, self.n_omega, z=self.z, use_torch=True)

    def small_split(self, X):
        return windowing(X, self.lag, z=self.small_z, use_torch=True)

    ###
    #       x: Input (shape (batch, channels, features))
    #       y: Label 
    #   model: Single model
    ###
    def compute_explanation(self, x, y, model):
        loss = 0
        cams = []
        for _x, _y in zip(x, y):
            res = model.forward(_x.unsqueeze(0).unsqueeze(0), return_intermediate=True)
            logits = res['logits'].squeeze()
            feats = res['feats']
            l = mse(logits, _y)
            r = np.expand_dims(gradcam(l, feats), 0)
            l = l.detach().item()
            loss += l
            cams.append(r)

        r = np.concatenate(cams, axis=0)
        l = loss

        return r, l

    def evaluate_on_validation(self, x_val, y_val):
        def roc_matrix(rocs, z=1):
            lag = rocs.shape[-1]
            m = np.ones((len(rocs), lag + len(rocs) * z - z)) * np.nan

            offset = 0
            for i, roc in enumerate(rocs):
                m[i, offset:(offset+lag)] = roc
                offset += z

            return m

        def roc_mean(roc_matrix):
            summation_matrix = roc_matrix.copy()
            summation_matrix[np.where(np.isnan(roc_matrix))] = 0
            sums = np.sum(summation_matrix, axis=0)
            nonzeros = np.sum(np.logical_not(np.isnan(roc_matrix)), axis=0)
            return sums / nonzeros

        losses = np.zeros((len(self.models)))

        if self.roc_mean:
            all_cams = np.zeros((len(self.models), self.n_omega))
        else:
            all_cams = []

        X, y = self.small_split(x_val)
        for n_m, m in enumerate(self.models):
            cams, loss = self.compute_explanation(X, y, m)
            losses[n_m] = loss

            if self.roc_mean:
                all_cams[n_m] = roc_mean(roc_matrix(cams, z=1))
            else:
                all_cams.append(cams)

        if not self.roc_mean:
            all_cams = np.array(all_cams)

        return losses, all_cams

    def calculate_rocs(self, x, cams): 
        all_rocs = []
        for i in range(len(cams)):
            rocs = []
            cams_i = cams[i] 

            if len(cams_i.shape) == 1:
                cams_i = np.expand_dims(cams_i, 0)

            for offset, cam in enumerate(cams_i):
                # Normalize CAMs
                max_r = np.max(cam)
                if max_r == 0:
                    continue
                normalized = cam / max_r

                # Extract all subseries divided by zeros
                after_threshold = normalized * (normalized > self.threshold)
                condition = len(np.nonzero(after_threshold)[0]) > 0

                if condition:
                    indidces = split_array_at_zero(after_threshold)
                    for (f, t) in indidces:
                        if t-f >= 2:
                            rocs.append(x[f+offset:(t+offset+1)])

            all_rocs.append(rocs)
        
        return all_rocs

    def find_best_forecaster(self, x, return_closest_roc=False):
        model_distances = np.ones(len(self.models)) * 1e10
        closest_rocs_agg = [None]*len(self.models)

        for i, m in enumerate(self.models):

            x = x.squeeze()
            for r in self.rocs[i]:
                distance = dtw(r, x)
                if distance < model_distances[i]:
                    model_distances[i] = distance
                    closest_rocs_agg[i] = r

        top_models = np.argsort(model_distances)[:self.topm]
        closest_rocs = []
        for i in top_models:
            if closest_rocs_agg[i] is not None:
                closest_rocs.append(closest_rocs_agg[i])

        # There might be more desired models than rocs available, so we need to reduce top models accordingly
        top_models = top_models[:len(closest_rocs)]

        if return_closest_roc:
            return top_models, closest_rocs

        return top_models

class Vanilla_OS_PGSM:

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
                if concept_drift(residuals, len(current_val), len(X_test)):
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

class Vanilla_OEP_ROC:

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

            drift_type_one = concept_drift(mean_residuals, len(current_val), len(X_test), drift_type="type1", R=1.5)
            drift_type_two = concept_drift(ensemble_residuals, len(current_val), len(X_test), drift_type="type2", R=100)

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