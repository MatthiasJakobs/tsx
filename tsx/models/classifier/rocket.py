import torch
import pickle
import torch.nn as nn
import numpy as np

from os.path import join

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split

from tsx.models.classifier import BasePyTorchClassifier

from eao.individual import Individual
from eao.optimizer import Optimizer
from eao.evaluator import Evaluator
from eao.logger import Logger, LOG_VERBOSE


def get_random_kernel_length():
    return [7, 9, 11][np.random.randint(0, 3)]

def get_random_weights(kernel_length):
    weights = torch.normal(torch.zeros(1,1,kernel_length), 1)
    return weights - torch.mean(weights)

def get_random_bias():
    return (-1 - 1) * torch.rand(1) + 1

def get_random_dilation(input_length, kernel_length):
    A = np.log2((input_length-1) / (float(kernel_length)-1))
    return torch.floor(2**(torch.rand(1)*A)).long().item()

def get_random_padding():
    return 0 if torch.rand(1)>0.5 else 1

def create_random_kernel_params(input_length):
    kernel_length = get_random_kernel_length()
    weights = get_random_weights(kernel_length)
    bias = get_random_bias()
    dilation = get_random_dilation(input_length, kernel_length)
    padding = get_random_padding()
    return [kernel_length, padding, dilation, weights, bias]

def create_kernel_from_params(kernel_length, padding=None, dilation=None, weights=None, bias=None):
    if isinstance(kernel_length, list): # if given tuple, unpack the params
        kernel_length, padding, dilation, weights, bias = kernel_length
    kernel = nn.Conv1d(1, 1, kernel_size=kernel_length, stride=1, padding=padding, dilation=dilation, bias=True)
    kernel.weight = nn.Parameter(weights, requires_grad=False)
    kernel.bias = nn.Parameter(bias, requires_grad=False)
    kernel.require_grad = False
    return kernel


class ROCKET(BasePyTorchClassifier):

    # TODO: Only for equal-length datasets?
    # TODO: Pytorch version appears to be very unstable. Needs more work
    def __init__(self, input_length=10, k=10_000, ridge=True, ppv_only=False, use_sigmoid=False, **kwargs):
        super(ROCKET, self).__init__(**kwargs)
        self.k = k
        self.ridge = ridge
        self.input_length = input_length
        self.ppv_only = ppv_only
        self.use_sigmoid = False

        self.kernels = []
        self.inform("Start building kernels")
        self.build_kernels()
        self.inform("Finished building kernels")

        if self.ridge:
            self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        else:
            if ppv_only:
                self.logits = nn.Linear(self.k, self.n_classes)
            else:
                self.logits = nn.Linear(2*self.k, self.n_classes)

            self.classifier = nn.Sequential(self.logits, nn.Softmax(dim=-1))

    def save(self):
        torch.save(self.kernels, 'rocket.kernels')
        if self.ridge:
            pickle.dump(self.classifier, open('rocket.classifier', 'wb'))
        else:
            torch.save(self.classifier.state_dict(), 'rocket.classifier')

    def load(self, path):
        self.kernels = torch.load(join(path, 'rocket.kernels'))
        if self.ridge:
            with open(join(path, 'rocket.classifier'), 'rb') as fp:
                self.classifier = pickle.load(fp)
        else:
            self.classifier.load_state_dict(torch.load(join(path, 'rocket.classifier')))

    def build_kernels(self):
        self.kernel_params = []
        for i in range(self.k):
            self.kernel_params.append(create_random_kernel_params(self.input_length))
            kernel = create_kernel_from_params(self.kernel_params[-1])
            self.kernels.append(kernel)

    def transform(self, X):
        if isinstance(X, type(np.zeros(1))):
            X = torch.from_numpy(X).float()

        multiplier = 1 if self.ppv_only else 2
        if X.shape[-1] == self.k * multiplier:
            return X

        return self.apply_kernels(X)

    def proba(self, x):
        x = self.transform(x)
        if self.ridge:
            return self.classifier.decision_function(x)
        else:
            return self.logits(x)

    def predict(self, x):
        x = self.transform(x)
        if self.ridge:
            return self.classifier.predict(x)
        else:
            return self.classifier(x)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.inform("Start fitting")
        if self.ridge:
            # Custom `fit` for Ridge regression
            X_train, y_train, X_test, y_test = self.preprocessing(X_train, y_train, X_test=X_test, y_test=y_test)
            self.classifier.fit(X_train, y_train)
            if X_test is not None and y_test is not None:
                print("ROCKET: Test set accuracy", self.classifier.score(X_test, y_test))
            self.fitted = True
        else:
            super().fit(X_train, y_train, X_test=X_test, y_test=y_test)

        self.inform("Finished fitting")

    def apply_kernels(self, X):
        features_ppv = []
        features_max = []
        with torch.no_grad():
            for i in range(self.k):
                if len(X.shape) == 2:
                    X = X.unsqueeze(1) # missing channel information

                transformed_data = self.kernels[i](X)

                features_ppv.append(self._ppv(transformed_data, dim=-1))
                if not self.ppv_only:
                    features_max.append(torch.max(transformed_data, dim=-1)[0])

            features_ppv = torch.cat(features_ppv, -1)
            if self.ppv_only:
                return features_ppv
            else:
                features_max = torch.cat(features_max, -1)
                return torch.cat((features_ppv, features_max), -1)

    def _ppv(self, x, dim=-1):
        # use sigmoid as a soft approximation for ">" activation
        if self.use_sigmoid:
            return torch.mean(torch.sigmoid(x), dim=-1)
        return torch.mean((x > 0).float(), dim=-1)

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        self.inform("Start preprocessing")
        X_train = self.apply_kernels(X_train)

        if X_test is not None:
            X_test = self.apply_kernels(X_test)

        self.inform("Finished preprocessing")
        return X_train, y_train, X_test, y_test

    def forward(self, x):
        if self.ridge:
            raise ValueError("'forward' should not be called if not a neural network model!")
        return self.classifier(x)


class RocketParams(Individual):
    
    def __init__(self, input_length, params, logger=None):
        super().__init__(logger)
        self.input_length = input_length
        self.params = params
        # print(str(self))

    def __repr__(self):
        return str(self.params)# + " (loss={})".format(self.loss_)

    def copy(self):
        params = [self.params[0], self.params[1], self.params[2], self.params[3], self.params[4]]
        ind = self.__class__(self.input_length, params, self.logger_)
        return ind

    @classmethod
    def random(cls, input_length, logger=None):
        return cls(input_length, create_random_kernel_params(input_length), logger=logger)

    def mutate(self, mutation_rate=0.1, mutation_width=1):
        # kernel_length, padding, dilation, weights, bias

        if isinstance(mutation_width, int):
            # interpret as absolute number
            k = mutation_width
        elif isinstance(mutation_width, float):
            # interpret as probability per vector element
            k = max(1, np.random.binomial(len(self.params), mutation_width))

        indices = np.random.choice(len(self.params), size=k, replace=False)

        if 0 in indices: # kernel length
            self.params[0] = get_random_kernel_length()
            changed_kernel_length = True
        else:
            changed_kernel_length = False
        if 1 in indices: # padding
            self.params[1] = get_random_padding()
        if changed_kernel_length or 2 in indices: # dilation
            self.params[2] = get_random_dilation(self.input_length, self.params[0])
        if changed_kernel_length or 3 in indices: # weights
            self.params[3] = get_random_weights(self.params[0])
        if 4 in indices: # bias
            self.params[4] = get_random_bias()
        # self.log('Added ' + ', '.join(['{} to vec[{}]'.format(noise_, ix) for ix, noise_ in zip(indices, noise)]) + 'of #'+str(self.id_), log_level=2, id=12)

    def cross(self, other, crossover_rate=0.5):
        # uniform crossover
        k = max(1, np.random.binomial(len(self.params), crossover_rate))
        indices = np.random.choice(len(self.params), size=k, replace=False)
        if 0 in indices and self.params[0] != other.params[0]:
            # kernel length is changed, so dilation and weights also have to be changed
            changed_kernel_length = True
        else:
            changed_kernel_length = False
        for idx in indices:
            if idx == 3: # make a copy of the weights tensor
                self.params[idx] = other.params[idx].detach().clone()
            else:
                self.params[idx] = other.params[idx]
        if changed_kernel_length:
            self.params[2] = other.params[2]
            self.params[3] = other.params[3].detach().clone()
        # self.log('#{} took indices {} from #{}'.format(self.id_, ', '.join(map(str,indices)), other.id_), log_level=2, id=13)


class EVOROCKET(Evaluator):

    # CLASSIC ROCKET:
    # sample k random kernels

    # EVOROCKET
    # sample k random kernels
    # split data in train and test
    # for gen / until convergence:
    #   mutate & recombine m kernels
    #   evaluate kernel fitness on stochastic amount of train / test data
    #   select k best kernels

    def __init__(self, pop_size=50, nr_children=50, generations=50, stoch_fitness_sample_rate=0.3, eval_iters=3):
        self.pop_size = pop_size
        self.nr_children = nr_children
        self.generations = generations
        self.stoch_fitness_sample_rate = stoch_fitness_sample_rate
        self.eval_iters = eval_iters
        self.model = None

    def eval(self, ind):
        # create and evaluate ROCKET with this one kernel
        accs = []
        for _ in range(self.eval_iters): # re-iterated stochastic kernel fitness evaluation
            fit_failed = True
            train_idcs = np.random.choice(self.X_train.shape[0], size=int(self.stoch_fitness_sample_rate * self.X_train.shape[0]), replace=False)
            eval_idcs = np.random.choice(self.X_train.shape[0], size=int(self.stoch_fitness_sample_rate * self.X_train.shape[0]), replace=False)
            X_train = self.X_train[train_idcs]
            y_train = self.y_train[train_idcs]
            X_eval = self.X_train[eval_idcs]
            y_eval = self.y_train[eval_idcs]
            while fit_failed:
                try: # recombination and crossover can lead to faulty dilation / kernel length values
                    model = ROCKET(input_length=X_train.shape[-1], k=1)
                    model.kernels = [create_kernel_from_params(ind.params)]
                    model.fit(X_train, y_train)
                    fit_failed = False
                except RuntimeError:
                    ind.params[2] -= 1
            eval_pred = model.predict(X_eval)
            accs.append(1 - np.where(eval_pred == y_eval.numpy())[0].size / y_eval.numpy().size)
        return np.mean(np.array(accs))

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train # self.X_eval, self.y_train, self.y_eval = train_test_split(X_train, y_train)

        conf = {
            'parents': self.pop_size,
            'offspring': self.nr_children,

            'do_crossover': True,
            'do_self_adaption': True,

            'mutation_prob': 0.9,
            'crossover_kwargs': {
                'crossover_rate': 0.5},
            'mutation_kwargs': {
                'mutation_rate': 1,
                'mutation_width': 0.1},

            'mutation_kwargs_bounds': {
                'mutation_width': (0.0, 1.0)
            }
        }
        with Logger(log_level=LOG_VERBOSE, file="main_log.txt") as l:
            opt = Optimizer(self, config=conf, logger=l)
            population = opt.run([RocketParams.random(X_train.shape[-1]) for _ in range(self.pop_size)], generations=self.generations)
        self.model = ROCKET(input_length=X_train.shape[-1], k=1)
        self.model.kernels = [create_kernel_from_params(ind.params) for ind in population]
        self.model.fit(X_train, y_train)

    def predict(self, x):
        if self.model is None:
            raise EnvironmentError('You have to FIT before you can PREDICT!')
        return self.model.predict(x)
