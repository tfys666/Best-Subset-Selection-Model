import numpy as np
from scipy import linalg
from itertools import combinations

class OptimalSubsetRegression:
    def __init__(self, x, y, index, names, K):
        self.x = x
        self.y = y
        self.index = index
        self.names = names
        self.K = K
        self.xtrain, self.xtest = x[index], x[~index]
        self.ytrain, self.ytest = y[index], y[~index]
        self.n, self.p = self.xtrain.shape
        self.feature_combinations = self._generate_feature_combinations()

    def _generate_feature_combinations(self):
        return [list(comb) for r in range(1, self.p + 1) for comb in combinations(range(self.p), r)]

    def _ols(self, x_centered, y_centered):
        beta_1, _ = linalg.lapack.dpotrs(linalg.cholesky(np.dot(x_centered.T, x_centered)), np.dot(x_centered.T, y_centered))
        beta_0 = y_centered.mean() - np.dot(beta_1, x_centered.mean(axis=0))
        return beta_0, beta_1

    def _train_model(self, x_centered, y_centered, features):
        return self._ols(x_centered[:, features], y_centered)

    def train(self):
        self.xtrain_mean = self.xtrain.mean(axis=0)
        self.xtrain_centered = self.xtrain - self.xtrain_mean
        self.ytrain_mean = self.ytrain.mean()
        self.ytrain_centered = self.ytrain - self.ytrain_mean
        self.models = [self._train_model(self.xtrain_centered, self.ytrain_centered, features)
                       for features in self.feature_combinations]

    def predict_err(self, x, y, model):
        b0, b1 = model
        err = y - b0 - np.dot(x, b1)
        return np.inner(err, err)

    def evaluate(self):
        self.err_test = [self.predict_err(self.xtest[:, features], self.ytest, model)
                         for features, model in zip(self.feature_combinations, self.models)]
        self.rss = [self.predict_err(self.xtrain[:, features], self.ytrain, model)
                    for features, model in zip(self.feature_combinations, self.models)]

    def cross_validate(self):
        indexs = np.array_split(np.random.permutation(np.arange(self.n)), self.K)
        self.cv_err = []

        for index in indexs:
            x_i = self.xtrain[index, :]
            y_i = self.ytrain[index]
            cv_err_i = [self.predict_err(x_i[:, features], y_i, model)
                        for features, model in zip(self.feature_combinations, self.models)]
            self.cv_err.append(np.mean(cv_err_i))

        self.avg_cv_err = np.mean(self.cv_err)

    def best_model(self):
        best_ind = np.argmin(self.cv_err)
        best_b_0, best_b_1 = self.models[best_ind]
        best_features = self.feature_combinations[best_ind]
        selected_names = [self.names[i] for i in best_features]
        return best_b_0, best_b_1, selected_names
