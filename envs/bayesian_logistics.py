import numpy as np
import pandas as pd

from envs.environment import Environment


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BayesianLogistics(Environment):
    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
                 intercept: bool = True, normalize: bool = True):
        self.n_obs, self.dim = X.shape
        assert self.n_obs == len(y), "length of y must be equal to the number of data"
        self.X = X
        self.y = y
        self.alpha = alpha

        if normalize:
            X_mean, X_std = self.X.mean(axis=0), self.X.std(axis=0)
            self.X = (self.X - X_mean) / X_std

        if intercept:
            self.dim += 1
            self.X = np.hstack((self.X, np.ones((self.n_obs, 1))))

        self.cache_beta = None
        self.cache_sigmoid = None

    def _update_sigmoid(self, beta):
        if (beta == self.cache_beta).all():
            return self.cache_sigmoid
        self.cache_beta = beta
        self.cache_sigmoid = sigmoid(self.X @ beta)
        return self.cache_sigmoid

    def init_theta(self):
        # simply return from the prior
        return np.random.randn(self.dim) * np.sqrt(self.alpha)

    def log_prior(self, beta):
        return - np.dot(beta, beta) / (2 * self.alpha)

    def log_likelihood(self, beta):
        z = np.dot(self.X, beta)

        log_sigmoids = -np.log1p(np.exp(-z))
        log_one_minus_sigmoids = -np.log1p(np.exp(z))
        log_lik = np.sum(self.y * log_sigmoids + (1 - self.y) * log_one_minus_sigmoids)
        return log_lik

    def log_density(self, beta):
        return self.log_prior(beta) + self.log_likelihood(beta)

    def gradient(self, beta):
        # the score function at `beta`
        grad_prior = -beta / self.alpha

        sigmoids = self._update_sigmoid(beta)
        tmp_matrix = (self.y - sigmoids)[:, np.newaxis] * self.X
        grad_lik = tmp_matrix.sum(axis=0)

        return grad_lik + grad_prior

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        pass

    def construct_G_grad(self, beta):
        sigmoids = self._update_sigmoid(beta)
        _lambda = np.diag(sigmoids * (1 - sigmoids))

        x_lambda = self.X.T @ _lambda  # hold it
        G_mat = x_lambda @ self.X + np.eye(self.dim) / self.alpha
        G_inv = np.linalg.inv(G_mat)

        # construct G_grad
        G_grad = np.zeros((self.dim, self.dim, self.dim))
        G_inv_grad = np.zeros_like(G_grad)
        for i in range(self.dim):
            V_i = np.diag((1 - 2 * sigmoids) * self.X[:, i])
            G_grad_i = x_lambda @ V_i @ self.X
            G_grad[i] = G_grad_i
            G_inv_grad[i] = -G_inv @ G_grad_i @ G_inv

        return G_mat, G_inv, G_inv_grad


def load_diabetes_dataset(file_loc=None):
    if not file_loc:
        file_loc = "../data/diabetes.csv"

    data = pd.read_csv(file_loc)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


if __name__ == '__main__':
    from tqdm import tqdm
    from samplers.riemann_lmc import RiemannLMCSampler

    X, y = load_diabetes_dataset()
    env = BayesianLogistics(X=X, y=y, alpha=100)
    sampler = RiemannLMCSampler(env, step_size_lr=0.015)
    sampler.step()

    states = []
    for i in tqdm(range(10000)):
        states.append(sampler.step())
