import numpy as np
from envs.environment import Environment
from scipy.stats import multivariate_normal

"""
TODO: the current environment only supports uncorrelated (i.e. diagonal) prior covariance matrix
"""


class BayesianLearningEnvironment(object):
    def __init__(self, dim, observations, prior_theta=np.zeros(2), prior_covariance=np.diag([10, 1])):
        self.prior_theta = prior_theta
        self.prior_covariance = prior_covariance
        self.observations = observations
        self.dim = dim
        # storing hessian matrices to avoid re-computations
        self.hessian_mat = None
        self.prior_mvn = multivariate_normal(mean=prior_theta, cov=prior_covariance)
        self.init_theta = np.zeros(self.dim)

    def gradient(self, theta):
        # Calculating the gradient of the posterior distribution
        assert len(theta) == 2
        theta_a, theta_b = theta[0], theta[1]
        grad_base = np.sum(self.observations - theta_a - 0.5 * theta_b)
        return np.array([grad_base - 0.1 * (theta_a - self.prior_theta[0]),
                         grad_base / 2 - (theta_b - self.prior_theta[1])])

    def hessian(self, theta):
        # For simplicity, we assume the Hessian of the likelihood is constant
        # and can be represented by the inverse of the noise covariance matrix.
        # This is true for a Gaussian likelihood function.
        if self.hessian_mat is None:
            hessian_lik = np.array([[-1, -0.5], [-0.5, -0.25]]) * len(self.observations)
            hessian_prior = -np.linalg.inv(self.prior_covariance)
            self.hessian_mat = hessian_lik + hessian_prior
        return self.hessian_mat

    # Below are useful for MCMC baseline

    def log_likelihood(self, theta):
        exponent = -0.5 * np.sum(np.square(self.observations - theta[0] - 0.5 * theta[1]))
        return exponent + len(self.observations) * np.log(1 / np.sqrt(2 * np.pi))

    def log_prior(self, theta):
        return self.prior_mvn.logpdf(theta)

# Example initialization
# true_theta = np.array([0, 1])  # True parameter value
# prior_covariance = np.diag([10, 1])  # Prior covariance matrix
# noise_covariance = np.eye(2)  # Noise covariance matrix
# num_observations = 100  # Number of observations to simulate
#
# environment = BayesianLearningEnvironment(true_theta, prior_covariance, noise_covariance, num_observations)
