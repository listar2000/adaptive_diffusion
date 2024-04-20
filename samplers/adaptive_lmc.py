import numpy as np
from scipy.linalg import sqrtm

from envs.environment import Environment
from envs.multi_modal import GaussianMixtures
from metrics.metrics import deviation_from_param, univariate_ess

OPTIMAL_RATE = 0.574


class AdaptiveLMCSampler(object):
    def __init__(self, env: Environment,
                 init_eps: float = 0.1,
                 step_size_lr: float = 0.015,
                 _lambda: float = 0.1,
                 initialize_iter: int = 500):

        self.env = env
        self.rho = step_size_lr
        self.epsilon = init_eps
        self.epsilon_normalized = self.epsilon
        self._lambda = _lambda

        init_theta = env.init_theta()
        self.current_theta = init_theta
        self.current_score = env.gradient(self.current_theta)
        self.current_density = env.log_density(self.current_theta)

        # self.thetas = []
        self.A = np.identity(len(init_theta))
        self.mu = 0
        self.dim = len(self.A)
        self.initialized = False
        self.initialize_iter = initialize_iter

        self.success, self.total = 0, 0

    def compute_h_diff(self, theta_a, theta_b, score_a, score_b, epsilon):
        h_a_b = 0.5 * (theta_a - theta_b - (epsilon / 4) * self.A @ score_b).T @ score_b
        h_b_a = 0.5 * (theta_b - theta_a - (epsilon / 4) * self.A @ score_a).T @ score_a
        return h_a_b - h_b_a

    def initialize(self):
        for _ in range(self.initialize_iter):
            prop_theta = self.current_theta + (self.epsilon / 2) * self.current_score \
                         + np.sqrt(self.epsilon) * np.random.normal(0, 1, self.dim)

            prop_density = self.env.log_density(prop_theta)
            prop_score = self.env.gradient(prop_theta)

            h_diff = self.compute_h_diff(self.current_theta, prop_theta, self.current_score, prop_score, self.epsilon)
            alpha = min(1, np.exp(prop_density + h_diff - self.current_density))
            # print(prop_theta, self.current_theta, prop_density, self.current_density, alpha)
            self.update_epsilon(alpha)
            if alpha > np.random.random():
                # accept
                self.current_theta = prop_theta
                self.current_score = prop_score
                self.current_density = prop_density

        self.initialized = True

    def update_A(self, current_theta):
        """
        Adapt the diffusion square matrix R recursively
        """
        if self.total == 1:  # init
            self.mu = current_theta
        elif self.total == 2:
            diff = (current_theta - self.mu).reshape(-1, 1)
            self.A = 0.5 * diff @ diff.T + self._lambda * np.eye(self.dim)
            self.mu = 0.5 * self.mu + 0.5 * current_theta
        else:
            n = self.total
            diff = (current_theta - self.mu).reshape(-1, 1)
            self.A = ((n - 2) / (n - 1)) * self.A + (1 / n) * diff @ diff.T
            self.mu = ((n - 1) / n) * self.mu + (1 / n) * current_theta

    def update_epsilon(self):
        """
        Adapt the learning rate
        """
        alpha = self.success / self.total
        self.epsilon = self.epsilon * (1 + self.rho * (alpha - OPTIMAL_RATE))
        trace_R = np.trace(self.A) / self.dim
        self.epsilon_normalized = self.epsilon / trace_R

    def step(self):
        # if not self.initialized:
        #     self.initialize()

        R = sqrtm(self.A)
        assert np.isreal(R).all(), self.A
        prop_theta = self.current_theta + (self.epsilon_normalized / 2) * self.A @ self.current_score \
            + np.sqrt(self.epsilon_normalized) * R @ np.random.normal(0, 1, self.dim)

        prop_density = self.env.log_density(prop_theta)
        prop_score = self.env.gradient(prop_theta)

        h_diff = self.compute_h_diff(self.current_theta, prop_theta,
                                     self.current_score, prop_score, self.epsilon_normalized)
        alpha = min(1, np.exp(prop_density + h_diff - self.current_density))

        if alpha > np.random.random():
            # accept
            self.current_theta = prop_theta
            self.current_score = prop_score
            self.current_density = prop_density
            # self.thetas.append(self.current_theta)
            self.success += 1
        self.total += 1
        self.update_A(self.current_score)
        self.update_epsilon()
        return self.current_theta


if __name__ == '__main__':
    from metropolis_hastings import MetropolisHastingsSampler
    from tqdm import tqdm
    # Initialize the GMM environment
    means = [np.array([1, 1]), np.array([-1, -1])]  # Example means
    covariances = [np.eye(2), np.eye(2)]  # Equal uncorrelated covariances
    weights = [0.5, 0.5]  # Equal weights

    gmm_env = GaussianMixtures(means, covariances, weights)
    # lmc_sampler = FisherLMCSampler(env=gmm_env, init_eps=1)
    # lmc_sampler.initialize()
    #
    # vanilla_sampler = MetropolisHastingsSampler(environment=gmm_env, proposal_std=1)

    lmc_samps, vanilla_samps = [], []
    for i in tqdm(range(10)):
        lmc_sampler = AdaptiveLMCSampler(env=gmm_env, init_eps=1)
        lmc_sampler.initialize()
        lmc_samps.append(np.array([lmc_sampler.step() for _ in range(5000)]))

        vanilla_sampler = MetropolisHastingsSampler(environment=gmm_env, proposal_std=1)
        vanilla_samps.append(np.array([vanilla_sampler.step() for _ in range(5000)]))

    lmc_samps = np.array(lmc_samps)
    vanilla_samps = np.array(vanilla_samps)
    print(lmc_samps.shape)

    print(univariate_ess(lmc_samps))
    print(univariate_ess(vanilla_samps))