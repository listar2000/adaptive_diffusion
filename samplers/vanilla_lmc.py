import numpy as np

from envs.environment import Environment
from envs.multi_modal import GaussianMixtures

OPTIMAL_RATE = 0.574


class VanillaLMCSampler(object):
    def __init__(self, env: Environment,
                 init_eps: float = 0.1,
                 step_size_lr: float = 0.015,
                 _lambda: float = 0.1,
                 initialize_iter: int = 500):

        self.env = env
        self.rho = step_size_lr
        self.epsilon = init_eps

        init_theta = env.init_theta()
        self.current_theta = init_theta
        self.current_score = env.gradient(self.current_theta)
        self.current_density = env.log_density(self.current_theta)

        self.dim = len(init_theta)
        self.initialized = False
        self.initialize_iter = initialize_iter

        self.success, self.total = 0, 0

    def compute_h_diff(self, theta_a, theta_b, score_a, score_b, epsilon):
        h_a_b = 0.5 * (theta_a - theta_b - (epsilon / 4) * score_b).T @ score_b
        h_b_a = 0.5 * (theta_b - theta_a - (epsilon / 4) * score_a).T @ score_a
        return h_a_b - h_b_a

    def initialize(self):
        for _ in range(self.initialize_iter):
            prop_theta = self.current_theta + (self.epsilon / 2) * self.current_score \
                         + np.sqrt(self.epsilon) * np.random.normal(0, 1, self.dim)

            prop_density = self.env.log_density(prop_theta)
            prop_score = self.env.gradient(prop_theta)

            h_diff = self.compute_h_diff(self.current_theta, prop_theta, self.current_score, prop_score, self.epsilon)
            alpha = min(1, np.exp(prop_density + h_diff - self.current_density))

            self.update_epsilon(alpha)
            if alpha > np.random.random():
                # accept
                self.current_theta = prop_theta
                self.current_score = prop_score
                self.current_density = prop_density

        self.initialized = True

    def update_epsilon(self, alpha):
        """
        Adapt the learning rate
        """
        self.epsilon = self.epsilon * (1 + self.rho * (alpha - OPTIMAL_RATE))
        self.rho = max(1e-3, self.rho * 0.999)

    def step(self):
        if not self.initialized:
            self.initialize()

        prop_theta = self.current_theta + (self.epsilon / 2) * self.current_score \
            + np.sqrt(self.epsilon) * np.random.normal(0, 1, self.dim)

        prop_density = self.env.log_density(prop_theta)
        prop_score = self.env.gradient(prop_theta)

        h_diff = self.compute_h_diff(self.current_theta, prop_theta,
                                     self.current_score, prop_score, self.epsilon)
        alpha = min(1, np.exp(prop_density + h_diff - self.current_density))

        self.update_epsilon(alpha)
        if alpha > np.random.random():
            # accept
            self.current_theta = prop_theta
            self.current_score = prop_score
            self.current_density = prop_density
            self.success += 1
        self.total += 1
        return self.current_theta


if __name__ == '__main__':
    from metropolis_hastings import MetropolisHastingsSampler
    from metrics.metrics import univariate_ess
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
        lmc_sampler = VanillaLMCSampler(env=gmm_env, init_eps=1)
        lmc_samps.append(np.array([lmc_sampler.step() for _ in range(5000)]))

        vanilla_sampler = MetropolisHastingsSampler(environment=gmm_env, proposal_std=1)
        vanilla_samps.append(np.array([vanilla_sampler.step() for _ in range(5000)]))

    lmc_samps = np.array(lmc_samps)
    vanilla_samps = np.array(vanilla_samps)
    print(lmc_samps.shape)

    print(univariate_ess(lmc_samps))
    print(univariate_ess(vanilla_samps))
