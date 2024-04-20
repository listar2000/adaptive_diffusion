import numpy as np


class MetropolisHastingsSampler:
    def __init__(self, environment, proposal_std, seed=None):
        self.environment = environment
        self.proposal_std = proposal_std
        if seed:
            np.random.seed(seed)
        self.current_theta = self.environment.init_theta()
        # self.current_theta = np.random.randn(2)  # Initialize at a random point
        # self.current_prior = self.environment.log_prior(self.current_theta)
        # self.current_lik = self.environment.log_likelihood(self.current_theta)
        self.dim = environment.dim
        self.current_density = self.environment.log_density(self.current_theta)
        self.accept, self.total = 0, 0

    def step(self):
        # Propose a new state (theta) using a Gaussian proposal distribution
        proposed_theta = self.current_theta + np.random.normal(0, self.proposal_std, size=self.dim)

        # Calculate the acceptance probability
        # proposed_prior = self.environment.log_prior(proposed_theta)
        # proposed_lik = self.environment.log_likelihood(proposed_theta)

        prop_density = self.environment.log_density(proposed_theta)

        # Compute the acceptance probability
        acceptance_prob = prop_density - self.current_density
        acceptance_prob = min(0, acceptance_prob)  # Ensure the probability is at most 1

        # Accept or reject the new state
        if np.log(np.random.rand()) < acceptance_prob:
            self.current_theta = proposed_theta  # Accept the new state
            # self.current_prior = proposed_prior
            # self.current_lik = proposed_lik
            self.current_density = prop_density
            self.accept += 1

        self.total += 1
        return self.current_theta

