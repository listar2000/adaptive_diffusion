import numpy as np
from envs.environment import Environment
from scipy.stats import multivariate_normal, norm

"""
theta_1 ~ N(0, sigma_1^2), theta_2 ~ N(0, sigma_2^2)
x_i ~ 0.5 * N(theta_1, sigma_x^2) + 0.5 * N(theta_1 + theta_2, sigma_x^2)
"""


def generate_bayesian_learning_simple(num, seed=2024, save_file="../data/bayesian_learning_easy.npz"):
    true_theta_0, true_theta_1 = 0, 1
    sigma_x = 2
    np.random.seed(seed)  # Set the random seed for reproducibility

    # np uses scale rather than variance
    # Generate a random array of 0s and 1s to decide from which distribution to sample
    choices = np.random.choice([0, 1], size=num, p=[0.5, 0.5])

    # Sample from the first Gaussian if choice is 0, and from the second if 1
    samples = np.where(choices == 0,
                       np.random.normal(true_theta_0, np.sqrt(sigma_x), num),
                       np.random.normal(true_theta_0 + true_theta_1, np.sqrt(sigma_x), num))

    data_dict = {
        "samples": samples.flatten(),
        "true_theta": np.array([true_theta_0, true_theta_1]),
        "mu_theta": np.zeros(2),
        "sigma_theta": np.array([10, 1]),
        "sigma_x": sigma_x
    }

    if save_file:
        np.savez(save_file, **data_dict)
        print(f"successfully saved {num} data points")

    return data_dict


def generate_bayesian_learning_hard(num, seed=2024, save_file="../data/bayesian_learning_hard.npz"):
    np.random.seed(seed)
    true_theta = np.zeros(10)
    sigma_theta, mu_theta = np.zeros(10), np.zeros(10)
    sigma_x = 2
    for i in range(10):
        mu_i, sigma_i = np.random.uniform(-2, 2), np.random.uniform(1, 10)
        sigma_theta[i] = sigma_i
        mu_theta[i] = mu_i
        true_theta[i] = np.random.normal(mu_i, np.sqrt(sigma_i))

    choices = np.random.choice([0, 1], size=num, p=[0.5, 0.5])
    samples = np.where(choices == 0,
                       np.random.normal(np.sum(true_theta[:5]), np.sqrt(sigma_x), num),
                       np.random.normal(np.sum(true_theta[5:]), np.sqrt(sigma_x), num))

    data_dict = {
        "samples": samples.flatten(),
        "true_theta": true_theta,
        "sigma_theta": sigma_theta,
        "mu_theta": mu_theta,
        "sigma_x": sigma_x
    }

    if save_file:
        np.savez(save_file, **data_dict)
        print(f"successfully saved {num} data points")

    return data_dict


class BayesianLearningSimple(Environment):
    def __init__(self, data):
        self.obs = data["samples"]
        sigma_theta = data["sigma_theta"]
        assert len(sigma_theta) == 2
        self.sigma_1, self.sigma_2 = sigma_theta[0], sigma_theta[1]
        self.dim = 2
        self.mu_theta = data["mu_theta"]
        self.sigma_x = data["sigma_x"]
        self.prior_mvn = multivariate_normal(mean=self.mu_theta, cov=[[self.sigma_1, 0], [0, self.sigma_2]])

    def init_theta(self) -> np.ndarray:
        return self.prior_mvn.rvs()

    def gradient(self, theta):
        theta_1, theta_2 = theta[0], theta[1]
        lik_1 = multivariate_normal(mean=theta_1, cov=self.sigma_x)
        lik_2 = multivariate_normal(mean=theta_1 + theta_2, cov=self.sigma_x)
        pdf_1, pdf_2 = lik_1.pdf(self.obs), lik_2.pdf(self.obs)
        grad_lik_denum = 0.5 * pdf_1 + 0.5 * pdf_2
        grad_lik_num = 0.5 * pdf_1 * (self.obs - theta_1) / self.sigma_x + \
                       0.5 * pdf_2 * (self.obs - theta_1 - theta_2) / self.sigma_x
        grad_prior = -theta / np.array([self.sigma_1, self.sigma_2])
        assert len(grad_lik_num) == len(grad_lik_denum)
        grad_lik = np.sum(grad_lik_num / grad_lik_denum)
        return grad_prior + grad_lik

    def hessian(self, theta):
        pass

    def log_density(self, theta):
        return self.log_likelihood(theta) + self.log_prior(theta)

    def log_likelihood(self, theta):
        theta_1, theta_2 = theta[0], theta[1]
        lik_1 = multivariate_normal(mean=theta_1, cov=self.sigma_x)
        lik_2 = multivariate_normal(mean=theta_1 + theta_2, cov=self.sigma_x)
        ll = np.sum(np.log(0.5 * lik_1.pdf(self.obs) + 0.5 * lik_2.pdf(self.obs)))
        return ll

    def log_prior(self, theta):
        return self.prior_mvn.logpdf(theta)


class BayesianLearningHard(Environment):
    def __init__(self, data):
        self.obs = data["samples"]
        self.sigma_theta = data["sigma_theta"]
        self.mu_theta = data["mu_theta"]
        self.sigma_x = data["sigma_x"]
        self.prior_mvn = multivariate_normal(mean=self.mu_theta, cov=np.diag(self.sigma_theta))
        self.dim = len(self.mu_theta)
        assert self.dim == 10

    def init_theta(self) -> np.ndarray:
        return self.prior_mvn.rvs()

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        pass

    def log_density(self, theta):
        return self.log_likelihood(theta) + self.log_prior(theta)

    def log_likelihood(self, theta):
        theta_sum_1, theta_sum_2 = np.sum(theta[:5]), np.sum(theta[5:])
        lik_1 = norm.pdf(self.obs, loc=theta_sum_1, scale=np.sqrt(self.sigma_x))
        lik_2 = norm.pdf(self.obs, loc=theta_sum_2, scale=np.sqrt(self.sigma_x))
        # lik_1 = multivariate_normal(mean=theta_sum_1, cov=self.sigma_x)
        # lik_2 = multivariate_normal(mean=theta_sum_2, cov=self.sigma_x)
        ll = np.sum(np.log(0.5 * lik_1 + 0.5 * lik_2))
        return ll

    def log_prior(self, theta):
        return self.prior_mvn.logpdf(theta)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        theta_sum_1, theta_sum_2 = np.sum(theta[:5]), np.sum(theta[5:])
        sigma_sd = np.sqrt(self.sigma_x)
        pdf_1 = norm.pdf(self.obs, loc=theta_sum_1, scale=sigma_sd)
        pdf_2 = norm.pdf(self.obs, loc=theta_sum_2, scale=sigma_sd)
        grad_lik_denum = 0.5 * pdf_1 + 0.5 * pdf_2
        grad_lik_num_1 = 0.5 * pdf_1 * (self.obs - theta_sum_1) / self.sigma_x
        grad_lik_num_2 = 0.5 * pdf_2 * (self.obs - theta_sum_2) / self.sigma_x

        grad_lik = np.zeros(self.dim)
        grad_lik[:5] = np.sum(grad_lik_num_1 / grad_lik_denum)
        grad_lik[5:] = np.sum(grad_lik_num_2 / grad_lik_denum)

        grad_prior = -(theta - self.mu_theta) / self.sigma_theta
        assert len(grad_lik) == len(grad_prior)
        return grad_prior + grad_lik


# Example initialization
# true_theta = np.array([0, 1])  # True parameter value
# prior_covariance = np.diag([10, 1])  # Prior covariance matrix
# noise_covariance = np.eye(2)  # Noise covariance matrix
# num_observations = 100  # Number of observations to simulate
#
# environment = BayesianLearningEnvironment(true_theta, prior_covariance, noise_covariance, num_observations)
if __name__ == "__main__":
    from tqdm import tqdm
    data = np.load("../data/bayesian_learning_hard.npz")
    env = BayesianLearningHard(data=data)

    from samplers.fisher_lmc import FisherLMCSampler
    from samplers.adaptive_lmc import AdaptiveLMCSampler

    samp = AdaptiveLMCSampler(env=env, _lambda=0.1)
    states = []
    for _ in tqdm(range(10000)):
        states.append(samp.step())

    states = np.array(states)
