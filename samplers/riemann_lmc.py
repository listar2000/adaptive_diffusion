from scipy.linalg import sqrtm
import numpy as np

OPTIMAL_RATE = 0.574


class RiemannLMCSampler(object):
    def __init__(self, env,
                 init_eps: float = 0.1,
                 step_size_lr: float = 0.015,
                 initialize_iter: int = 500):

        self.env = env
        self.rho = step_size_lr
        self.epsilon = init_eps
        self.epsilon_normalized = self.epsilon

        init_theta = env.init_theta()
        self.current_theta = init_theta
        self.current_score = env.gradient(self.current_theta)
        self.current_density = env.log_density(self.current_theta)

        self.dim = len(init_theta)
        self.G = np.identity(self.dim)
        self.G_inv = np.identity(self.dim)
        self._gamma = np.zeros(self.dim)

        self.initialized = False
        self.initialize_iter = initialize_iter

        self.success, self.total = 0, 0

    def compute_h_diff(self, theta_a, theta_b,
                       score_a, score_b, gamma_a, gamma_b,
                       G_a, G_b, G_inv_a, G_inv_b, epsilon):
        inner_a_b = (epsilon / 4) * (G_inv_b @ score_b + gamma_b)
        inner_b_a = (epsilon / 4) * (G_inv_a @ score_a + gamma_a)
        h_a_b = 0.5 * (theta_a - theta_b - inner_a_b).T @ (score_b + G_b @ gamma_b)
        h_b_a = 0.5 * (theta_b - theta_a - inner_b_a).T @ (score_a + G_a @ gamma_a)
        return h_a_b - h_b_a

    # def compute_h_init(self, theta_a, theta_b,
    #                    score_a, score_b, epsilon):
    #     inner_a_b = (epsilon / 4) * score_b
    #     inner_b_a = (epsilon / 4) * score_a
    #     h_a_b = 0.5 * (theta_a - theta_b - inner_a_b).T @ score_b
    #     h_b_a = 0.5 * (theta_b - theta_a - inner_b_a).T @ score_a
    #     return h_a_b - h_b_a
    #
    # def initialize(self):
    #     init_total, init_success = 0, 0
    #     for _ in range(self.initialize_iter):
    #         prop_theta = self.current_theta + (self.epsilon / 2) * self.current_score \
    #                      + np.sqrt(self.epsilon) * np.random.normal(0, 1, self.dim)
    #
    #         prop_density = self.env.log_density(prop_theta)
    #         prop_score = self.env.gradient(prop_theta)
    #
    #         # ignore the G and G_inv here, all identity at initialization
    #         h_diff = self.compute_h_init(self.current_theta, prop_theta,
    #                                      self.current_score, prop_score, self.epsilon)
    #         alpha = min(1, np.exp(prop_density + h_diff - self.current_density))
    #
    #         self.update_epsilon(alpha)
    #         if alpha > np.random.random():
    #             # accept
    #             self.current_theta = prop_theta
    #             self.current_score = prop_score
    #             self.current_density = prop_density
    #             init_success += 1
    #         init_total += 1
    #
    #     self.G, self.G_inv, self._gamma = self.prop_G(self.current_theta)
    #     self.initialized = True

    def prop_G(self, prop_theta):
        """
        Adapt the diffusion square matrix G recursively
        """
        G, G_inv, G_inv_grad = self.env.construct_G_grad(prop_theta)
        prop_gamma = np.zeros(self.dim)

        for i in range(self.dim):
            prop_gamma[i] = np.sum(G_inv_grad[np.arange(self.dim), i, np.arange(self.dim)])

        return G, G_inv, prop_gamma

    def update_epsilon(self, alpha):
        """
        Adapt the learning rate
        """
        # alpha = (self.success + 1) / (self.total + 1)
        self.epsilon = self.epsilon * (1 + self.rho * (alpha - OPTIMAL_RATE))
        self.epsilon_normalized = self.epsilon
        # trace_R = np.trace(self.G_inv) / self.dim
        # self.epsilon_normalized = self.epsilon / trace_R

    def step(self):
        # if not self.initialized:
        #     self.initialize()

        R = sqrtm(self.G_inv)
        prop_theta = self.current_theta + \
         (self.epsilon_normalized / 2) * \
         (self.G_inv @ self.current_score + self._gamma) + \
         np.sqrt(self.epsilon_normalized) * R @ np.random.normal(0, 1, self.dim)

        prop_density = self.env.log_density(prop_theta)
        prop_score = self.env.gradient(prop_theta)

        prop_G, prop_G_inv, prop_gamma = self.prop_G(prop_theta)

        h_diff = self.compute_h_diff(self.current_theta, prop_theta,
                                     self.current_score, prop_score,
                                     self._gamma, prop_gamma,
                                     self.G, prop_G, self.G_inv, prop_G_inv, self.epsilon_normalized)
        alpha = min(1, np.exp(prop_density + h_diff - self.current_density))

        self.update_epsilon(alpha)
        if alpha > np.random.random():
            # accept
            self.current_theta = prop_theta
            self.current_score = prop_score
            self.current_density = prop_density
            # self.thetas.append(self.current_theta)
            self.G = prop_G
            self.G_inv = prop_G_inv
            self._gamma = prop_gamma
            self.success += 1
        self.total += 1
        return self.current_theta
