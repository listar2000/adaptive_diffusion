import numpy as np


class HessianBasedAdaptiveDiffusion:
    """
    The Hessian-based Adaptive Diffusion algorithm proposed by paper
    https://arxiv.org/abs/2009.12690
    """
    def __init__(self, env, epsilon_scheduler, alpha_scheduler, beta_scheduler,
                 initial_S=None, initial_D=None, classic=False):
        self.env = env
        self.dim = env.dim
        # Initial skew-symmetric matrix S
        if initial_S is None:
            initial_S = np.array([[0, -1.5], [1.5, 0]])
        self.S = initial_S
        # Initial matrix D for gradient approximation
        if initial_D is None:
            initial_D = np.array([[np.zeros(self.dim) for _ in range(self.dim)] for _ in range(self.dim)])
        self.D = initial_D
        self.theta = env.init_theta
        self.pi_current = self.env.log_prior(self.theta) + self.env.log_likelihood(self.theta)

        self.epsilon_scheduler = epsilon_scheduler  # Step size for theta update
        self.alpha_scheduler = alpha_scheduler  # Step size for S update
        self.beta_scheduler = beta_scheduler  # Inverse temperature

        self.epsilon, self.alpha, self.beta = None, None, None
        self.classic = classic
        self.accepted = 0

    def update_theta(self, grad_theta):
        noise = np.sqrt(2 * self.epsilon / self.beta) * np.random.randn(self.dim)
        skew_base = self.epsilon * (np.eye(self.dim) + self.S)
        prop_theta = self.theta + skew_base @ grad_theta + noise

        # MCMC steps
        q_base = -1/(4 * self.epsilon)
        q_prop_current = q_base * np.linalg.norm(noise)
        grad_prop = self.env.gradient(prop_theta)
        q_current_prop = q_base * np.linalg.norm(self.theta - prop_theta - skew_base @ grad_prop)

        pi_prop = self.env.log_prior(prop_theta) + self.env.log_likelihood(prop_theta)
        log_accept_prob = min(pi_prop + q_current_prop - self.pi_current - q_prop_current, 0)

        if np.log(np.random.rand()) < log_accept_prob:
            self.theta = prop_theta
            self.accepted += 1
            self.pi_current = pi_prop

        return self.theta

    def update_S(self, grad_theta):
        for i in range(self.dim):
            for j in range(i):  # Ensure i > j for upper triangular update
                D_ij = self.D[i, j]
                self.S[i, j] += self.alpha * grad_theta.T @ D_ij
                self.S[j, i] = -self.S[i, j]  # Enforce skew-symmetry

    def update_D(self, grad_theta, hessian_theta):
        term1 = (np.eye(self.dim) + self.S) @ hessian_theta
        # self.D = self.D - self.epsilon * term1 @ self.D
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    d_skew_sym = self.derivative_skew_symmetric(self.dim, i, j)
                    self.D[i, j] += self.epsilon * term1 @ self.D[i, j]
                    self.D[i, j] += self.epsilon * d_skew_sym @ grad_theta

    @staticmethod
    def derivative_skew_symmetric(n, i, j):
        """
        Generates the derivative of a n x n skew-symmetric matrix A with respect to its entry A_ij.
        The derivative is a matrix with 1 in position (i, j), -1 in position (j, i), and zero elsewhere.

        Parameters:
        n (int): Size of the skew-symmetric matrix (n x n).
        i (int): Row index for the derivative (0-indexed).
        j (int): Column index for the derivative (0-indexed).

        Returns:
        np.array: The derivative matrix.
        """
        derivative = np.zeros((n, n))
        derivative[i, j] = 1
        derivative[j, i] = -1
        return derivative

    def step(self):
        grad_theta = self.env.gradient(self.theta)  # Environment provides gradient
        hessian_theta = self.env.hessian(self.theta)  # Environment provides Hessian

        # update learning rates & temperatures based on schedule
        self.epsilon = self.epsilon_scheduler()
        self.alpha = self.alpha_scheduler()
        self.beta = self.beta_scheduler()

        if self.classic:
            self.update_theta(grad_theta)
            return self.theta

        # update core matrices according to the algorithm
        self.update_theta(grad_theta)
        self.update_S(grad_theta)
        self.update_D(grad_theta, hessian_theta)

        # Return updated theta
        return self.theta
