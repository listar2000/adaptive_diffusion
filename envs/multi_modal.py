from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.environment import Environment

import numpy as np
from typing import List

from scipy.stats import multivariate_normal
from scipy.linalg import cholesky


class GaussianMixtures(Environment):
    def __init__(self, means: List[np.array], variances: List[np.ndarray], weights: List):
        assert len(means) == len(variances) and len(means) == len(weights)
        self.n = len(means[0])
        self.means = means
        self.variances = variances
        self.weights = np.array(weights)
        self.weights /= np.sum(self.weights)  # Normalize weights
        self.cached_contour_plot = None
        self.use_batch = True

    def init_theta(self) -> np.ndarray:
        # Randomly select a component based on weights
        chosen_component = np.random.choice(len(self.weights), p=self.weights)
        # Return the mean of the chosen component
        return self.means[chosen_component]

    def log_density(self, xs: np.array):
        total_density = np.sum([w * multivariate_normal.pdf(xs, mean=mu, cov=var)
                                for w, mu, var in zip(self.weights, self.means, self.variances)], axis=0)
        return np.log(total_density)

    def gradient(self, xs: np.array):
        gradients = np.zeros(xs.shape[0])
        total_density = np.zeros(xs.shape[0])

        for w, mu, var in zip(self.weights, self.means, self.variances):
            density = multivariate_normal.pdf(xs, mean=mu, cov=var)
            inv_cov = np.linalg.inv(var)
            grad_component = -inv_cov @ (xs - mu)
            gradients += (w * density) * grad_component
            total_density += w * density

        normalized_gradients = gradients / total_density
        return normalized_gradients

    def gradient_batch(self, xs):
        # Initialize gradients
        gradients = np.zeros_like(xs)

        # For broadcasting, reshape weights and total_density
        total_density = np.zeros(xs.shape[0])

        for w, mu, var in zip(self.weights, self.means, self.variances):
            # Density for each sample
            density = multivariate_normal.pdf(xs, mean=mu, cov=var).reshape(-1, 1)

            inv_cov = np.linalg.inv(var)
            grad_component = (xs - mu) @ inv_cov
            gradients += w * density * grad_component
            total_density += w * density.squeeze()

        return gradients / total_density.reshape(-1, 1)

    def contour_plot(self, title=None):
        # if self.cached_contour_plot is not None:
        #     return self.cached_contour_plot

        fig, ax = plt.subplots()
        # Create a grid
        x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        XY = np.dstack([X, Y])

        # Compute density
        Z = np.apply_along_axis(self.log_density, 2, XY)

        # Plot
        ax.contour(X, Y, np.exp(Z))
        if not title:
            ax.set_title('Contour Plot of the Gaussian Mixture Model')
        else:
            ax.set_title(title)
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')

        # Cache the plot
        self.cached_contour_plot = ax
        return fig, ax

    def hessian(self):
        pass


class MetropolisHastingsSampler:
    def __init__(self, environment, proposal_std=1.0):
        self.environment = environment
        self.proposal_std = proposal_std

    def _reset(self):
        self.accepted_moves = []
        self.moved_distances = []
        self.total_iterations = 0

    def sample(self, num_samples):
        self._reset()

        current_sample = np.zeros(self.environment.dim)  # Initial sample
        self.accepted_moves.append(current_sample)
        current_density = self.environment.log_density(current_sample)
        samples = []

        for _ in range(num_samples):
            self.total_iterations += 1
            new_sample = current_sample + np.random.normal(scale=self.proposal_std, size=2)
            new_density = self.environment.log_density(new_sample)

            # Calculate acceptance probability
            acceptance_prob = min(1, np.exp(new_density - current_density))

            # Accept or reject the new sample
            if np.random.rand() < acceptance_prob:
                move_distance = np.linalg.norm(new_sample - current_sample)
                self.moved_distances.append(move_distance)
                if len(self.accepted_moves) < 20:
                    self.accepted_moves.append(new_sample)

                current_sample = new_sample
                current_density = new_density
            samples.append(current_sample)

        return np.array(samples)

    def acceptance_rate(self):
        return len(self.moved_distances) / self.total_iterations

    def average_moved_distance(self, total=True):
        if self.total_iterations == 0:
            return 0
        if total:
            return sum(self.moved_distances) / self.total_iterations
        else:
            return np.mean(self.moved_distances)

    def plot_accepted_moves_on_contour(self):
        # Generate the contour plot from the environment
        ax = self.environment.contour_plot()

        # Extract the first 20 accepted moves
        accepted_moves = np.array(self.accepted_moves)

        # Calculate transition lengths and normalize them
        if len(accepted_moves) > 1:
            # transitions = np.linalg.norm(np.diff(accepted_moves, axis=0), axis=1)
            # transitions_normalized = (transitions - transitions.min()) / (transitions.max() - transitions.min())
            #
            # # Create a colormap
            # cmap = plt.cm.viridis

            # Plot each transition with an arrow
            for i in range(len(accepted_moves) - 1):
                start, end = accepted_moves[i], accepted_moves[i + 1]
                ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                         color="black",
                         length_includes_head=True,
                         head_width=0.1, head_length=0.1)

        plt.show()


class LangevinSampler:
    def __init__(self, environment, epsilon=0.1, alpha=0.):
        self.environment = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.skew_matrix = np.array([[0, -self.alpha], [self.alpha, 0]])

        # mcmc steps
        self.accepted_moves = []
        self.moved_distances = []
        self.total_iterations = 0

        # # MH-adjustment
        # self.current_sample_gradient = 0
        # self.current_log_density = 0

    def _reset(self):
        self.accepted_moves = []
        self.moved_distances = []
        self.total_iterations = 0

    def sample(self, num_samples):
        self._reset()

        current_sample = np.zeros(self.environment.dim)  # Initial sample
        current_sample_gradient = self.environment.gradient(current_sample)
        current_log_density = self.environment.log_density(current_sample)
        self.accepted_moves.append(current_sample)
        samples = [current_sample]

        for _ in range(num_samples - 1):
            self.total_iterations += 1
            noise = np.random.randn(self.environment.dim) * np.sqrt(2 * self.epsilon)

            skew_base = self.epsilon * (np.eye(self.environment.dim) + self.skew_matrix)
            # Update rule
            new_sample = current_sample + skew_base @ current_sample_gradient + noise

            # MH-adjustment step
            q_base = -1 / (4 * self.epsilon)
            new_sample_grad = self.environment.gradient(new_sample)
            new_log_density = self.environment.log_density(new_sample)
            current_to_new = q_base * np.linalg.norm(noise) ** 2
            new_to_current = q_base * np.linalg.norm(current_sample - new_sample - skew_base @ new_sample_grad) ** 2

            log_accept_prob = min(new_log_density + new_to_current - current_log_density - current_to_new, 0)

            if np.log(np.random.rand()) < log_accept_prob:
                move_distance = np.linalg.norm(new_sample - current_sample)
                self.moved_distances.append(move_distance)
                if len(self.accepted_moves) < 20:
                    self.accepted_moves.append(new_sample)

                current_sample = new_sample
                current_sample_gradient = new_sample_grad
                current_log_density = new_log_density

            samples.append(current_sample)

        return np.array(samples)

    def acceptance_rate(self):
        return len(self.moved_distances) / self.total_iterations

    def average_moved_distance(self, total=True):
        if self.total_iterations == 0:
            return 0
        if total:
            return sum(self.moved_distances) / self.total_iterations
        else:
            return np.mean(self.moved_distances)

    def plot_accepted_moves_on_contour(self):
        ax = self.environment.contour_plot()

        # Extract the first 20 accepted moves
        accepted_moves = np.array(self.accepted_moves)

        # Calculate transition lengths and normalize them
        if len(accepted_moves) > 1:
            # transitions = np.linalg.norm(np.diff(accepted_moves, axis=0), axis=1)
            # transitions_normalized = (transitions - transitions.min()) / (transitions.max() - transitions.min())
            #
            # # Create a colormap
            # cmap = plt.cm.viridis

            # Plot each transition with an arrow
            for i in range(len(accepted_moves) - 1):
                start, end = accepted_moves[i], accepted_moves[i + 1]
                ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                         color="black",
                         length_includes_head=True,
                         head_width=0.1, head_length=0.1)

        plt.show()


def compute_sample_covariance(samples):
    # Ensure samples is a 2D array
    if samples.ndim != 2:
        raise ValueError("Samples should be a 2D array.")

    # Compute the covariance matrix
    covariance_matrix = np.cov(samples, rowvar=False)
    return covariance_matrix


class EnsembleLangevinSampler:
    def __init__(self, environment, ensemble=10, epsilon=0.1, alpha=0.1):
        self.environment = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.ensemble = ensemble
        self.skew_matrix = np.array([[0, -self.alpha], [self.alpha, 0]])
        self.use_batch = environment.use_batch

        # mcmc steps
        self.accepted_moves = []
        self.moved_distances = []
        self.total_iterations = 0

    def _reset(self):
        self.accepted_moves = []
        self.moved_distances = []
        self.total_iterations = 0

    def sample(self, iterations, use_cov=True, init_sigma=1):
        self._reset()

        # Initialize the ensemble of samples
        current_samples = np.random.randn(self.ensemble, self.environment.dim) * init_sigma  # Shape: (N, n)

        if self.use_batch:
            current_samples_gradients = self.environment.gradient_batch(current_samples)
            current_log_densities = self.environment.log_density(current_samples)
        else:
            current_samples_gradients, current_log_densities = [], []
            for i in range(self.ensemble):
                current_samples_gradients.append(self.environment.gradient(current_samples[i]))
                current_log_densities.append(self.environment.log_density(current_samples[i]))
            current_samples_gradients = np.array(current_samples_gradients)
            current_log_densities = np.array(current_log_densities)

        all_samples = []

        for _ in tqdm(range(iterations)):
            self.total_iterations += 1

            if use_cov:
                e_cov = compute_sample_covariance(current_samples)
                noise_covariance = 2 * self.epsilon * e_cov
                noise = np.random.multivariate_normal(np.zeros(self.environment.dim), noise_covariance, self.ensemble)
            else:
                noise = np.random.randn(self.ensemble, self.environment.dim) * np.sqrt(2 * self.epsilon)

            # Update rule for the whole ensemble
            if use_cov:
                skew_base = self.epsilon * (e_cov + self.skew_matrix)
            else:
                skew_base = self.epsilon * (np.eye(self.environment.dim) + self.skew_matrix)

            new_samples = current_samples + np.dot(skew_base, current_samples_gradients.T).T + noise

            # MH-adjustment step for the whole ensemble
            q_base = -1 / (4 * self.epsilon)

            if self.use_batch:
                new_samples_gradients = self.environment.gradient_batch(new_samples)
                new_log_densities = self.environment.log_density(new_samples)
            else:
                new_samples_gradients, new_log_densities = [], []
                for i in range(self.ensemble):
                    new_samples_gradients.append(self.environment.gradient(new_samples[i]))
                    new_log_densities.append(self.environment.log_density(new_samples[i]))
                new_samples_gradients = np.array(new_samples_gradients)
                new_log_densities = np.array(new_log_densities)

            current_to_new = q_base * np.linalg.norm(noise, axis=1) ** 2
            new_to_current = q_base * np.linalg.norm(current_samples - new_samples -
                                                     np.dot(skew_base, new_samples_gradients.T).T, axis=1) ** 2

            log_accept_probs = np.minimum(new_log_densities + new_to_current -
                                          current_log_densities - current_to_new, 0)

            accept_mask = np.log(np.random.rand(self.ensemble)) < log_accept_probs
            move_distances = np.linalg.norm(new_samples - current_samples, axis=1)
            self.moved_distances.extend(move_distances[accept_mask])

            # Update accepted moves and samples for the next iteration
            current_samples[accept_mask] = new_samples[accept_mask]
            current_samples_gradients[accept_mask] = new_samples_gradients[accept_mask]
            current_log_densities[accept_mask] = new_log_densities[accept_mask]
            all_samples.append(current_samples.copy())

        return np.vstack(all_samples)

    def acceptance_rate(self):
        return len(self.moved_distances) / (self.total_iterations * self.ensemble)

    def average_moved_distance(self, total=True):
        if self.total_iterations == 0:
            return 0
        if total:
            return sum(self.moved_distances) / (self.total_iterations * self.ensemble)
        else:
            return np.mean(self.moved_distances)

    def plot_accepted_moves_on_contour(self):
        ax = self.environment.contour_plot()

        # Extract the first 20 accepted moves
        accepted_moves = np.array(self.accepted_moves)

        # Calculate transition lengths and normalize them
        if len(accepted_moves) > 1:
            # transitions = np.linalg.norm(np.diff(accepted_moves, axis=0), axis=1)
            # transitions_normalized = (transitions - transitions.min()) / (transitions.max() - transitions.min())
            #
            # # Create a colormap
            # cmap = plt.cm.viridis

            # Plot each transition with an arrow
            for i in range(len(accepted_moves) - 1):
                start, end = accepted_moves[i], accepted_moves[i + 1]
                ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                         color="black",
                         length_includes_head=True,
                         head_width=0.1, head_length=0.1)

        plt.show()


if __name__ == '__main__':
    # Initialize the GMM environment
    means = [np.array([1, 1]), np.array([-1, -1])]  # Example means
    covariances = [np.eye(2), np.eye(2)]  # Equal uncorrelated covariances
    weights = [0.5, 0.5]  # Equal weights

    gmm_env = GaussianMixtures(means, covariances, weights)
    print("showing countour plot")
    contour_plot = gmm_env.contour_plot()
    plt.show()
