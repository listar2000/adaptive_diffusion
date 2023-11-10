"""
Bivariate Gaussian sampling with adaptive SGLD algorithm
"""
import numpy as np
from matplotlib import pyplot as plt


def gradient_hessian_u(x, mu, sigma, grad_only=False):
    """
    Compute gradient of U for bivariate Gaussian.
    mu: the mean vector (length 2)
    sigma: the 2x2 covariance matrix
    """
    # grads
    sigma1_square, rho_sigma1_sigma2, _, sigma2_square = sigma.flatten()
    simga1_sigma2 = np.sqrt(sigma1_square * sigma2_square)
    rho = rho_sigma1_sigma2 / simga1_sigma2
    rho_inv = 1 / (1 - rho ** 2)

    grad_1 = ((x[0] - mu[0]) / sigma1_square) - (rho * (x[1] - mu[1]) / simga1_sigma2)
    grad_2 = ((x[1] - mu[1]) / sigma2_square) - (rho * (x[0] - mu[0]) / simga1_sigma2)

    if grad_only:
        return rho_inv * np.array([grad_1, grad_2])

    # hessian
    hessian = np.array([[-1 / sigma1_square, rho / simga1_sigma2], [rho / simga1_sigma2, -1 / sigma2_square]])

    return rho_inv * np.array([grad_1, grad_2]), rho_inv * hessian


def log_likelihood(x, mu, sigma):
    return -0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)


def log_proposal(x_prime, x, grad_U, dt, Q):
    """Compute log proposal density for Langevin dynamics."""
    term = x_prime - x + dt * (grad_U + Q @ grad_U)
    return -np.sum(term ** 2) / (4 * dt)


def adaptive_langevin_dynamics(mu, sigma, steps, dt, alpha, verbose=False):
    """Simulate Langevin dynamics for bivariate Gaussian."""
    X = np.zeros((steps, 2))
    q_coefs, d_coefs = [1], [np.zeros(2)]
    current_log_like = log_likelihood(X[0], mu, sigma)
    success = 0

    for t in range(1, steps):
        grad_u, hessian_u = gradient_hessian_u(X[t - 1], mu, sigma)
        # set up the diffusion matrix
        Q = np.array([[0, q_coefs[-1]], [-q_coefs[-1], 0]])
        proposed_X = X[t - 1] - (grad_u + Q @ grad_u) * dt + np.sqrt(2 * dt) * np.random.randn(2)

        # MALA step
        proposed_log_like = log_likelihood(proposed_X, mu, sigma)
        forward_log_proposal = log_proposal(proposed_X, X[t - 1], grad_u, dt, Q)
        backward_grad = gradient_hessian_u(proposed_X, mu, sigma, grad_only=True)
        backward_log_proposal = \
            log_proposal(X[t - 1], proposed_X, backward_grad, dt, Q)

        log_accept_ratio = proposed_log_like + backward_log_proposal - current_log_like - forward_log_proposal

        if np.log(np.random.rand()) < log_accept_ratio:
            X[t] = proposed_X
            current_log_like = proposed_log_like
            success += 1

            # update the cofficients (adaptive part)
            # d_coefs is of size 2
            q = q_coefs[-1] - alpha * (grad_u @ d_coefs[-1])
            d = d_coefs[-1] - dt * ((np.eye(2) + Q) @ hessian_u @ d_coefs[-1]) - dt * np.array(
                [[0, 1], [-1, 0]]) @ grad_u

            q_coefs.append(q)
            d_coefs.append(d)
        else:
            X[t] = X[t - 1]
            q_coefs.append(q_coefs[-1])
            d_coefs.append(d_coefs[-1])

    return X, q_coefs, d_coefs, success
