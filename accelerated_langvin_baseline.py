import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.bayesian_learning import BayesianLearningEnvironment
from hessian_adaptive_diffusion import HessianBasedAdaptiveDiffusion
from schedule import ConstantSchedule

# Define the grid values for eps and s
eps_values = [0.01, 0.015, 0.02]
s_values = [0.01, 0.1, 1]

baseline_data = np.load("baseline.npz")
# Ground truth theta for comparison
observations = baseline_data["obs"]
ground_truth_theta = baseline_data["post_means"]  # Replace with your actual ground truth theta

# Initialize schedules for alpha and beta
alpha_scd = ConstantSchedule(1)
beta_scd = ConstantSchedule(1)
# Store the results
grid_search_means = {}
grid_search_results = {}

# Perform the grid search
for eps in eps_values:
    for s in s_values:
        print("start with ", eps, s)
        # Create the anti-symmetric matrix S
        S = np.array([[0, -s], [s, 0]])

        # Initialize the environment and algorithm
        env = BayesianLearningEnvironment(2, observations)
        eps_scd = ConstantSchedule(eps)
        accelerate_langevin = HessianBasedAdaptiveDiffusion(env, eps_scd, alpha_scd, beta_scd, S, classic=True)

        # Record means every 1000 steps
        theta_means = []
        samples = []

        for i in tqdm(range(500000)):
            theta = accelerate_langevin.step()
            samples.append(theta)

            if (i + 1) % 1000 == 0:
                theta_means.append(np.mean(samples, axis=0)[0])  # Mean of theta_1

        print("accepted", accelerate_langevin.accepted / 500000)
        # Store the results
        grid_search_means[(eps, s)] = theta_means

# Plotting
plt.figure(figsize=(12, 8))
for (eps, s), means in grid_search_means.items():
    plt.plot(range(1000, 501000, 1000), means, label=f'eps={eps}, s={s}')

plt.axhline(y=ground_truth_theta[0], color='r', linestyle='--', label='Ground Truth')
plt.xlabel('Iterations')
plt.ylabel('Mean of Theta 1')
plt.title('Convergence of Theta Mean for Different Eps and S Combinations')
plt.legend()
plt.show()
