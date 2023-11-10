import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.bayesian_learning import BayesianLearningEnvironment
from hessian_adaptive_diffusion import HessianBasedAdaptiveDiffusion
from schedule import ConstantSchedule

baseline_data = np.load("baseline.npz")

observations = baseline_data["obs"]
post_means = baseline_data["post_means"]

# placeholders here
alpha_scd = ConstantSchedule(1)
beta_scd = ConstantSchedule(1)

# Store the results for each epsilon
eps_results = {}

# for eps in [0.05, 0.1, 0.2, 0.5]:
for eps in [0.01, 0.015, 0.02]:
    print("start with eps", eps)
    eps_scd = ConstantSchedule(eps)
    env = BayesianLearningEnvironment(2, observations)
    classic_langevin = HessianBasedAdaptiveDiffusion(env, eps_scd, alpha_scd, beta_scd, classic=True)

    theta_means = []
    samples = []

    for _ in tqdm(range(500000)):
        theta = classic_langevin.step()
        samples.append(theta)

        if len(samples) % 1000 == 0:
            theta_means.append(np.mean(samples, axis=0))

    print("accept rate:", classic_langevin.accepted / 50000)

    eps_results[eps] = theta_means

# Plotting
fig, ax = plt.figure(figsize=(10, 6))
for eps, means in eps_results.items():
    means = np.array(means)
    plt.plot(means[:, 1], label=f'eps={eps}')

plt.plot(post_means[1], label='Ground Truth', linestyle='--')
plt.xlabel('Iterations (in thousands)')
plt.ylabel('Mean of Theta 1')
plt.title('Convergence of Theta Mean for Different Epsilons')
plt.legend()
plt.show()
fig.savefig("classic_eps.png")

