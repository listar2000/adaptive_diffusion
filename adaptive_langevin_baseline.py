import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.bayesian_learning import BayesianLearningEnvironment
from hessian_adaptive_diffusion import HessianBasedAdaptiveDiffusion
from schedule import ConstantSchedule, ExponentialDecaySchedule

baseline_data = np.load("baseline.npz")

observations = baseline_data["obs"]
post_means = baseline_data["post_means"]

# placeholders here
alpha_scd = ConstantSchedule(0.01)
beta_scd = ConstantSchedule(1)
eps_scd = ConstantSchedule(0.01)

env = BayesianLearningEnvironment(2, observations)
adaptive_langevin = HessianBasedAdaptiveDiffusion(env, eps_scd, alpha_scd, beta_scd, classic=False)

theta_means = []
S_list = []
D_list = []
samples = []

for _ in tqdm(range(50000)):
    theta = adaptive_langevin.step()
    samples.append(theta)
    S_list.append(adaptive_langevin.S)
    D_list.append(adaptive_langevin.D[0, 1])

    if len(samples) % 1000 == 0:
        theta_means.append(np.mean(samples, axis=0))

print("accept rate:", adaptive_langevin.accepted / 50000)


