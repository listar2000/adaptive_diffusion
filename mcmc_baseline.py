# Generate samples
import numpy as np

from envs.bayesian_learning import BayesianLearningEnvironment
from samplers.metropolis_hastings import MetropolisHastingsSampler
from tqdm import tqdm

np.random.seed(2024)
observations = np.random.randn(100) + 0.5
environment = BayesianLearningEnvironment(observations)
mh_sampler = MetropolisHastingsSampler(environment, proposal_std=0.08)

# Run the sampler for a number of steps
samples = []
for _ in tqdm(range(5001000)):  # 100000 samples + 1000 burn-in
    sample = mh_sampler.step()
    samples.append(sample)

# Discard the burn-in samples
samples = np.array(samples[1000:])

np.savez("baseline.npz", obs=observations, post_means=np.mean(samples, axis=0))
#
# # Create the histogram data
# hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=30, density=True)
#
# # Construct arrays for the anchor positions of the bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0
#
# # Construct arrays with the dimensions for the bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#
# # Labels and titles
# ax.set_xlabel('Theta 1')
# ax.set_ylabel('Theta 2')
# ax.set_zlabel('Frequency')
# ax.set_title('3D Histogram of Posterior Samples')
#
# plt.show()