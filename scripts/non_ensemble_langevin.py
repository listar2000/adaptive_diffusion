import envs.bayesian_learning as bl
import envs.multi_modal as mm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_ground_truth_mean(environment):
    # Weighted mean of the individual Gaussians
    weighted_means = [w * m for w, m in zip(environment.weights, environment.means)]
    ground_truth_mean = np.sum(weighted_means, axis=0)
    return ground_truth_mean


def mean_deviation_over_time(samples, ground_truth_mean):
    deviations = []
    cumulative_sum = np.zeros_like(samples[0])

    for t in range(1, len(samples) + 1):
        cumulative_sum += samples[t - 1]
        empirical_mean = cumulative_sum / t
        deviation = np.linalg.norm(empirical_mean - ground_truth_mean)
        deviations.append(deviation)

    return deviations


def plot_mean_deviation(samples_list, environment, labels):
    ground_truth_mean = calculate_ground_truth_mean(environment)
    plt.figure(figsize=(10, 6))

    for samples, label in zip(samples_list, labels):
        deviations = mean_deviation_over_time(samples, ground_truth_mean)
        plt.plot(range(1, len(samples) + 1), deviations, label=label)

    plt.xlabel('Number of Samples (t)')
    plt.ylabel('Mean Deviation')
    plt.title('Mean Deviation Over Time for Multiple Samplers')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    TRAILS = 20
    N = 50000
    BURN_IN_RATIO = 0.1
    STORE_NAME = "best_langevin_sampler"

    means = [np.array([-2, -2]), np.array([2, 2])]  # Example 2D means
    cov_mat = np.array([[1.5, 0.5], [0.5, 1]])
    covariances = [cov_mat, cov_mat]  # 2x2 identity matrices for each component
    weights = [0.5, 0.5]  # Equal weights for each component

    gmm_env = mm.GaussianMixtures(means, covariances, weights)
    sampler = mm.LangevinSampler(gmm_env, epsilon=0.16, alpha=0.1)
    # sampler = mm.MetropolisHastingsSampler(gmm_env, proposal_std=1)
    true_mean = calculate_ground_truth_mean(gmm_env)
    print("true mean:", true_mean)

    all_results = np.empty((TRAILS, N))

    for i in tqdm(range(TRAILS)):
        samples = sampler.sample(int(N * (1 + BURN_IN_RATIO)))
        samples = samples[int(BURN_IN_RATIO * N):]
        deviations = mean_deviation_over_time(samples, true_mean)
        assert len(deviations) == N, print("length:", len(deviations))
        all_results[i] = deviations

    np.save(f"../data/bad_gmm/{STORE_NAME}.npy", all_results)




