from matplotlib import pyplot as plt

from envs.multi_modal import GaussianMixtures
from samplers.vanilla_lmc import VanillaLMCSampler
from samplers.metropolis_hastings import MetropolisHastingsSampler
from samplers.fisher_lmc import FisherLMCSampler
from samplers.adaptive_lmc import AdaptiveLMCSampler
from tqdm import tqdm
import numpy as np


def load_gmm_env_1():
    means = [np.array([1, 1]), np.array([-1, -1])]  # Example means
    covariances = [np.eye(2), np.eye(2)]  # Equal uncorrelated covariances
    weights = [0.5, 0.5]  # Equal weights

    gmm_env = GaussianMixtures(means, covariances, weights)
    ground_truth = np.zeros(2)
    return gmm_env, ground_truth


def load_gmm_env_2():
    means = [np.array([-2, -2]), np.array([2, 2])]  # Example 2D means
    cov_mat = np.array([[1.5, 0.5], [0.5, 1]])
    covariances = [cov_mat, cov_mat]  # 2x2 identity matrices for each component
    weights = [0.5, 0.5]  # Equal weights for each component

    gmm_env = GaussianMixtures(means, covariances, weights)
    ground_truth = np.zeros(2)
    return gmm_env, ground_truth


def load_gmm_env_3():
    means = [np.array([0, -2]), np.array([-2, 2]), np.array([2, 2])]  # Example means
    covariances = [np.eye(2), 2 * np.eye(2), 2 * np.eye(2)]  # Equal uncorrelated covariances
    weights = [1/3, 1/3, 1/3]  # Equal weights

    gmm_env = GaussianMixtures(means, covariances, weights)
    ground_truth = np.array([0, 2/3])
    return gmm_env, ground_truth


def generate_dataset():
    TRIAL_TIMES = 10
    ENSEMBLE = 5
    N = 10000

    prefix = "gmm_3_"
    env, ground_truth = load_gmm_env_3()

    fisher_esbs = []
    ada_esbs = []
    mh_esbs = []
    vanilla_esbs = []
    for trial in range(TRIAL_TIMES):
        fisher_esb = []
        ada_esb = []
        mh_esb = []
        vanilla_esb = []
        for e in tqdm(range(ENSEMBLE)):
            fisher = FisherLMCSampler(env=env, init_eps=1)
            fisher_esb.append([fisher.step() for _ in range(N)])
            ada = AdaptiveLMCSampler(env=env, init_eps=1)
            ada_esb.append([ada.step() for _ in range(N)])
            vanilla = VanillaLMCSampler(env=env, init_eps=1)
            vanilla_esb.append([vanilla.step() for _ in range(N)])
            mh = MetropolisHastingsSampler(environment=env, proposal_std=1)
            mh_esb.append([mh.step() for _ in range(N)])

        fisher_esbs.append(np.array(fisher_esb))
        ada_esbs.append(np.array(ada_esb))
        mh_esbs.append(np.array(mh_esb))
        vanilla_esbs.append(np.array(vanilla_esb))

    with open(f'../data/gmm/' + prefix + 'fisher.npy', 'wb') as f:
        np.save(f, fisher_esbs)
        print("saving fisher success")

    with open(f'../data/gmm/' + prefix + 'ada.npy', 'wb') as f:
        np.save(f, ada_esbs)
        print("saving ada success")

    with open(f'../data/gmm/' + prefix + 'mh.npy', 'wb') as f:
        np.save(f, mh_esbs)
        print("saving mh success")

    with open(f'../data/gmm/' + prefix + 'vanilla.npy', 'wb') as f:
        np.save(f, vanilla_esbs)
        print("saivng vanilla success")


def plot_and_save_gmm_contours():
    env1, _ = load_gmm_env_1()
    env2, _ = load_gmm_env_2()
    env3, _ = load_gmm_env_3()

    plt.tight_layout()
    fig1, ax1 = env1.contour_plot(title="GMM 1")
    plt.savefig("gmm_1.png")
    fig2, ax2 = env2.contour_plot(title="GMM 2")
    plt.savefig("gmm_2.png")
    fig3, ax3 = env3.contour_plot(title="GMM 3")
    plt.savefig("gmm_3.png")


def plot_deviation_plot_gmm1():
    from metrics.metrics import deviation_from_param
    # File paths
    data_files = ['gmm_2_ada.npy', 'gmm_2_fisher.npy', 'gmm_2_vanilla.npy', 'gmm_2_mh.npy']
    labels = ['Ada', 'Fisher', 'Vanilla', 'MH']

    # Initialize lists to hold means and SDs
    means = []
    sds = []

    # Load, compute mean and SD for each file
    for file in data_files:
        data = np.load(f'../data/gmm/{file}')  # Shape of data is assumed to be (N, M)
        data = data.reshape(-1, 10000, 2)

        devs = np.zeros((data.shape[0], 100))
        for i in tqdm(range(100)):
            idx = min(i * 100 + 1, data.shape[1])
            devs[:, i] = np.sqrt(data[:, :idx, :].mean(axis=1) ** 2).sum(axis=1)

        mean = np.mean(devs, axis=0)  # Mean across M, results in a vector of shape (N,)
        sd = np.std(devs, axis=0)  # SD across M, also results in a vector of shape (N,)
        means.append(mean)
        sds.append(sd)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Colors for the plots
    colors = ['red', 'green', 'blue', 'orange']

    for i in range(4):
        x = np.arange(len(means[i])) * 100  # Assuming N is the same for all datasets
        plt.plot(x, means[i], label=labels[i], color=colors[i])
        plt.fill_between(x, means[i] - sds[i], means[i] + sds[i], color=colors[i], alpha=0.1)

    plt.title('GMM 2: Means with ±1 SD')
    plt.xlabel('# of iterations')
    plt.ylabel('Deviation')
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig("gmm_2_deviation.png")
    return means, sds


if __name__ == '__main__':
    # import glob
    # from metrics.metrics import univariate_ess
    #
    # files = glob.glob("../data/gmm/*.npy")
    #
    # for file in files:
    #     with open(file, 'rb') as f:
    #         print(file)
    #         dataset = np.load(f)
    #         ess = []
    #         for data in dataset:
    #             ess.append(univariate_ess(data))
    #
    #         ess = np.array(ess)
    #         print("mean:", ess.mean(axis=0).mean())
    #         print("sd:", ess.std(axis=0).mean())
    means, stds = plot_deviation_plot_gmm1()
