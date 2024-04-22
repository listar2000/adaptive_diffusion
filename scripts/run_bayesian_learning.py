import numpy as np
from tqdm import tqdm

from envs.bayesian_learning import BayesianLearningHard, BayesianLearningSimple
from samplers.metropolis_hastings import MetropolisHastingsSampler
from samplers.adaptive_lmc import AdaptiveLMCSampler
from samplers.fisher_lmc import FisherLMCSampler
from samplers.vanilla_lmc import VanillaLMCSampler


def load_bayesian_learning(mode="easy"):
    if mode == "easy":
        data = np.load("../data/bayesian_learning_easy.npz")
        env = BayesianLearningSimple(data=data)
    else:
        data = np.load("../data/bayesian_learning_hard.npz")
        env = BayesianLearningHard(data=data)
    return env


MH_BASELINE_STD = {"easy": 0.59, "hard": 0.28}


def generate_mh_baseline(TRIALS=100000, mode="easy"):
    np.random.seed(2024)
    total = int(TRIALS * 1.1)
    burn_in = total - TRIALS

    env = load_bayesian_learning(mode)
    file_name = f"bayesian_learning_{mode}_baseline.npz"

    std = MH_BASELINE_STD[mode]
    sampler = MetropolisHastingsSampler(environment=env, proposal_std=std)

    samples = [sampler.step() for _ in tqdm(range(total))]
    samples = np.array(samples[burn_in:])
    with open("../data/bayesian_learning/" + file_name, "wb") as f:
        np.save(f, samples)
        print(f"successfully saving MH baseline in {mode}")

    means = samples.mean(axis=0)
    with open(f"../data/bayesian_learning/{mode}_mean.npz", "wb") as f:
        np.save(f, means)
        print("successfully saving means")


def generate_experiments(TRIAL_TIMES=10, ENSEMBLE=5, N=10000, mode="easy"):

    prefix = f"bayesian_learning_{mode}_"

    env = load_bayesian_learning(mode=mode)

    TOTAL_N = int(N * 1.2)
    BURNIN = TOTAL_N - N

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
            fisher_esb.append([fisher.step() for _ in range(TOTAL_N)])
            ada = AdaptiveLMCSampler(env=env, init_eps=1)
            ada_esb.append([ada.step() for _ in range(TOTAL_N)])
            vanilla = VanillaLMCSampler(env=env, init_eps=1)
            vanilla_esb.append([vanilla.step() for _ in range(TOTAL_N)])
            mh = MetropolisHastingsSampler(environment=env, proposal_std=1)
            mh_esb.append([mh.step() for _ in range(TOTAL_N)])

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


if __name__ == "__main__":
    generate_mh_baseline(mode="easy")
    generate_mh_baseline(mode="hard")
