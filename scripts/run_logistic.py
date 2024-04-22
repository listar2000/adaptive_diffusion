import numpy as np
from tqdm import tqdm

from envs.bayesian_logistics import load_caravan_dataset, load_diabetes_dataset, BayesianLogistics
from samplers.metropolis_hastings import MetropolisHastingsSampler
from samplers.adaptive_lmc import AdaptiveLMCSampler
from samplers.fisher_lmc import FisherLMCSampler
from samplers.vanilla_lmc import VanillaLMCSampler
from samplers.riemann_lmc import RiemannLMCSampler



MH_BASELINE_STD = {"caravan": 0.06, "diabetes": 0.13}


def test_mh_baseline(mode="caravan"):
    np.random.seed(2024)
    total = 2000

    if mode == "caravan":
        X, y = load_caravan_dataset()
    else:
        X, y = load_diabetes_dataset()
    env = BayesianLogistics(X=X, y=y, alpha=4)

    stds = np.linspace(0.05, 0.1, 20)
    for std in stds:
        sampler = MetropolisHastingsSampler(environment=env, proposal_std=std)
        for _ in range(total):
            sampler.step()
        print(std, ":", sampler.accept / total)



def generate_mh_baseline(TRIALS=100000, mode="caravan"):
    np.random.seed(2024)
    total = int(TRIALS * 1.1)
    burn_in = total - TRIALS

    if mode == "caravan":
        X, y = load_caravan_dataset()
        env = BayesianLogistics(X=X, y=y, alpha=4)
    else:
        X, y = load_diabetes_dataset()
        env = BayesianLogistics(X=X, y=y, alpha=100)

    file_name = f"logistic_{mode}_baseline.npz"

    std = MH_BASELINE_STD[mode]
    sampler = MetropolisHastingsSampler(environment=env, proposal_std=std)

    samples = [sampler.step() for _ in tqdm(range(total))]
    samples = np.array(samples[burn_in:])
    with open("../data/" + file_name, "wb") as f:
        np.save(f, samples)
        print(f"successfully saving MH baseline in {mode}")

    means = samples.mean(axis=0)
    with open(f"../data/{mode}_mean.npz", "wb") as f:
        np.save(f, means)
        print("successfully saving means")


def generate_experiments(TRIAL_TIMES=10, ENSEMBLE=5, N=10000, mode="caravan"):
    import time
    prefix = f"logistic_{mode}_"

    if mode == "caravan":
        X, y = load_caravan_dataset()
        env = BayesianLogistics(X=X, y=y, alpha=4)
    else:
        X, y = load_diabetes_dataset()
        env = BayesianLogistics(X=X, y=y, alpha=100)

    TOTAL_N = int(N * 1.2)
    BURNIN = TOTAL_N - N

    fisher_esbs = []
    ada_esbs = []
    rieman_esbs = []
    # mh_esbs = []
    vanilla_esbs = []
    for trial in range(TRIAL_TIMES):
        fisher_esb = []
        ada_esb = []
        # mh_esb = []
        vanilla_esb = []
        rieman_esb = []
        for e in tqdm(range(ENSEMBLE)):
            t = time.time()
            times = {}
            fisher = FisherLMCSampler(env=env)
            fisher_esb.append([fisher.step() for _ in range(TOTAL_N)][BURNIN:])
            times["fisher"] = time.time() - t
            t = time.time()
            ada = AdaptiveLMCSampler(env=env)
            ada_esb.append([ada.step() for _ in range(TOTAL_N)][BURNIN:])
            times["ada"] = time.time() - t
            t = time.time()
            vanilla = VanillaLMCSampler(env=env)
            vanilla_esb.append([vanilla.step() for _ in range(TOTAL_N)][BURNIN:])
            times["vanilla"] = time.time() - t
            t = time.time()
            rieman = RiemannLMCSampler(env=env)
            rieman_esb.append([rieman.step() for _ in range(TOTAL_N)][BURNIN:])
            times["rieman"] = time.time() - t
            print(times)
            # mh = MetropolisHastingsSampler(environment=env, proposal_std=1)
            # mh_esb.append([mh.step() for _ in range(TOTAL_N)][BURNIN:])

        fisher_esbs.append(np.array(fisher_esb))
        ada_esbs.append(np.array(ada_esb))
        # mh_esbs.append(np.array(mh_esb))
        vanilla_esbs.append(np.array(vanilla_esb))
        rieman_esbs.append(np.array(rieman_esb))


    with open(f'../data/' + prefix + 'fisher.npy', 'wb') as f:
        np.save(f, fisher_esbs)
        print("saving fisher success")

    with open(f'../data/' + prefix + 'ada.npy', 'wb') as f:
        np.save(f, ada_esbs)
        print("saving ada success")

    with open(f'../data/' + prefix + 'rieman.npy', 'wb') as f:
        np.save(f, rieman_esbs)
        print("saving mh success")

    with open(f'../data/' + prefix + 'vanilla.npy', 'wb') as f:
        np.save(f, vanilla_esbs)
        print("saivng vanilla success")


if __name__ == "__main__":
    generate_experiments(N=10000, mode="diabetes")
