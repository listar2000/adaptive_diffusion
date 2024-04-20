from typing import *

import numpy as np
from numpy.fft import fft, ifft


def averaged_move_distance(samples: np.ndarray, warmup: Union[float, int] = 0.1):
    if isinstance(warmup, float):
        warmup = int(samples.shape[0] * warmup)
    assert warmup < samples.shape[0], "warmup must be strictly less than total samples size"
    samples_to_calc = samples[warmup:, :]
    diffs = np.diff(samples_to_calc, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.mean(distances)


def fft_autocorr(chain):
    """
    Compute autocorrelations using FFT for a single chain.
    """
    n = chain.size
    f = fft(chain - np.mean(chain), n=2*n)
    acf = ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf


def univariate_ess(batch_samples: np.ndarray, warmup: Union[float, int] = 0.1):
    if isinstance(warmup, float):
        warmup = int(batch_samples.shape[1] * warmup)
    assert warmup < batch_samples.shape[1], "warmup must be strictly less than total samples size"

    chains = batch_samples[:, warmup:, :]
    M, N, D = chains.shape
    ess_per_param = np.zeros(D)

    for d in range(D):
        param_chains = chains[:, :, d]

        # Calculate W and B for this parameter
        s_square = np.var(param_chains, axis=1, ddof=1)
        W = np.mean(s_square)
        chain_means = np.mean(param_chains, axis=1)
        B = N * np.var(chain_means, ddof=1)
        # # Calculate \hat{V} using the \hat{R} method
        V_hat = (N - 1) / N * W + B / N

        # Geyer's truncation approach for ESS
        var_chains = param_chains - np.mean(param_chains, axis=1, keepdims=True)
        # autocorr = np.array([np.correlate(var_chains[m], var_chains[m], mode='full')
        #                      for m in range(M)])[:, N - 1:] / (N - np.arange(N))
        autocorr = np.array([fft_autocorr(chain) for chain in param_chains])
        rho = 1 - (W - np.mean(autocorr * s_square[:, np.newaxis], axis=0)) / V_hat
        positive_rho = rho[::2][:np.argmax(rho[::2] <= 0) + 1]

        # Monotone sequence estimator for ESS
        ess = M * N / (-1 + 2 * np.sum(positive_rho))
        ess_per_param[d] = ess

    return ess_per_param


def deviation_from_param(samples: np.ndarray, func: Callable, ground_truth, ord=None,
                         warmup: Union[float, int] = 0.1):
    if isinstance(warmup, float):
        warmup = int(samples.shape[0] * warmup)
    assert warmup < samples.shape[0], "warmup must be strictly less than total samples size"
    samples_to_calc = samples[warmup:, :]

    devs = np.zeros(len(samples_to_calc))

    for i in range(len(samples_to_calc)):
        devs[i] = np.linalg.norm(func(samples_to_calc[:(i+1), :]) - ground_truth, ord=ord)

    return devs
