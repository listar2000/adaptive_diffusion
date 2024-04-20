import numpy as np


def estimate_autocorrelation(series):
    """Estimate autocorrelation using FFT for efficiency"""
    n = len(series)
    series = series - np.mean(series)
    f = np.fft.fft(series, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf


def calc_iat(series, M=None):
    """Compute the integrated autocorrelation time up to lag M"""
    acf = estimate_autocorrelation(series)
    if M is None:
        M = len(acf) // 2  # A simple heuristic
    return 1 + 2 * np.sum(acf[1:M+1])


if __name__ == "__main__":
    # Example of usage
    data_files = ['best_langevin_sampler.npy', 'best_mh_sampler.npy', 'vanilla_sampler.npy']
    for file in data_files:
        data = np.load(f"../data/bad_gmm/{file}")  # Assuming shape is (N, M), M chains of N samples each
        iats = []
        for i in range(data.shape[1]):  # Loop over each chain
            tau = calc_iat(data[:, i])
            iats.append(tau)

        # Compute mean and SD of IATs for this file
        mean_iat = np.mean(iats)
        sd_iat = np.std(iats)

        print(f"{file}: Mean IAT = {mean_iat}, SD IAT = {sd_iat}")