import numpy as np
import matplotlib.pyplot as plt

# File paths
data_files = ['vanilla_sampler.npy', 'best_langevin_sampler.npy', 'best_mh_sampler.npy']
labels = ['vanilla mh', 'vanilla lang', 'best lang']

# Initialize lists to hold means and SDs
means = []
sds = []

# Load, compute mean and SD for each file
for file in data_files:
    data = np.load(f'../data/bad_gmm/{file}')  # Shape of data is assumed to be (N, M)
    mean = np.mean(data, axis=0)  # Mean across M, results in a vector of shape (N,)
    sd = np.std(data, axis=0)  # SD across M, also results in a vector of shape (N,)
    means.append(mean)
    sds.append(sd)

# Plotting
plt.figure(figsize=(10, 6))

# Colors for the plots
colors = ['red', 'green', 'blue']

for i in range(len(data_files)):
    x = np.arange(len(means[i]))  # Assuming N is the same for all datasets
    plt.plot(x, means[i], label=labels[i], color=colors[i])
    plt.fill_between(x, means[i]-sds[i], means[i]+sds[i], color=colors[i], alpha=0.2)

plt.title('Means with ±1 SD')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()
