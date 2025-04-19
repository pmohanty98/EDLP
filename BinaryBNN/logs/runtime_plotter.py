import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory where samplers are stored
base_dir = 'adult'  # Replace with the actual path to the "adult" folder

# Target order for samplers
target_order = ['gibbs', 'gwg', 'edula-glu', 'dula', 'edula', 'edmala-glu', 'dmala', 'edmala']
sampler_times = {}

# Traverse through each sampler directory and gather times for each sampler
for sampler_dir in os.listdir(base_dir):
    sampler_path = os.path.join(base_dir, sampler_dir)
    if os.path.isdir(sampler_path):
        time_file = os.path.join(sampler_path, 'time.npy')

        if os.path.exists(time_file):
            times = np.load(time_file)
            # Remove the suffix '_-1_0' to match target_order
            clean_sampler_name = sampler_dir.replace('_-1_0', '')
            mean_time = np.mean(times)
            std_dev_time = np.std(times)
            print(clean_sampler_name, mean_time, std_dev_time)
            sampler_times[clean_sampler_name] = (mean_time, std_dev_time)

# Sort the sampler times according to the target order
sorted_means = [sampler_times[sampler][0] for sampler in target_order if sampler in sampler_times]
sorted_std_devs = [sampler_times[sampler][1] for sampler in target_order if sampler in sampler_times]

# Define colors for each bar in the order specified
colors = plt.cm.Pastel1(np.linspace(0, 1, len(target_order)))  # Using a colormap for different colors

# Plot the bar plot with error bars for standard deviation
plt.figure(figsize=(12, 8))
plt.bar(target_order, sorted_means, yerr=sorted_std_devs, color=colors, capsize=5, edgecolor='black')
plt.xlabel('Sampler', fontsize=16)
plt.ylabel('Mean Elapsed Time (seconds)', fontsize=16)
plt.title('Mean Time and Standard Deviation for Each Sampler', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(rotation=45)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig("bnn_runtime_appendix.png", dpi=300, bbox_inches='tight')

#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




# Target order for samplers
target_order = ['gibbs', 'gwg' ,'dula', 'edula', 'dmala', 'edmala']
sampler_times = {}

# Traverse through each sampler directory and gather times for each sampler
for sampler_dir in os.listdir(base_dir):
    sampler_path = os.path.join(base_dir, sampler_dir)
    if os.path.isdir(sampler_path):
        time_file = os.path.join(sampler_path, 'time.npy')

        if os.path.exists(time_file):
            times = np.load(time_file)
            # Remove the suffix '_-1_0' to match target_order
            clean_sampler_name = sampler_dir.replace('_-1_0', '')
            mean_time = np.mean(times)
            std_dev_time = np.std(times)
            sampler_times[clean_sampler_name] = (mean_time, std_dev_time)

# Sort the sampler times according to the target order
sorted_means = [sampler_times[sampler][0] for sampler in target_order if sampler in sampler_times]
sorted_std_devs = [sampler_times[sampler][1] for sampler in target_order if sampler in sampler_times]

# Define colors for each bar in the order specified
colors = plt.cm.Pastel2(np.linspace(0, 1, len(target_order)))  # Using a colormap for different colors

# Plot the bar plot with error bars for standard deviation
plt.figure(figsize=(12, 8))
plt.bar(target_order, sorted_means, yerr=sorted_std_devs, color=colors, capsize=5, edgecolor='black')
plt.xlabel('Sampler', fontsize=16)
plt.ylabel('Mean Elapsed Time (seconds)', fontsize=16)
plt.title('Mean Time and Standard Deviation for Each Sampler', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(rotation=45)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig("bnn_runtime.png", dpi=300, bbox_inches='tight')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


