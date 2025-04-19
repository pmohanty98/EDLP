import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import seaborn as sns
import pandas as pd


samplers=['dula','edula', 'dmala','edmala']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

i=0
for s in samplers:
    mean=np.load('tsp_cost_'+s+'.npy')
    sd = np.load('tsp_std_' + s + '.npy')
    plt.bar([s], [mean], yerr=[sd], capsize=10, color=colors[i], edgecolor='black')
    i+=1

    # Labeling the plot
plt.ylabel('Cost',fontsize=14)
plt.xlabel('Sampler',fontsize=14)
plt.title(' Mean Cost and Standard Deviation',fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('cost'+".png", dpi=300, bbox_inches='tight')
plt.show()


i=0
for s in samplers:
    mean=np.mean(np.load("tsp_diversity_scatterplot_"+s+".npy"))
    sd = np.std(np.load("tsp_diversity_scatterplot_"+s+".npy"))
    plt.bar([s], [mean], yerr=[sd], capsize=10, color=colors[i], edgecolor='black')
    i+=1

    # Labeling the plot
plt.ylabel('Pairwise Mismatch Count',fontsize=13)
plt.xlabel('Sampler',fontsize=13)
plt.title('Mean PMC (Optimal) and Standard Deviation',fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('swap_optimal'+".png", dpi=300, bbox_inches='tight')
plt.show()


i=0
for s in samplers:
    mean=np.mean(np.load("all_swap_"+s+".npy"))
    sd = np.std(np.load("all_swap_"+s+".npy"))
    plt.bar([s], [mean], yerr=[sd], capsize=10, color=colors[i], edgecolor='black')
    i+=1

    # Labeling the plot
plt.ylabel('Pairwise Mismatch Count',fontsize=13)
plt.xlabel('Sampler',fontsize=13)
plt.title('Mean PMC(Overall) and Standard Deviation',fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('swap_overall'+".png", dpi=300, bbox_inches='tight')
plt.show()

# Load data for each sampler into a list
all_costs = []
for s in samplers:
    # Replace with your actual filename format
    costs = np.load(f"tsp_cost_scatterplot_{s}.npy")
    all_costs.append(costs)

# Create the violin plot
sampler_labels = []
cost_values = []
for sampler_name, cost_array in zip(samplers, all_costs):
    sampler_labels.extend([sampler_name] * len(cost_array))
    cost_values.extend(cost_array)

df = pd.DataFrame({"Sampler": sampler_labels, "Cost": cost_values})

# Create a violin plot

sns.violinplot(
    data=df,
    x="Sampler",
    y="Cost",
    inner="box",      # or "quartile", "stick", etc.
    cut=0,            # cut=0 can prevent the violin from extending beyond min/max
    palette=colors # or any palette you like
)
plt.ylabel('Cost',fontsize=14)
plt.xlabel('Sampler',fontsize=14)
plt.title("Violin Plots for TSP Costs", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('violin_cost'+".png", dpi=300, bbox_inches='tight')
plt.show()




all_costs = []
for s in samplers:
    # Replace with your actual filename format
    costs = np.load(f"tsp_diversity_scatterplot_{s}.npy")
    np.load("tsp_diversity_scatterplot_" + s + ".npy")
    all_costs.append(costs)

# Create the violin plot
sampler_labels = []
cost_values = []
for sampler_name, cost_array in zip(samplers, all_costs):
    sampler_labels.extend([sampler_name] * len(cost_array))
    cost_values.extend(cost_array)

df = pd.DataFrame({"Sampler": sampler_labels, "PMC": cost_values})

# Create a violin plot
plt.figure(figsize=(8,5))
sns.violinplot(
    data=df,
    x="Sampler",
    y="PMC",
    inner="box",      # or "quartile", "stick", etc.
    cut=0,            # cut=0 can prevent the violin from extending beyond min/max
    palette=colors # or any palette you like
)
plt.ylabel('Pairwise Mismatch Count',fontsize=14)
plt.xlabel('Sampler',fontsize=14)
plt.title("Violin Plots for Solution Similarity(Optimal)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig('violin_diversity'+".png", dpi=300, bbox_inches='tight')
plt.show()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']  # Add markers as needed for each sampler

# Initialize the figure for the joint plot
g = sns.JointGrid()

# Loop through each sampler and plot
for i, s in enumerate(samplers):
    # Load data for the sampler
    costs = np.load(f"tsp_cost_scatterplot_{s}.npy")
    diversities = np.load(f"tsp_diversity_scatterplot_{s}.npy")

    # Scatter plot for each sampler
    g.ax_joint.scatter(costs, diversities, alpha=0.8, color=colors[i], label=s, s=30, edgecolors='black', marker=markers[i])

    # Add vertical lines for min and max cost
    #g.ax_joint.axvline(x=np.min(costs), color=colors[i], linestyle='-', alpha=0.7)
    #g.ax_joint.axvline(x=np.max(costs), color=colors[i], linestyle='-', alpha=0.7)

    # Add horizontal line for max diversity
    #g.ax_joint.axhline(y=28, color='black', linestyle='--', alpha=1)
    #g.ax_joint.axhline(y=0, color='black', linestyle='--', alpha=1)

    # KDE plots for each sampler’s cost and diversity
    sns.kdeplot(costs, ax=g.ax_marg_x, color=colors[i], fill=True, bw_adjust=0.2, alpha=0.2)
    sns.kdeplot(diversities, ax=g.ax_marg_y, color=colors[i], fill=True, bw_adjust=0.2, alpha=0.2, vertical=True)

# Set labels, title, and legend
g.ax_joint.set_xlabel('Cost',fontsize=13, labelpad=0.2)
g.ax_joint.set_ylabel('PMC from Best Path',fontsize=13, labelpad=0.1)
#g.ax_joint.set_aspect(1.0 / g.ax_joint.get_data_ratio(), adjustable='box')
g.ax_joint.legend(title='Sampler', loc='lower right', fontsize=12)
g.ax_joint.set_title('Cost vs Mean PMC (Optimal) in TSP Sampling', fontsize=11.5, pad=80)

# Display the plot
plt.savefig('scatterplot_tsp'+".png", dpi=300, bbox_inches='tight')

plt.show()

