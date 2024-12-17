import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data
data = {
    'Cost': [14, 15, 18, 17, 18, 19, 40, 21, 22, 23, 24],
    'Diversity': [5, 20, 15, 20, 25, 30, 35, 10, 15, 10, 25],
    'Sampler': ['dula', 'edula', 'dmala', 'edmala', 'dula', 'edula', 'dmala', 'edmala', 'dula', 'edula', 'dmala']
}

df = pd.DataFrame(data)

# Define color palette
palette = {
    'dula': 'red', 'edula': 'orange', 'dmala': 'green', 'edmala': 'purple'
}

# Create jointplot and get the JointGrid object
g = sns.jointplot(
    x='Cost', y='Diversity', data=df, hue='Sampler', kind='scatter', palette=palette,
    marginal_kws=dict(fill=True)
)

# Set alpha for KDE plots
alpha_value = 0.2  # Adjust alpha as needed

# Add KDEs along x and y axes for each sampler with specified alpha
for sampler, color in palette.items():
    subset = df[df['Sampler'] == sampler]
    sns.kdeplot(subset['Cost'], ax=g.ax_marg_x, color=color, fill=True, bw_adjust=0.3, alpha=alpha_value)
    sns.kdeplot(subset['Diversity'], ax=g.ax_marg_y, color=color, fill=True, bw_adjust=0.3, alpha=alpha_value, vertical=True)

# Adjust legend placement within the plot
plt.legend(title='Sampler', loc='upper right', bbox_to_anchor=(1, 1))

# Apply tight layout and display plot
plt.tight_layout()
plt.show()