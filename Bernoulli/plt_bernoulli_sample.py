import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label

sns.set(color_codes=True)
sns.set_style("whitegrid")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np

eg1=np.load('eigenspectrum_dmala_.npy')
plt.hist(eg1, bins=30,color='lightblue', edgecolor='black', alpha=0.5, label='dmala')
eg1=np.load('eigenspectrum_edmala_.npy')
plt.hist(eg1, bins=30,color='pink', edgecolor='black', alpha=0.5, label='edmala')
plt.xlabel('Eigenvalue Magnitude', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.tight_layout()
plt.legend()
plt.show()
plt.clf()

"""
samplers=['edmala']
etas=[  0.02, 0.05,0.08, 0.1,0.2, 0.5, 0.8,1,2,5]
for s in samplers:
    for eta in etas:
        mean=np.load('sensitivity_analysis_'+str(eta)+'_'+str(s)+'_.npy')
        stde=np.load('sensitivity_analysis_se_'+str(eta)+'_'+str(s)+'_.npy')
        N=range(0,len(mean))
        N= [t * 1 for t in N]
        plt.plot(N, mean,marker='', label=str(eta))
        plt.fill_between(N, mean - stde, mean + stde, alpha=0.3)

    plt.legend(loc='best',fontsize='x-small')
    plt.xlabel('Iterations', fontsize=12.5)
    plt.ylabel(r"$\mathcal{F}(\theta_a, \eta)$", fontsize=12.5)
    plt.title('Local Entropy for various '+ r"$\eta$", fontsize=14)
    plt.show()
    plt.clf()




for s in samplers:
    for eta in etas:
        mean=np.load('l2norm_'+str(eta)+'_'+str(s)+'_.npy')
        stde=np.load('l2norm_se_'+str(eta)+'_'+str(s)+'_.npy')
        N=range(0,len(mean))
        N= [t * 1 for t in N]
        plt.plot(N, mean,marker='', label=str(eta))
        plt.fill_between(N, mean - stde, mean + stde, alpha=0.3)

    plt.legend(loc='best',fontsize='x-small')
    plt.xlabel('Iterations', fontsize=12.5)
    plt.ylabel(r"$||\theta_a, \theta||$", fontsize=12.5)
    plt.title('L2 Norm for various '+ r"$\eta$", fontsize=14)
    plt.show()
    plt.clf()

mean=[]
std=[]
for s in samplers:
    for eta in etas:
        data=np.load('flat_mode_probability_'+str(eta)+'_'+str(s)+'_.npy')
        mean.append(np.mean(data))
        std.append(np.std(data))
        print(np.mean(data), np.std(data),eta)
    N=range(0,len(mean))
    N= [t * 1 for t in N]
    # Plot with error bars
    plt.figure(figsize=(7, 4))
    x = np.arange(len(mean))  # x-axis points (0, 1, 2, ...)
    plt.errorbar(x, mean, yerr=std, fmt='o-', capsize=5, label="Mean ± Std Dev")

    # Optional: add labels and style
    plt.title("Effect on Flat Mode identification with decreasing coupling")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Average Flat Mode citing Probability")
    plt.grid(True)
    plt.legend()
    plt.xticks(x, etas)
    plt.tight_layout()
    plt.show()

    plt.clf()

for s in samplers:
    data=[]
    for eta in etas:
        l=np.load('acceptance_ratio_'+str(eta)+'_'+str(s)+'_.npy')
        data.append(l)
    data=np.array(data)
    means = data[:, 0]
    std_devs = data[:, 1]
    x = np.arange(len(means))  # x-axis points (0, 1, 2, ...)

    # Plot with error bars
    plt.figure(figsize=(7, 4))
    plt.errorbar(x, means, yerr=std_devs, fmt='o-', capsize=5, label="Mean ± Std Dev")

    # Optional: add labels and style
    plt.title("Effect on Acceptance probability with decreasing coupling")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Average MH Acceptance Probability")
    plt.grid(True)
    plt.legend()
    plt.xticks(x,etas)
    plt.tight_layout()
    plt.show()

    plt.clf()
"""