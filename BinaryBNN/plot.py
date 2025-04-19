
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
samplers=['edula','edmala']
etas=[  0.05,0.08, 0.1,0.2, 0.5, 0.8,1.0,2.0,5.0, 8.0]



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
    plt.title('L2 Norm for various '+ r"$\eta$"+" "+str(s), fontsize=14)
    plt.show()
    plt.clf()

etas=[0.01, 0.02,  0.05,0.08, 0.1,0.2, 0.5, 0.8,1.0,2.0,5.0, 8.0,10.0]
edula_rmse_mean = [0.60751504, 0.64876074, 0.47278106, 0.47379276, 0.4740749, 0.47583017, 0.47639367, 0.47770935,
                   0.47723475, 0.47735035, 0.47834504, 0.47747493, 0.4772311]
edula_rmse_std = [0.010722064, 0.024461633, 0.008134299, 0.0077392287, 0.007932165, 0.007846624, 0.0077003567,
                  0.0070010023, 0.0060746036, 0.0070548616, 0.0066014836, 0.007767846, 0.0067420495]

# EDMALA Test RMSE Mean and Std
edmala_rmse_mean = [0.5099684, 0.5099684, 0.5099684, 0.5071129, 0.4787498, 0.4720919, 0.47652632,
                    0.47678378, 0.47709295, 0.47868204, 0.4773827, 0.4780279, 0.4780923]
edmala_rmse_std = [5.9604645e-08, 5.9604645e-08, 5.9604645e-08, 0.0020571067, 0.010750137, 0.0058419,
                   0.0057884115, 0.0067517143, 0.0068211337, 0.0063716294, 0.006419037, 0.0065125814, 0.007341961]

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(etas, edula_rmse_mean, yerr=edula_rmse_std, label='EDULA', fmt='o-', capsize=5)
plt.errorbar(etas, edmala_rmse_mean, yerr=edmala_rmse_std, label='EDMALA', fmt='s-', capsize=5)

plt.xlabel(r"$\eta$", fontsize=15)
plt.ylabel('Validation RMSE', fontsize=15)
plt.title('Impact on validation RMSE with decreasing coupling',fontsize=18)
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.tight_layout()
plt.show()

samplers=['edmala']
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