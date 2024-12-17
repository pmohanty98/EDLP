import argparse
import math
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
from numpy.random import permutation

import tsp, samplers, block_samplers
import pandas as pd

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
from matplotlib.colors import Normalize
import time
import itertools
import random
from matplotlib.colors import ListedColormap

def swap_distance(path1, path2):
    """Calculates the number of swaps (2-opt moves) needed to convert path1 into path2."""
    path1=path1+(path1[0],)
    path2 = path2 + (path2[0],)

    distance = 0
    for i in range(len(path1)):
        for j in range(i + 1, len(path1)):
            # Check if the pairwise order is different in the two paths
            if (path1[i], path1[j]) != (path2[i], path2[j]):
                distance += 1
    return distance

def calculate_swap_diversity_unique(sampled_paths, best_path,model):
    """Calculates swap diversity and cost only for unique sampled paths.""" # To track unique paths
    costs = []
    diversities = []

    for path in sampled_paths:
        diversity = swap_distance( path, best_path)
        cost = model.forward(path).item()

        costs.append(cost*-1)
        diversities.append(diversity)

    return costs, diversities

def plot_cost_vs_diversity_unique(sampled_paths, best_path,model, sampler,dir):
    """Plots cost vs diversity (swap distance) for unique TSP paths."""
    costs, diversities = calculate_swap_diversity_unique(sampled_paths, best_path,model)

    np.save("{}/tsp_cost_scatterplot_{}.npy".format(dir, sampler), costs)
    np.save("{}/tsp_diversity_scatterplot_{}.npy".format(dir, sampler), diversities)

    plt.figure(figsize=(10, 6))
    plt.scatter(costs, diversities, c=diversities, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Diversity from Best Path (Swap Distance)")
    plt.xlabel('Cost')
    plt.ylabel('Diversity (Swap Distance from Best Path)')
    plt.title('Cost vs Path Diversity in TSP Sampling for '+str(sampler))
    plt.show()




def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = args.nchains

    dim = 8
    model = tsp.TSP8( dim=dim)
    model.to(device)
    t = args.sampler
    chain = []
    chain_a = []
    chain_1=[]
    chain_paths=[]

    cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    permutations = list(itertools.permutations(cities))
    #permutation=permutations[0]+(permutations[0][0],)
    y = permutations[0]

    if t == "dula":
        sampler = samplers.LangevinSamplerTSP(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.1, mh=False)
    elif t == "edula":
        sampler = samplers.ELangevinSamplerTSP(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.1, alpha_a=0.1, mh=False,
                                            eta=0.1)


    elif t == "dmala":
        sampler = samplers.LangevinSamplerTSP(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.2, mh=True)

    elif t == "edmala":
        sampler = samplers.ELangevinSamplerTSP(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.2, alpha_a=0.1, mh=True,
                                            eta=0.8)



    else:
        assert False, 'Not implemented'


    x = y
    x_a = torch.stack([model.city_positions[city] for city in x])

    for i in range(args.n_steps + 1):

        if i % args.thinning == 0:

            chain.append(x)
            if sampler.entropy == True:
                chain_a.append(x_a)

            if args.burnin < i:
                if not any(xhat == item for item in chain_paths):
                    chain_1.append(model.forward(xhat).item())
                    chain_paths.append(xhat)

        if (sampler.entropy == False):
            xhat = sampler.step(x, model)
            #print(xhat)
        else:
            sample_tuple = sampler.step(x, x_a.detach(), model)
            xhat = sample_tuple[0]
            xahat = sample_tuple[1].detach()

        x = xhat
        if (sampler.entropy == True):
            x_a = xahat


    if args.sampler == 'edmala' or args.sampler == 'edmala-glu' or args.sampler == 'dmala':
        print("Average Acceptance Probability: " + str(np.round(np.mean(sampler.a_s), decimals=3)))
        print("Standard Deviation: " + str(np.round(np.std(sampler.a_s), decimals=3)))


    #print(chain_paths)
    #print(chain_1)

    np.save("{}/tsp_cost_{}.npy".format(args.save_dir, args.sampler), -1*np.mean(chain_1))
    #p.save("{}/tsp_std_{}_{}.npy".format(args.save_dir, args.seed, args.sampler), np.std(chain_1))

    #print(np.mean(chain_1), np.std(chain_1))


    best_path = chain_paths[np.argmin([(-1)*model.forward(p).item() for p in chain_paths])]  # Find the best path
    plot_cost_vs_diversity_unique(chain_paths, best_path,model, args.sampler,args.save_dir)

    SD=[]
    for path in chain_paths:
        for path1 in chain_paths:
            if path == path1:
                continue
            SD.append(swap_distance(path,path1))

    np.save("{}/all_swap_{}.npy".format(args.save_dir, args.sampler), SD)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--burnin', type=int, default=2000)
    parser.add_argument('--n_steps', type=int, default=10000)  # 50,000
    parser.add_argument('--thinning', type=int, default=1)  # 1
    parser.add_argument('--interval', type=int, default=1000)  # 1000
    parser.add_argument('--viz', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--nchains', type=int, default=1)
    parser.add_argument('--sampler', type=str, default='edula')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--alpha_a', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=1)

    parser.add_argument('--save_dir', type=str, default="./TSP")
    args = parser.parse_args()
    main(args)
