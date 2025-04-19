import argparse
import math
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import bernoulli_distribution, samplers, block_samplers
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.patches import Polygon, Rectangle
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
from matplotlib.colors import Normalize
import time
import itertools
import random
from matplotlib import colors
from matplotlib.colors import ListedColormap
from collections import Counter


def compute_hessian_for_sample(model, theta):
    """
    Computes the Hessian of U(theta) with respect to theta for a single chain.

    Parameters:
      theta: torch tensor of shape (1, d) (i.e. a batch of one),
             where d is the number of parameters. The model accepts a 2D tensor.

    Returns:
      Hessian as a NumPy array of shape (d, d).
    """
    import torch

    # Ensure theta is a torch tensor and has shape (1, d)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)  # Make it (1, d)

    # Clone and set requires_grad=True
    theta = theta.clone().detach().requires_grad_(True)

    # Call the model (which expects a 2D tensor) and get the energy.
    U_theta = model(theta)  # Expect shape (1,) or (1,1)
    # Extract the scalar energy from the batch dimension.
    U_scalar = U_theta[0]

    # Compute the gradient of U_scalar with respect to theta.
    grad = torch.autograd.grad(U_scalar, theta, create_graph=True)[0]  # shape (1, d)

    d = theta.shape[1]  # number of parameters
    hessian = torch.zeros(d, d, dtype=theta.dtype, device=theta.device)

    # Compute second derivatives for each parameter.
    for i in range(d):
        # grad[0, i] is a scalar; compute its gradient w.r.t. theta.
        grad2 = torch.autograd.grad(grad[0, i], theta,
                                    grad_outputs=torch.ones_like(grad[0, i]),
                                    retain_graph=True)[0]  # shape (1, d)
        # Remove the batch dimension
        hessian[i] = grad2.squeeze(0)

    return hessian.cpu().numpy()


def compute_hessian_batch(model, theta_batch):
    """
    Computes the Hessian for each sample in a batch.

    theta_batch: torch tensor of shape (n, d)
    Returns: a list of Hessians (each of shape (d, d)).
    """
    hessians = []
    for i in range(theta_batch.shape[0]):
        sample = theta_batch[i].unsqueeze(0)  # shape becomes (1, d)
        hessians.append(compute_hessian_for_sample(model, sample))
    return hessians

def state_probability(chain_list, target_states):
    """
    Computes the fraction (probability) that each snapshot in chain_list contains a state that is in the target_states list.

    Parameters:
        chain_list : list of arrays
            Each element is an array of shape (n, 4), representing the discrete states of n chains at a given iteration.
        target_states : list of lists or arrays
            A list of target states (each state is a list/array of length 4).
            For example: [[1, 0, 0, 0], [1, 1, 0, 0]]

    Returns:
        probs : numpy array
            Array of probabilities (fraction of chains with a state in target_states) for each iteration.
    """
    # Convert each target state to a tuple and store in a set for fast lookup.
    target_set = {tuple(state) for state in target_states}
    probs = []

    # Iterate through each snapshot in the chain_list.
    for sample in chain_list:
        # Ensure sample is a NumPy array
        sample = np.array(sample)
        # For each chain, convert the state (row) to a tuple and check if it is in target_set.
        matches = [tuple(row) in target_set for row in sample]
        # Compute the fraction of chains that match any target state.
        fraction = np.mean(matches)
        probs.append(fraction)

    return np.array(probs)

def sample_to_key(sample):
    # Convert the tensor to numpy, cast to integer (if needed), flatten, and join as string
    return ''.join(sample.cpu().numpy().astype(int).astype(str).flatten())
def hamming_distance_batch(tensors, reference):
    return (tensors != reference).sum(dim=-1)


def local_entropy(theta_a, model, eta):
    """
    Compute the local entropy for a batch of theta_a vectors.

    Args:
        theta_a: Tensor of shape (n, 4) where each row is a theta_a sample.
        model: A model that takes input of shape (batch_size, 4) and outputs scalar energy per row.
        eta: Coupling parameter (float).

    Returns:
        mean_local_entropy: Mean of the computed local entropies.
        stderr_local_entropy: Standard error of the local entropies.
    """
    binary_combinations = list(itertools.product([0, 1], repeat=model.data_dim))
    parameter_space = torch.tensor(binary_combinations, dtype=torch.float64)  # Shape: (16, 4)

    # Compute energy for the full parameter space
    energies = torch.stack([model(param.unsqueeze(0)).sum() for param in parameter_space])  # Shape: (16,)
    energies = energies.unsqueeze(0)  # Shape: (1, 16) to broadcast with batch

    # Ensure theta_a is a 2D tensor of shape (n, 4)
    if theta_a.dim() == 1:
        theta_a = theta_a.unsqueeze(0)  # Shape: (1, 4)

    # Compute squared distances from all theta_a to all 16 binary vectors
    param_diff = parameter_space.unsqueeze(0) - theta_a.unsqueeze(1)  # Shape: (n, 16, 4)
    second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)  # Shape: (n, 16)

    # Compute exp and log-sum
    exp_local_entropy = torch.exp(energies - second_part)  # Shape: (n, 16)
    internal_sum = exp_local_entropy.sum(dim=1)  # Shape: (n,)
    local_entropy_vals = torch.log(internal_sum)  # Shape: (n,)

    # Compute mean and standard error
    mean = local_entropy_vals.mean().item()
    stderr = local_entropy_vals.std(unbiased=True).item() / torch.sqrt(torch.tensor(len(local_entropy_vals), dtype=torch.float64))

    return mean, stderr


def plot_dual_heatmap(
    empirical_probs,
    target_probs,
    state_labels,
    grid_order,
    group1_states=None,
    group2_states=None,
    title="Dual Heatmap: for ",
    filename=None
):
    """
    Creates a 4x4 grid where each cell is split diagonally into two triangles:
      - The upper triangle is colored by the target probability (Reds).
      - The lower triangle is colored by the empirical probability (Blues).

    We distinguish between:
      - state_labels: the order of states in the empirical_probs / target_probs arrays.
      - grid_order:  the arrangement of these states in the 4x4 grid.

    Parameters:
    -----------
    empirical_probs : array-like of length 16
        Empirical probabilities, in the order of state_labels.
    target_probs : array-like of length 16
        Target probabilities, in the same order as state_labels.
    state_labels : list of str, length 16
        The labeling/order that matches the indices of empirical_probs / target_probs.
        E.g., ['0000','0001', ...] in the same order as your arrays.
    grid_order : list of str, length 16
        The desired 4x4 arrangement. E.g., a Gray-code or row-major ordering
        you want to use for the 4x4 display.
    group1_states : list of str, optional
        States you want to highlight in green rectangles (e.g. "sharp modes").
    group2_states : list of str, optional
        States you want to highlight in purple rectangles (e.g. "flat modes").
    title : str
        Title for the figure.
    filename : str, optional
        If provided, saves the figure to this file.
    """

    # Convert to numpy arrays if not already
    empirical_probs = np.array(empirical_probs, dtype=float)
    """
    print(empirical_probs)
    empirical_probs[4]=0.06
    empirical_probs[2] = 0.13
    empirical_probs[7] = 0.30
    print(state_labels)
    print(grid_order)
    """

    target_probs = np.array(target_probs, dtype=float)

    # Create a mapping from state_label -> index in the input arrays
    # i.e. if state_labels[i] == '0001', then label_to_idx['0001'] = i
    label_to_idx = {label: i for i, label in enumerate(state_labels)}

    # We'll create 4x4 arrays that map each cell to the appropriate probabilities
    grid_size = 4
    grid_empirical = np.zeros((grid_size, grid_size), dtype=float)
    grid_target = np.zeros((grid_size, grid_size), dtype=float)

    # Fill the 4x4 arrays by looking up the correct index from label_to_idx
    for cell_idx, label in enumerate(grid_order):
        row = cell_idx // grid_size
        col = cell_idx % grid_size

        # index in the arrays:
        arr_idx = label_to_idx[label]  # which position in ET/p arrays
        grid_empirical[row, col] = empirical_probs[arr_idx]
        grid_target[row, col] = target_probs[arr_idx]

    # Setup colormaps based on max probabilities
    norm_target = colors.Normalize(vmin=0, vmax=np.max(grid_target))
    norm_empirical = colors.Normalize(vmin=0, vmax=np.max(grid_empirical))
    cmap_target = plt.cm.Reds
    cmap_empirical = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)

    # Draw grid lines
    ax.set_xticks(np.arange(0, grid_size+1))
    ax.set_yticks(np.arange(0, grid_size+1))
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # We can place ticks at the center of cells if we want 4 labels
    x_centers = np.arange(0.5, grid_size, 1.0)
    y_centers = np.arange(0.5, grid_size, 1.0)
    ax.set_xticks(x_centers, minor=False)
    ax.set_yticks(y_centers, minor=False)
    ax.set_xticklabels([s[2:] for s in grid_order[:4]], fontsize=12)
    ax.set_yticklabels([s[:2] for s in grid_order[::4]], fontsize=12)

    # Hide the old major ticks
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

    # Now, fill each cell with two triangles
    for cell_idx, label in enumerate(grid_order):
        row = cell_idx // grid_size
        col = cell_idx % grid_size
        # corners
        bl = (col, row)
        br = (col+1, row)
        tr = (col+1, row+1)
        tl = (col, row+1)

        p_emp = grid_empirical[row, col]
        p_tar = grid_target[row, col]

        # Colors
        color_emp = cmap_empirical(norm_empirical(p_emp))
        color_tar = cmap_target(norm_target(p_tar))

        # upper triangle (target)
        upper_triangle = Polygon([tl, tr, br], closed=True, color=color_tar, ec='k', lw=0.5)
        # lower triangle (empirical)
        lower_triangle = Polygon([tl, br, bl], closed=True, color=color_emp, ec='k', lw=0.5)

        ax.add_patch(upper_triangle)
        ax.add_patch(lower_triangle)

        # text annotation
        ax.text(col+0.5, row+0.75, f"{p_tar:.2f}", ha='center', va='center', fontsize=10, color='black')
        ax.text(col+0.5, row+0.25, f"{p_emp:.2f}", ha='center', va='center', fontsize=10, color='black')

    # Highlight group1_states (green) and group2_states (purple)
    if group1_states is not None:
        for s in group1_states:
            if s in label_to_idx:
                cell_id = grid_order.index(s)  # find its position in the 4x4
                r = cell_id // grid_size
                c = cell_id % grid_size
                rect = Rectangle((c, r), 1, 1, fill=False, edgecolor='green', lw=2)
                ax.add_patch(rect)
    if group2_states is not None:
        for s in group2_states:
            if s in label_to_idx:
                cell_id = grid_order.index(s)
                r = cell_id // grid_size
                c = cell_id % grid_size
                rect = Rectangle((c, r), 1, 1, fill=False, edgecolor='purple', lw=2)
                ax.add_patch(rect)

    # Build the legend outside
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Reds(0.6), edgecolor='k', label='Target Distribution'),
        Patch(facecolor=plt.cm.Blues(0.6), edgecolor='k', label='Empirical Distribution')
    ]
    if group1_states:
        legend_elements.append(Patch(facecolor='none', edgecolor='green', lw=2, label='Sharp Mode'))
    if group2_states:
        legend_elements.append(Patch(facecolor='none', edgecolor='purple', lw=2, label='Flat Mode'))

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.,
        fontsize=10,
        frameon=False
    )

    ax.set_xlabel("Last 2 Bits", fontsize=12)
    ax.set_ylabel("First 2 Bits", fontsize=12)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def auxillary_dist(theta_a, model, eta):
    le = local_entropy(theta_a, model, eta)
    prob = np.exp(le)
    # Z=np.sum(prob)
    Z = 1

    return prob / Z


def generate_transition_heatmap(transitions, n_steps, thinning, state_labels, t):
    len = int(n_steps / thinning) + 1

    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(transitions, cmap=cmap, cbar=True, xticklabels=state_labels, annot=True,
                yticklabels=range(1, len), linewidths=.5, linecolor='black', fmt='.3f')
    plt.xlabel('States')
    plt.ylabel('Iterations')
    plt.yticks(fontsize=8)
    plt.title('State Transitions Heatmap for ' + t)
    plt.show()


def generate_sample_distribution(chain_a, t):
    theta_a_components = []
    for i in range(4):
        theta_a_components.append(np.array([tensor[0, i].item() for tensor in chain_a]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(len(theta_a_components[0]) ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(theta_a_components[i], bins=b, edgecolor='black', density=True)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Empirical Density')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle('Sample Distribution for ' + t)

    # Show the plot
    plt.show()


def generate_target_distribution(num_samples, model, eta, t):
    samples = generate_random_4d_samples(num_samples)
    stationary_probabilities = auxillary_dist(samples, model, eta)
    # print(stationary_probabilities)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(num_samples ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(np.array([tensor[i] for tensor in samples]), bins=b, edgecolor='black', density=True,
                     color='green', weights=stationary_probabilities)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Stationary Density')

    plt.tight_layout()
    fig.suptitle('Stationary Distribution for ' + t)
    plt.show()


def generate_target_distribution_glu(n, model, eta, t):
    cov_matrix = eta * np.eye(4)

    # Generate the initial x_a
    x = torch.randint(low=0, high=2, size=(4,), dtype=torch.float64).unsqueeze(0)
    x_a = x

    # Generate all binary combinations and create parameter space tensor
    binary_combinations = list(itertools.product([0, 1], repeat=4))
    parameter_space = torch.tensor(binary_combinations, dtype=torch.float64)  # Shape: (16, 4)

    samples = []
    energies = torch.stack([model(param.unsqueeze(0)).sum() for param in parameter_space])

    for _ in range(n):
        # Compute energy functions for the entire parameter space

        # Compute the second part for all parameter space combinations
        param_diff = parameter_space.unsqueeze(1) - x_a.unsqueeze(0)  # Shape: (16, 1, 4)
        second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)  # Shape: (16, 1)

        # Compute probabilities
        probs = torch.exp(energies.unsqueeze(1) - second_part).squeeze()  # Shape: (16,)
        total_sum = probs.sum().item()
        probs = probs / total_sum

        # Convert probabilities and parameters to numpy for random.choices
        probs_np = probs.numpy()
        params_np = parameter_space.numpy()

        # Sample one parameter based on computed probabilities
        chosen_param = params_np[random.choices(range(len(probs_np)), probs_np, k=1)[0]]

        # Sample from the multivariate normal distribution
        sample = np.random.multivariate_normal(chosen_param, cov_matrix, 1)
        samples.append(sample)

        # Update x_a
        x_a = torch.tensor(sample, dtype=torch.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(n ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(np.array([tensor[0, i] for tensor in samples]), bins=b, edgecolor='black', density=True,
                     color='green')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Stationary Density for ' + t)

    plt.tight_layout()
    fig.suptitle('Stationary Distribution(GLU) for ' + t)
    plt.show()


def generate_random_4d_samples(num_samples, lower_bound=-5, upper_bound=5):
    # Generate random samples within the specified bounds
    samples = np.random.uniform(lower_bound, upper_bound, size=(num_samples, 4))
    return torch.tensor(samples, dtype=torch.float64)


def plot_heatmap(prob_grid, cmap, title, filename):
    plt.figure(figsize=(10, 8))

    state_order = ['1001', '1000', '1010', '1011',
                   '0001', '0000', '0010', '0011',
                   '0101', '0100', '0110', '0111',
                   '1101', '1100', '1110', '1111']

    # Create a mapping from state to grid index
    state_map = {state: idx for idx, state in enumerate(state_order)}
    grid_size = 4

    # Define custom tick labels based on state_order
    xticklabels = [state[2:] for state in state_order[:grid_size]]  # Last 2 bits for x-axis
    yticklabels = [state[:2] for state in state_order[::grid_size]]  # First 2 bits for y-axis

    heatmap = sns.heatmap(prob_grid, annot=False, fmt=".2f", cmap=cmap,
                          cbar_kws={'label': 'Probability', 'shrink': 0.9, 'aspect': 30},
                          vmin=0, vmax=0.12,
                          xticklabels=xticklabels,  # Custom x-axis labels
                          yticklabels=yticklabels)  # Custom y-axis labels

    cbar = heatmap.collections[0].colorbar  # Get the color bar object
    cbar.set_label('Probability', fontsize=18)

    # Define states to highlight
    group1_states = ['0010', '0111']  # Sharp Mode (Green)
    group2_states = ['0100', '1001']  # Flat Mode (Purple)
    ax = plt.gca()
    for state in group1_states:
        x = state_map[state] // grid_size
        y = state_map[state] % grid_size
        rect = patches.Rectangle((y, x), 1, 1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    for state in group2_states:
        x = state_map[state] // grid_size
        y = state_map[state] % grid_size
        rect = patches.Rectangle((y, x), 1, 1, linewidth=2, edgecolor='purple', facecolor='none')
        ax.add_patch(rect)

    legend_elements = [
        Patch(facecolor='none', edgecolor='green', label='Sharp Mode'),
        Patch(facecolor='none', edgecolor='purple', label='Flat Mode')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.1), fontsize=15,
               frameon=False)
    plt.xlabel('Last 2 Bits', fontsize=19)
    plt.ylabel('First 2 Bits', fontsize=19)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def get_LMAE(GT, ET, nchain):
    absolute_errors = torch.abs(GT - ET)

    # Calculate the average absolute error
    lmae = np.log(torch.mean(absolute_errors, dim=1))
    ste = torch.std(lmae) / np.sqrt(nchain)
    lmae = torch.mean(lmae)

    return lmae.item(), ste.item()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = args.nchains
    epsilon = 0.00600

    dim = 4
    tensor_states = list(itertools.product([0, 1], repeat=4))
    state_labels = [''.join(map(str, state)) for state in tensor_states]
    state_map = {"".join(map(str, state)): idx for idx, state in enumerate(tensor_states)}
    p = np.array([
        0.07688966, 0.04725621, 0.125, 0.01667631,
        0.08688745, 0.07688966, 0.07688966, 0.16756987,
        0.04725621, 0.05825, 0.01667631, 0.04725621,
        0.07688966, 0.04725621, 0.02 - 0.000996, 0.01335262
    ])



    colors = np.array(['skyblue'] * len(p))
    colors[[2, 7]] = 'pink'
    colors[[4, 9]] = 'plum'

    plt.figure(figsize=(10, 6))
    plt.bar(state_labels, p, color=colors, edgecolor='black')
    # Customize plot
    plt.xlabel('States', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.title('Probability Mass Function (PMF)', fontsize=18)
    plt.ylim(0, max(p) + 0.005)
    plt.savefig(str('target_hist.png'), dpi=300, bbox_inches='tight')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    y = torch.randint(low=0, high=2, size=(n, dim), dtype=torch.float)
    print(y)
    # y = torch.tensor([[1.00, 0.00, 1.00, 0.00]])
    model = bernoulli_distribution.JointBernoulli(p, dim=dim)
    model.to(device)
    t = args.sampler
    j = 0
    transitions = np.zeros((int(args.n_steps - args.burnin / args.thinning), 2 ** dim))
    state_counts = {"".join(map(str, state)): 0 for state in tensor_states}
    chain = []
    chain_state = []
    chain_a = []
    MAE = []
    Prob_states = []
    TIME = []
    SE = []
    if t == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(model.data_dim)
    elif t == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
    elif "hb-" in t:
        block_size, hamming_dist = [int(v) for v in t.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)

    elif t == "dula":
        sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.1225, mh=False)
    elif t == "edula":
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.2, alpha_a=0.1, mh=False,
                                            eta=args.eta)

        # <0.1,0.4,0.1>
        #eta=0.1

    elif t == "edula-glu":
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 1,
                                                   fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                   alpha=0.2, mh=False,
                                                   eta=8)
        # eta=4
    elif t == "dmala":
        sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.1, mh=True)

    elif t == "edmala":
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.4, alpha_a=0.1, mh=True,
                                            eta=args.eta)
        # <0.2,0.01,0.5>>
        # <0.08>

    elif t == "edmala-glu":
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 1,
                                                   fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                   alpha=1, mh=True,
                                                   eta=args.eta)

    elif t == "gwg":
        sampler = samplers.DiffSampler(model.data_dim, 1,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)

    elif "gwg-" in t:
        n_hops = int(t.split('-')[1])
        sampler = samplers.MultiDiffSampler(model.data_dim, 1,
                                            approx=True, temp=2., n_samples=n_hops)
    elif "bg-" in t:
        block_size = int(t.split('-')[1])
        sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)

    else:
        assert False, 'Not implemented'
    probabilities = torch.zeros(n, 16)

    x = y
    x_a = x
    weights = torch.tensor([8, 4, 2, 1], dtype=torch.int64).to(x.device)
    for i in range(args.n_steps + 1):

        if i % args.thinning == 0:


            if (sampler.entropy == False):
                xhat = sampler.step(x.detach(), model).detach()
            else:
                sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
                xhat = sample_tuple[0].detach()
                xahat = sample_tuple[1].detach()

            x = xhat
            if (sampler.entropy == True):
                x_a = xahat

            if args.burnin < i:
                chain.append(x)
                if sampler.entropy == True:
                    chain_a.append(x_a)


    combined = torch.cat(chain, dim=0)  # Now shape is (n, 4)

    # Use torch.unique to get the unique rows and their counts
    unique_rows, counts = torch.unique(combined, dim=0, return_counts=True)
    #print(unique_rows, counts)

    # Compute the empirical probabilities by normalizing the counts
    all_possible = torch.tensor(list(itertools.product([0, 1], repeat=4)), dtype=torch.float32)

    # Prepare a list to hold counts for all 16 states (fill with zero for missing states)
    full_counts = []
    for state in all_possible:
        # Convert the state to a tuple for comparison
        state_tuple = tuple(state.numpy().astype(int))
        # Check if this state appears in unique_rows
        # We convert each row in unique_rows to tuple for comparison
        found = False
        for row, cnt in zip(unique_rows, counts):
            if tuple(row.numpy().astype(int)) == state_tuple:
                full_counts.append(cnt.item())
                found = True
                break
        if not found:
            full_counts.append(0)

    full_counts = torch.tensor(full_counts, dtype=torch.float32)

    ET = full_counts / full_counts.sum()
    plt.figure(figsize=(10, 6))
    plt.bar(state_labels, p, color='lightskyblue', edgecolor='black', alpha=0.4, label='Target')
    plt.bar(state_labels, ET, color='palegreen', edgecolor='black', alpha=0.5, label='Empirical')
    print(np.log(np.mean(np.abs(p-np.array(ET)))))



    # Customize plot with ICML/ICLR standards in mind
    plt.xlabel('States', fontsize=19)
    plt.ylabel('Probability', fontsize=19)
    if args.sampler == "edmala" or args.sampler == "edula":
        name=t.upper() +" for "+"$\eta$="+str(args.eta)
    else:
        name= t.upper()

    plt.title(name, fontsize=20)
    plt.ylim(0, max(ET) + 0.05)
    plt.legend(loc='best', fontsize=18, frameon=False)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tick_params(axis='both', which='major', labelsize=11)

    if args.sampler=="dmala" or args.sampler=="dula":
        name=f"{t}_hist_"+".png"
    else:
        name=f"{t}_hist_" + str(args.eta) + ".png"
    print(name)
    plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()

    grid_order = ['1001', '1000', '1010', '1011',
                   '0001', '0000', '0010', '0011',
                   '0101', '0100', '0110', '0111',
                   '1101', '1100', '1110', '1111']
    # Optionally, define groups to highlight:
    group1 = ['0010', '0111']
    group2 = ['0100', '1001']

    plot_dual_heatmap(ET, p, state_labels=state_labels,
                      group1_states=group1, group2_states=group2,
                      title="Dual Heatmap: "+args.sampler, filename="dual_heatmap_"+str(args.sampler)+".png", grid_order=grid_order)
    le_list = []
    le_list_se = []
    #print(chain_a)
    for a in chain_a:
        mean,stde = local_entropy(a, model, args.eta)
        #print(mean, stde)
        le_list.append(mean)
        le_list_se.append(stde)
    #print(le_list)
    np.save("sensitivity_analysis_" + str(args.eta) +str("_")+str(args.sampler)+"_.npy", le_list)
    np.save("sensitivity_analysis_se_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", le_list_se)

    if args.sampler=="edmala" or args.sampler=="edula":

        l2_norms = np.linalg.norm(np.array(chain) - np.array(chain_a), axis=1)

        # Optional: compute mean and standard error
        mean_l2 = l2_norms.mean(axis=1)
        std_l2 = l2_norms.std(ddof=1, axis=1)
        se_l2 = std_l2 / np.sqrt(len(l2_norms[0]))

        np.save("l2norm_" + str(args.eta) +str("_")+str(args.sampler)+"_.npy", mean_l2)
        np.save("l2norm_se_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", se_l2)

    target_states = [
        [0, 1, 0, 0],
        [1, 0, 0, 1]
    ]

    flat_mode_probability=state_probability(chain,target_states)
    np.save("flat_mode_probability_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", flat_mode_probability)

    eigenvalues_list = []

    # Loop over each theta_sample in collect:
    for theta_sample in chain:
        # Compute Hessians for each chain in this batch.
        hessians = compute_hessian_batch(model, theta_sample)  # returns a list of Hessians if 2D
        # Compute eigenvalues for each Hessian and add to eigenvalues_list
        for H in hessians:
            eigvals = np.linalg.eigvalsh(H)
            eigenvalues_list.append(eigvals)

    # Concatenate all eigenvalues into one array
    eigenvalues_array = np.concatenate(eigenvalues_list)

    np.save("eigenspectrum_"+ str(args.sampler) + "_.npy", eigenvalues_array)

    from scipy.stats import skew



    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues_array, bins=30,color='lightblue', edgecolor='black')
    Q1 = np.percentile(eigenvalues_array, 25)
    Q3 = np.percentile(eigenvalues_array, 75)
    IQR = Q3 - Q1

    print(f"IQR: {IQR}")
    print(np.mean(eigenvalues_array),np.std(eigenvalues_array))
    plt.xlabel('Eigenvalue Magnitude', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title('Hessian Eigenspectrum for 4D Joint Bernoulli for ' + str(args.sampler), fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()


    plt.show()

    print()


    if args.sampler == 'edmala' or args.sampler == 'edmala-glu' or args.sampler == 'dmala':
        le_list=[np.round(np.mean(sampler.a_s), decimals=3),np.round(np.std(sampler.a_s), decimals=3) ]
        np.save("acceptance_ratio_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", le_list)

        print("Average Acceptance Probability: " + str(np.round(np.mean(sampler.a_s), decimals=3)))
        print("Standard Deviation: " + str(np.round(np.std(sampler.a_s), decimals=3)))


# generate_transition_heatmap(transitions, args.n_steps - args.burnin, args.thinning, state_labels, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--burnin', type=int, default=200)
    parser.add_argument('--n_steps', type=int, default=1000)  # 10,000
    parser.add_argument('--thinning', type=int, default=1)  # 1
    parser.add_argument('--viz', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--nchains', type=int, default=4)
    parser.add_argument('--sampler', type=str, default='dula')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--alpha_a', type=float, default=0.001)
    parser.add_argument('--eta', type=float, default=0.1)
    # temp = ['rand-gibbs','bg-1', 'gwg-1', 'dula', 'dmala', 'edula', 'edmala', 'edula-glu', 'edmala-glu']

    parser.add_argument('--save_dir', type=str, default="./bernoulli_sample_data")
    args = parser.parse_args()
    print((args.seed))
    main(args)
