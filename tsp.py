import torch
import itertools



device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.distributions as dists
from tqdm import tqdm
import igraph as ig
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import argparse


def euclidean_distance(city1, city2):
    return torch.norm(city1 - city2)

class TSP8(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.city_positions = {
    'A': torch.tensor([1.00, 0.00, 0.00], requires_grad=True),
    'B': torch.tensor([0.00, 1.00, 0.00], requires_grad=True),
    'C': torch.tensor([0.00, 0.00, 1.00], requires_grad=True),
    'D': torch.tensor([1.00, 1.00, 0.00], requires_grad=True),
    'E': torch.tensor([0.00, 1.00, 1.00], requires_grad=True),
    'F': torch.tensor([1.00, 0.00, 1.00], requires_grad=True),
    'G': torch.tensor([1.00, 1.00, 1.00], requires_grad=True),
    'H': torch.tensor([0.00, 0.00, 0.00], requires_grad=True)
}


        self.reverse_city_positions = {tuple(position.tolist()): city for city, position in self.city_positions.items()}

        self.weights = {
            ('A', 'B'): 2.0, ('B', 'A'): 1.0,
            ('A', 'C'): 1.5, ('C', 'A'): 1.8,
            ('A', 'D'): 1.2, ('D', 'A'): 1.7,
            ('B', 'C'): 2.5, ('C', 'B'): 2.2,
            ('B', 'D'): 2.0, ('D', 'B'): 1.3,
            ('C', 'D'): 1.6, ('D', 'C'): 1.9,
            # Extend the weights for the additional cities
            ('A', 'E'): 2.1, ('E', 'A'): 1.9,
            ('A', 'F'): 2.3, ('F', 'A'): 1.5,
            ('A', 'G'): 2.2, ('G', 'A'): 1.6,
            ('A', 'H'): 2.4, ('H', 'A'): 1.8,
            ('B', 'E'): 2.6, ('E', 'B'): 2.1,
            ('B', 'F'): 2.5, ('F', 'B'): 1.9,
            ('B', 'G'): 2.8, ('G', 'B'): 1.7,
            ('B', 'H'): 2.9, ('H', 'B'): 1.5,
            ('C', 'E'): 1.4, ('E', 'C'): 1.8,
            ('C', 'F'): 1.6, ('F', 'C'): 1.7,
            ('C', 'G'): 1.5, ('G', 'C'): 1.6,
            ('C', 'H'): 1.7, ('H', 'C'): 1.9,
            ('D', 'E'): 2.4, ('E', 'D'): 1.5,
            ('D', 'F'): 2.3, ('F', 'D'): 1.7,
            ('D', 'G'): 2.2, ('G', 'D'): 1.8,
            ('D', 'H'): 2.6, ('H', 'D'): 2.0,
            ('E', 'F'): 2.5, ('F', 'E'): 1.4,
            ('E', 'G'): 2.8, ('G', 'E'): 1.5,
            ('E', 'H'): 2.6, ('H', 'E'): 1.6,
            ('F', 'G'): 2.2, ('G', 'F'): 1.4,
            ('F', 'H'): 2.3, ('H', 'F'): 1.5,
            ('G', 'H'): 2.5, ('H', 'G'): 1.7
        }



        self.data_dim = dim

    def forward(self,permutation):
        #print(permutation)
        total_cost = torch.tensor(0.0, requires_grad=True)

        for i in range(len(permutation)-1 ):
            city1, city2 = permutation[i], permutation[i + 1]
            total_cost = total_cost + self.weights[(city1, city2)] * euclidean_distance(self.city_positions[city1],
                                                                                 self.city_positions[city2])

        # Add the cost of returning to the starting city

        city_last, city_first = permutation[-1], permutation[0]
        total_cost = total_cost + self.weights[(city_last, city_first)] * euclidean_distance(self.city_positions[city_last],
                                                                                      self.city_positions[city_first])


        return -total_cost

        # Compute each term independently for the entire batch

    def compute_gradient(self, permutation):
        # Reset gradients from previous calculations
        for pos in self.city_positions.values():
            if pos.grad is not None:
                pos.grad.zero_()

        # Calculate the cost for the current permutation
        cost = self.forward(permutation)

        # Backpropagate to compute the gradients
        cost.backward()

        # Extract the gradients for each city
        gradients = [self.city_positions[city].grad.clone().detach() for city in permutation]
        gradients = torch.stack(gradients)
        return gradients


# Generate all permutations of the cities
