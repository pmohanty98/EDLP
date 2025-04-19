
import torch


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


import torch
import torch.nn as nn
import torch.distributions as dists
from tqdm import tqdm
import igraph as ig
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import argparse


class Bernoulli(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.prob_success = torch.tensor(p)
        self.init_dist = dists.Bernoulli(self.prob_success)
        self.data_dim = dim

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    def forward(self, x):
        return x * torch.logit(self.prob_success) + torch.log(1 - self.prob_success)

    def gradient(self):
        return torch.logit(self.prob_success)


class JointBernoulli(nn.Module):
    def __init__(self, p,dim):
        super().__init__()
        self.p0000 = torch.tensor(p[0])
        self.p0001 = torch.tensor(p[1])
        self.p0010 = torch.tensor(p[2])
        self.p0011 = torch.tensor(p[3])

        self.p0100 = torch.tensor(p[4])
        self.p0101 = torch.tensor(p[5])
        self.p0110 = torch.tensor(p[6])
        self.p0111 = torch.tensor(p[7])

        self.p1000 = torch.tensor(p[8])
        self.p1001 = torch.tensor(p[9])
        self.p1010 = torch.tensor(p[10])
        self.p1011 = torch.tensor(p[11])

        self.p1100 = torch.tensor(p[12])
        self.p1101 = torch.tensor(p[13])
        self.p1110 = torch.tensor(p[14])
        self.p1111 = torch.tensor(p[15])


        self.data_dim = dim


    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # Compute each term independently for the entire batch
        result = (
                (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * torch.log(self.p0000) +
                (1 - x1) * (1 - x2) * (1 - x3) * (x4) * torch.log(self.p0001) +
                (1 - x1) * (1 - x2) * (x3) * (1 - x4) * torch.log(self.p0010) +
                (1 - x1) * (1 - x2) * (x3) * (x4) * torch.log(self.p0011) +
                (1 - x1) * (x2) * (1 - x3) * (1 - x4) * torch.log(self.p0100) +
                (1 - x1) * (x2) * (1 - x3) * (x4) * torch.log(self.p0101) +
                (1 - x1) * (x2) * (x3) * (1 - x4) * torch.log(self.p0110) +
                (1 - x1) * (x2) * (x3) * (x4) * torch.log(self.p0111) +
                (x1) * (1 - x2) * (1 - x3) * (1 - x4) * torch.log(self.p1000) +
                (x1) * (1 - x2) * (1 - x3) * (x4) * torch.log(self.p1001) +
                (x1) * (1 - x2) * (x3) * (1 - x4) * torch.log(self.p1010) +
                (x1) * (1 - x2) * (x3) * (x4) * torch.log(self.p1011) +
                (x1) * (x2) * (1 - x3) * (1 - x4) * torch.log(self.p1100) +
                (x1) * (x2) * (1 - x3) * (x4) * torch.log(self.p1101) +
                (x1) * (x2) * (x3) * (1 - x4) * torch.log(self.p1110) +
                (x1) * (x2) * (x3) * (x4) * torch.log(self.p1111)
        )

        return result

