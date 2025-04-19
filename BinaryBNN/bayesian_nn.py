import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import random
import torch
torch.cuda.empty_cache()
import time
from torch.optim import Adam, Adagrad, SGD
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import adult_loader as ad
import compas_loader as cp
import blog_loader as bg
import news_loader as ns
import matplotlib.pyplot as plt
import itertools
import argparse
from GWG_release import samplers


def local_entropy(theta_a, model, eta, chunk_size=256):
    binary_combinations = list(itertools.product([0, 1], repeat=model.hidden_dim))
    parameter_space = torch.tensor(binary_combinations, dtype=torch.float32)  # (16, 4)

    energies = torch.stack([model(param.unsqueeze(0)).sum() for param in parameter_space])  # (16,)
    energies = energies.unsqueeze(0)  # (1, 16)

    if theta_a.dim() == 1:
        theta_a = theta_a.unsqueeze(0)
    theta_a = theta_a.to(torch.float32)

    local_entropy_vals = []

    for i in range(0, theta_a.shape[0], chunk_size):
        chunk = theta_a[i:i+chunk_size]  # (chunk_size, 4)
        param_diff = parameter_space.unsqueeze(0) - chunk.unsqueeze(1)  # (chunk_size, 16, 4)
        second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)  # (chunk_size, 16)
        exp_local_entropy = torch.exp(energies - second_part)  # (chunk_size, 16)
        internal_sum = exp_local_entropy.sum(dim=1)  # (chunk_size,)
        local_entropy_vals.append(torch.log(internal_sum))

    local_entropy_vals = torch.cat(local_entropy_vals)

    mean = local_entropy_vals.mean().item()
    stderr = local_entropy_vals.std(unbiased=True).item() / torch.sqrt(torch.tensor(len(local_entropy_vals), dtype=torch.float32))

    return mean, stderr

def compute_hessian_for_sample(model, theta_batch):
    """
    Computes the Hessian of U(theta) w.r.t. theta for a batch of samples.

    Parameters:
      theta_batch: torch tensor of shape (B, d), where B is batch size.

    Returns:
      List of B Hessians, each of shape (d, d).
    """
    theta_batch = theta_batch.clone().detach().requires_grad_(True)
    U_theta = model(theta_batch)  # shape: (B,) or (B, 1)

    if U_theta.ndim > 1:
        U_theta = U_theta.squeeze(-1)

    hessians = []
    for i in range(theta_batch.shape[0]):
        grad = torch.autograd.grad(U_theta[i], theta_batch, create_graph=True)[0][i]  # shape: (d,)
        H = []
        for j in range(grad.shape[0]):
            grad2 = torch.autograd.grad(grad[j], theta_batch, retain_graph=True)[0][i]
            H.append(grad2)
        H = torch.stack(H)
        hessians.append(H.cpu().numpy())

    return hessians


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


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

EPOCH = 1000+1
TEMP = 100.

parser = argparse.ArgumentParser()
parser.add_argument('--burin', type=int, default=1000)
parser.add_argument('--sampler', type=str, default='edmala')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--alpha_a', type=float, default=0.00100)
parser.add_argument('--eta', type=float, default=100)
parser.add_argument('--dataset', type=str, default='compas')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=-1)
args = parser.parse_args()

setup_seed(args.seed)

log_dir = 'logs/%s/%s_%d_%d'%(args.dataset, args.sampler, args.batchsize, args.seed)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(args.sampler)


class BayesianNN(nn.Module):
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        super(BayesianNN, self).__init__()
        #self.lambda_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.1, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1] 
        self.hidden_dim = hidden_dim

    def forward_data(self, inputs, theta):
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -1].reshape(-1, 1, 1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.tanh(torch.bmm(inputs, w1) + b1)
        #print(inter.shape, w2.shape, b2.shape, self.hidden_dim, (self.n_features + 1) * self.hidden_dim)
        out_logit = torch.bmm(inter, w2) + b2
        out = out_logit.squeeze()
        out = torch.sigmoid(out)
        
        return out



    def forward(self, theta):
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        # w_prior should be decided based on current lambda (not sure)
        w_prior = Normal(0., 1.)

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs = self.forward_data(X_batch[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)*TEMP

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data  # (8) in paper
        log_p = log_p_data
        
        return log_p



def train_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_test.unsqueeze(0).repeat(model.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data / X_test.shape[0]  # (8) in paper
        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), rmse.mean().cpu().numpy()

def test_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        w_prior = Normal(0., 1.)

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        log_p_data = (outputs.mean(dim=0) - y_test).pow(2) 
        log_p_data = (-1.)*log_p_data.mean()

        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), np.sqrt(rmse.mean().cpu().numpy())



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.dataset == 'adult':
        X_train, y_train, X_test, y_test = ad.load_data(get_categorical_info=False)
    elif args.dataset == 'compas':
        X_train, y_train, X_test, y_test = cp.load_data(get_categorical_info=False)
    elif args.dataset == 'blog':
        X_train, y_train, X_test, y_test = bg.load_data(get_categorical_info=False)
    elif args.dataset == 'news':
        X_train, y_train, X_test, y_test = ns.load_data(get_categorical_info=False)
    else:
        print('Not Available')
        assert False

    n = X_train.shape[0]
    n = int(0.99*n)
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(np.max(X_train), np.min(X_train), np.mean(y_train), np.mean(y_test))

    feature_num = X_train.shape[1]
    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)

    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    """
    X_val = torch.tensor(X_val).float().to(device)
    y_val = torch.tensor(y_val).float().to(device)
    X_test=X_val
    y_test=y_val
    """

    X_train_mean, X_train_std = torch.mean(X_train[:, :], dim=0), torch.std(X_train[:, :], dim=0)
    X_train[:, :] = (X_train [:, :]- X_train_mean) / X_train_std
    X_test[:, :] = (X_test[:, :] - X_train_mean) / X_train_std
    
    if args.batchsize == -1:
        num_particles, batch_size, hidden_dim = 50, X_train.shape[0], 100 # 500 for others, 100 for blog
    else:
        num_particles, batch_size, hidden_dim = 50, args.batchsize, 100

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)
    model = model.to(device)

    # Random initialization (based on expectation of gamma distribution)
    theta = torch.cat([torch.zeros([num_particles, (X_train.shape[1] +2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(0.01))]) 
    theta = torch.bernoulli(torch.ones_like(theta)*0.5).to(device)
    theta_a=theta.to(device)
    print(theta.shape)
    dim = theta.shape[1]
    
    if args.sampler == 'gibbs':
        sampler = samplers.PerDimGibbsSampler(dim, rand=True)
    elif args.sampler == 'gwg':
        sampler = samplers.DiffSampler(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif args.sampler == 'dula':
        sampler = samplers.LangevinSampler(dim, 1,fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.alpha, mh=False)
    elif args.sampler == 'dmala':
        sampler = samplers.LangevinSampler(dim, 1,fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.alpha, mh=True)
    elif args.sampler == 'edula':
        sampler = samplers.ELangevinSampler(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           alpha=args.alpha,alpha_a=args.alpha_a , eta=args.eta ,mh=False)
        #blog={0.000001,5}
        #compas={0.1,5}
        #news={0.00001, 4}
        #adult={}
    elif args.sampler == 'edmala':
        sampler = samplers.ELangevinSampler(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           alpha=args.alpha,alpha_a=args.alpha_a, eta=args.eta ,mh=True)
        #compas={0.05,2000}
        #adult={0.001, 2000}
    elif args.sampler == 'edula-glu':
        sampler = samplers.ELangevinSamplerwithGLU(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           alpha=args.alpha, eta=args.eta,mh=False)
    elif args.sampler == 'edmala-glu':
        sampler = samplers.ELangevinSamplerwithGLU(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           alpha=args.alpha, eta=args.eta ,mh=True)

    else:
        print('Not Available')
        assert False
    
    training_ll_cllt = []
    test_ll_cllt = []
    tracking_time=[]
    chain=[]
    chain_a=[]


    for epoch in range(EPOCH):
        st = time.perf_counter()
        if (sampler.entropy == False):
            theta_hat = sampler.step(theta.detach(), model).detach()
            theta = theta_hat.data.detach().clone()

        else:
            sample_tuple = sampler.step(theta.detach(), theta_a.detach(), model)
            theta = sample_tuple[0].data.detach().clone()
            theta_a = sample_tuple[1].data.detach().clone()

        running_time = time.perf_counter() - st
        tracking_time.append(running_time)
        chain.append(theta)
        chain_a.append(theta_a)


        if epoch % 5 == 0:
            training_ll, training_rmse = train_log(model, theta, X_train, y_train)
            training_ll_cllt.append(training_ll)
            
            test_ll, test_rmse = test_log(model, theta, X_test, y_test)
            test_ll_cllt.append(test_rmse)

            if epoch % 100 == 0:
                print(epoch, 'Training LL:', training_ll, 'Test LL:', test_ll)
                print(epoch, 'Training RMSE:', training_rmse, 'Test RMSE:', test_rmse)


    """
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
    
    """



    if args.sampler=="edmala" or args.sampler=="edula":

        l2_norms = np.linalg.norm(np.array(chain) - np.array(chain_a), axis=1)

        # Optional: compute mean and standard error
        mean_l2 = l2_norms.mean(axis=1)
        std_l2 = l2_norms.std(ddof=1, axis=1)
        se_l2 = std_l2 / np.sqrt(len(l2_norms[0]))

        np.save("l2norm_" + str(args.eta) +str("_")+str(args.sampler)+"_.npy", mean_l2)
        np.save("l2norm_se_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", se_l2)

    x=np.array(training_ll_cllt)
    y=np.array(test_ll_cllt)
    np.save('%s/training_ll.npy'%(log_dir), x)
    np.save('%s/test_rmse.npy'%(log_dir), y)
    np.save('%s/time.npy'%(log_dir), tracking_time)
    print('Training LL Mean +- std '+str( np.mean(x))+" "+str(np.std(x)))
    print('Test RMSE Mean +- std ' + str(np.mean(y)) + " " + str(np.std(y)))

    print(sampler.a_s)



    if args.sampler == 'edmala' or args.sampler == 'edmala-glu' or args.sampler == 'dmala':
        le_list=[np.round(np.mean(sampler.a_s), decimals=3),np.round(np.std(sampler.a_s), decimals=3) ]
        np.save("acceptance_ratio_" + str(args.eta) + str("_") + str(args.sampler) + "_.npy", le_list)

        print("Average Acceptance Probability: " + str(np.round(np.mean(sampler.a_s), decimals=3)))
        print("Standard Deviation: " + str(np.round(np.std(sampler.a_s), decimals=3)))

    

if __name__ == '__main__':
    main()
