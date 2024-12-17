import argparse
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools


def neighbourhood_prob_generation(model,x):
    n_tensors, dim = x.shape

    # Create a mask that will flip each bit one by one
    flip_mask = torch.eye(dim).long().unsqueeze(0).repeat(n_tensors, 1, 1)

    # Expand the original tensor to align with the flip mask (5, 25, 25)

    # Flip one bit at a time for each tensor using XOR with the mask
    x=x.to(torch.long)
    x_expanded = x.unsqueeze(1).repeat(1, dim, 1)
    flipped_tensors = x_expanded ^ flip_mask

    # Reshape the flipped tensor into a (5 * 25, 25) tensor
    flipped_tensors = flipped_tensors.view(-1, dim)

    # The result is a tensor of shape (5 * 25, 25)
    output=model(flipped_tensors.to(torch.float))

    batches = output.view(5, 25)

    # Compute the mean and standard error for each batch
    means = batches.mean(dim=1)
    std_errors = batches.std(dim=1) / torch.sqrt(torch.tensor(25.0))  # Standard error = std / sqrt(n)

    # Print the results
    """
    for i in range(5):
        print(f"Batch {i + 1}: Mean = {means[i]}, Std Error = {std_errors[i]}")
    """

    return torch.mean(means).item(), torch.mean(std_errors).item()

def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def Local_Entropy(theta_a, model, eta):
    dim = args.dim ** 2
    A = model.J
    b = model.bias

    lst = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1) + torch.sum(b * x, dim=-1))
    energies = f(lst)
    energies = energies / torch.sum(energies)


    # Convert theta_a to tensor if not already
    theta_a_tensor = torch.stack(theta_a)  # Shape: (num_samples, 1, 4)

    # Compute the differences and the second part
    param_diff = lst.unsqueeze(0) - theta_a_tensor  # Shape: (num_samples, 16, 4)
    second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)

    exp_local_entropy = torch.exp(energies.unsqueeze(0) - second_part)
    internal_sum=exp_local_entropy.sum(dim=1)# Shape: (num_samples, 16)
    local_entropy = torch.log(internal_sum)
    # Shape: (num_samples,)

    return local_entropy.numpy()
def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def get_log_rmse(x, gt_mean):
    x = 2. * x - 1.
    residuals = x - gt_mean
    n = x.shape[0]
    c=torch.log(torch.sqrt((residuals ** 2).mean(dim=1)))
    log_rmse=torch.mean(c)
    se=torch.std(c)/np.sqrt(n)



    # Return the logRMSE and its standard error
    return log_rmse.cpu().detach().numpy(), se.cpu().detach().numpy()


def tv(samples):
    gt_probs = np.load("{}/gt_prob_{}_{}.npy".format(args.save_dir, args.dim, args.bias))
    arrs, uniq_cnt = np.unique(samples, axis=0, return_counts=True)
    sample_probs = np.zeros_like(gt_probs)

    for i in range(arrs.shape[0]):
        sample_probs[i] = (uniq_cnt[i] * (1.) - 1.) / samples.shape[0]
    l_dist = np.abs((gt_probs - sample_probs)).sum()


def get_gt_mean(args, model):
    dim = args.dim ** 2
    A = model.J
    b = model.bias
    lst = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1) + torch.sum(b * x, dim=-1))
    flst = f(lst)
    plst = flst / torch.sum(flst)

    """

    plt.hist(range(0,2**dim), bins=2**dim, density=True, edgecolor='black', color='pink', weights=plst)
    plt.show()
    """





    gt_mean = torch.sum(lst * plst.unsqueeze(1).expand(-1, lst.size(1)), 0)
    torch.save(gt_mean.cpu(),
               "{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir, args.dim, args.sigma, args.bias))

    # gt_mean = torch.load("{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias)).to(device)
    return gt_mean


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    gt_mean = get_gt_mean(args, model)

    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                     p, normalize=False, nrow=int(x.size(0) ** .5))
    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}
    means = {}

    HD=[]
    HD_e=[]
    M=[]
    M_e=[]
    E=[]
    E_e=[]

    P=[]
    P_e=[]
    SE=[]

    rmses = {}
    times_list={}
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    # temps = ['bg-1', 'hb-10-1','gwg-1','dula', 'dmala','edula','edmala','edula-glu','edmala-glu']
    temp = args.sampler
    print(temp)

    if temp == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(model.data_dim,n=1)
    elif temp == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(model.data_dim,n=1, rand=True)
    elif temp == "lb":
        sampler = samplers.PerDimLB(model.data_dim)
    elif "bg-" in temp:
        block_size = int(temp.split('-')[1])
        sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
    elif "hb-" in temp:
        block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
    elif temp == "gwg":
        sampler = samplers.DiffSampler(model.data_dim, 1,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif "gwg-" in temp:
        n_hops = int(temp.split('-')[1])
        sampler = samplers.MultiDiffSampler(model.data_dim, args.refine,
                                            approx=True, temp=2., n_samples=n_hops)

    elif temp == "dula":
        sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1,
                                           mh=False)


        #2

    elif temp == "dmala":
        sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.8,
                                           mh=True)


    elif temp == 'edmala':
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.8, alpha_a=0.0000001, mh=True, eta=0.1)
        #<0.8, 0.0000001, 0.1>
    elif temp == 'edmala-glu':
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 2,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.8, mh=True, eta=0.1)
    elif temp == "edula":
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.1,
                                            alpha_a=0.0001, mh=False, eta=0.05)


    elif temp == 'edula-glu':
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 2,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.1, mh=False, eta=0.05)

    elif temp=='fbp':
        arrays = [np.load(f"FBP/ising_fbp_{i}.npy") for i in range(5)]


        # Initialize an empty list to store the (5, 25) tensors
        tensor_list = []

        # Assuming each array has 5000 samples, we loop through each sample index
        for j in range(5000):
            # Take the j-th sample from each array and stack them to form a (5, 25) tensor
            tensor = torch.stack([torch.tensor(arrays[i][j],  dtype=torch.float32) for i in range(5)], dim=0)
            tensor_list.append(tensor)

        item_prev=tensor_list[0]
        for item in tensor_list:

            p, p_e = neighbourhood_prob_generation(model, (item+1)/2)
            m = torch.mean(torch.sum( item , dim=1) / 25).item()
            m_e = torch.std(torch.sum( item , dim=1) / 25).item() / np.sqrt(25)

            e = torch.mean((model((item+1)/2))).item()
            e_e = torch.std((model((item+1)/2))).item() / np.sqrt(25)

            h = (item != item_prev).float().sum(-1).mean().item()
            h_e = torch.std((item != item_prev).float().sum(-1)).item()/5

            print("Average Magnetization: " + str(m) + " " + str(m_e))
            print("Average Probability: " + str(e) + " " + str(e_e))
            print("Average Hamming Distance: " + str(h) + " " + str(h_e))
            # print("Average Acceptance prob: "+str(np.mean(sampler.a_s)))
            print("\n")

            M.append(m)
            M_e.append(m_e)

            E.append(e)
            E_e.append(e_e)

            HD.append(h)
            HD_e.append(h_e)

            P.append(p)
            P_e.append(p_e)
            item_prev=item

        np.save("{}/ising_sample_HD_{}.npy".format(args.save_dir, temp), HD)
        np.save("{}/ising_sample_HD_Error_{}.npy".format(args.save_dir, temp), HD_e)
        np.save("{}/ising_sample_Diff_{}.npy".format(args.save_dir, temp), np.abs(np.array(P) - np.array(E)))
        np.save("{}/ising_sample_Diff_Error_{}.npy".format(args.save_dir, temp),
                np.sqrt(np.square(P_e) + np.square(E_e)))
        np.save("{}/ising_sample_Magnetization_{}.npy".format(args.save_dir, temp), M)
        np.save("{}/ising_sample_Magnetization_Error_{}.npy".format(args.save_dir, temp), M_e)

        E = np.array(E)
        E_e = np.array(E_e)

        N = range(0, len(E))
        plt.plot(N, E, marker='', label='Energy', alpha=0.5)
        plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)

        E = np.array(P)
        E_e = np.array(P_e)

        plt.plot(N, E, marker='', label='Surrounding Energy', alpha=0.5)
        plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)

        plt.xlabel('Iterations' ,fontsize=16)
        plt.ylabel('Average Energy',  fontsize=16)
        plt.title('Energy Changes in Ising Model for ' + str(args.sampler),  fontsize=18)
        plt.legend(fontsize=14)
        plt.show()
        plt.clf()

        E = np.array(M)
        E_e = np.array(M_e)

        plt.plot(N, E, marker='')
        plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)
        plt.xlabel('Iterations')
        plt.ylabel('Average Magnetization')
        plt.title('Magnetization Changes in Ising Model for ' + str(args.sampler))
        # plt.axhline(y=1, color='black', linestyle='-')
        # plt.axhline(y=-1, color='black', linestyle='-')
        plt.legend()
        plt.show()
        plt.clf()

        E = np.array(HD)
        E_e = np.array(HD_e)

        plt.plot(N, E, marker='')
        plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)
        plt.xlabel('Iterations')
        plt.ylabel('Average Hamming Distance')
        plt.title('Consecutive Hamming Distance Changes in Ising Model for ' + str(args.sampler))
        # plt.axhline(y=25, color='black', linestyle='-')
        # plt.axhline(y=0, color='black', linestyle='-')
        plt.legend()
        plt.show()
        plt.clf()
        exit()



    else:
        raise ValueError("Invalid sampler...")







    x = x0.clone().detach()
    x_a = x
    times[temp] = []
    hops[temp] = []
    chain = []
    chain_a=[]
    chain_a_1=[]
    cur_time = 0.
    mean = torch.zeros_like(x)
    times_list[temp] = []
    st=time.time()
    rmses[temp] = []
    for i in range(args.n_steps):
        # do sampling and time it

        if (sampler.entropy == False):
            xhat=sampler.step(x.detach(),model).detach()
        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            xahat = sample_tuple[1].detach()

        cur_time += time.time() - st

        # compute hamming dist
        cur_hops = (x != xhat).float().sum(-1).mean().item()
        cur_hops_std=torch.std((x != xhat).float().sum(-1)).item()/5

        # update trajectory
        x = xhat
        if (sampler.entropy == True):
            x_a = xahat
            # print(x_a)

        mean = mean + x
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(x.cpu().numpy()[0][None])
                if sampler.entropy==True:
                    chain_a.append(x_a)
            else:
                xc = x
                h = (xc != ess_samples[0][None]).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])
                if sampler.entropy == True:
                    chain_a.append(x_a)

        if i>args.burn_in:
            if i % args.show_sample==0:
                p,p_e=neighbourhood_prob_generation(model,x)
                m=torch.mean(torch.sum(2*x-1, dim=1)/25).item()
                m_e=torch.std(torch.sum(2*x-1, dim=1)/25).item()/np.sqrt(25)

                e=torch.mean((model(x))).item()
                e_e=torch.std((model(x))).item()/np.sqrt(25)

                h=cur_hops
                h_e=cur_hops_std

                print("Average Magnetization: "+str(m)+" "+str(m_e))
                print("Average Probability: "+str(e)+" "+str(e_e))
                print("Average Hamming Distance: " + str(h) + " " + str(h_e))
                #print("Average Acceptance prob: "+str(np.mean(sampler.a_s)))
                print("\n")

                M.append(m)
                M_e.append(m_e)

                E.append(e)
                E_e.append(e_e)

                HD.append(h)
                HD_e.append(h_e)

                P.append(p)
                P_e.append(p_e)


    means[temp] = mean / args.n_steps
    chain = np.concatenate(chain, 0)
    chains[temp] = chain
    if not args.no_ess:
        ess[temp] = get_ess(chain, args.burn_in)

    np.save("{}/ising_sample_HD_{}.npy".format(args.save_dir, temp), HD)
    np.save("{}/ising_sample_HD_Error_{}.npy".format(args.save_dir, temp), HD_e)
    np.save("{}/ising_sample_Diff_{}.npy".format(args.save_dir, temp), np.abs(np.array(P)-np.array(E)))
    np.save("{}/ising_sample_Diff_Error_{}.npy".format(args.save_dir, temp), np.sqrt(np.square(P_e) + np.square(E_e)))
    np.save("{}/ising_sample_Magnetization_{}.npy".format(args.save_dir, temp), M)
    np.save("{}/ising_sample_Magnetization_Error_{}.npy".format(args.save_dir, temp), M_e)

    def list_to_dict_with_duplicates(lst):
        from collections import defaultdict
        tensor_dict = defaultdict(list)
        for index, tensor in enumerate(lst):
            tensor_key = tuple(tensor.tolist())  # Convert tensor to a tuple
            tensor_dict[tensor_key].append(index)
        return tensor_dict

    #print(rmses[temp])
    #print(len(list_to_dict_with_duplicates(chain)))

    E=np.array(E)
    E_e=np.array(E_e)

    N = range(0, len(E))
    plt.plot(N, E, marker='',label='Energy',alpha=0.5)
    plt.fill_between(N,  E- E_e, E + E_e, alpha=0.3)

    E = np.array(P)
    E_e = np.array(P_e)


    plt.plot(N, E, marker='',label='Surrounding Energy',alpha=0.5)
    plt.fill_between(N,  E- E_e, E + E_e, alpha=0.3)

    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Average Energy', fontsize=14)
    plt.title('Energy Changes in Ising Model for '+str(args.sampler), fontsize=17)
    plt.legend(fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f"{args.sampler}_energy.png", dpi=300, bbox_inches='tight')

    plt.show()
    plt.clf()

    E = np.array(M)
    E_e = np.array(M_e)

    plt.plot(N, E, marker='')
    plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel('Average Magnetization')
    plt.title('Magnetization Changes in Ising Model for '+str(args.sampler))
    #plt.axhline(y=1, color='black', linestyle='-')
    #plt.axhline(y=-1, color='black', linestyle='-')
    plt.legend()
    plt.show()
    plt.clf()

    E = np.array(HD)
    E_e = np.array(HD_e)

    plt.plot(N, E, marker='')
    plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel('Average Hamming Distance')
    plt.title('Consecutive Hamming Distance Changes in Ising Model for '+str(args.sampler))
    #plt.axhline(y=25, color='black', linestyle='-')
    #plt.axhline(y=0, color='black', linestyle='-')
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/ising_sample")
    parser.add_argument('--n_steps', type=int, default=7000)
    parser.add_argument('--sampler', type=str, default='hb-10-1')
    parser.add_argument('--alpha_a', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--eta', type=float, default=0.1)
    # parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--show_sample', type=int, default=1)
    parser.add_argument('--burn-in', type=int, default=2000)



    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_test_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234567)
    # model def
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--refine', type=int, default=1)
    parser.add_argument('--sweeps', type=int, default=1)
    parser.add_argument('--denoise', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--bias', type=float, default=0.2)

    # logging
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)

    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    parser.add_argument('--no_ess', action="store_true")
    args = parser.parse_args()

    main(args)
