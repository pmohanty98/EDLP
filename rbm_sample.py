import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools

import math

def neighbourhood_prob_generation(model, x, k):
    n_tensors, dim = x.shape

    # Calculate the number of k-bit flip combinations
    num_combinations = math.comb(dim, k)  # dim choose k

    # Generate all combinations of k bits to flip
    bit_combinations = list(itertools.combinations(range(dim), k))  # Generate (dim choose k) combinations

    # Create a flip mask that will flip k bits for each combination of positions
    flip_mask = torch.zeros((n_tensors, num_combinations, dim), dtype=torch.long)

    for idx, combination in enumerate(bit_combinations):
        for bit_pos in combination:
            flip_mask[:, idx, bit_pos] = 1  # Set the bit to flip in the mask

    # Expand the original tensor to align with the flip mask
    x = x.to(torch.long)
    x_expanded = x.unsqueeze(1).repeat(1, num_combinations, 1)  # Expand for each combination

    # Flip k bits at a time using XOR with the mask
    flipped_tensors = x_expanded ^ flip_mask

    # Reshape the flipped tensors into a 2D tensor of shape (n_tensors * num_combinations, dim)
    flipped_tensors = flipped_tensors.view(-1, dim)

    # Apply the model to the flipped tensors and obtain output
    output = model(flipped_tensors.to(torch.float))

    # Reshape the output back to (n_tensors, num_combinations)
    batches = output.view(n_tensors, num_combinations)

    # Compute the mean and standard error for each batch
    means = batches.mean(dim=1)
    std_errors = batches.std(dim=1) / torch.sqrt(torch.tensor(float(num_combinations)))  # Standard error

    # Return the mean of means and the mean of standard errors
    return torch.mean(means).item(), torch.mean(std_errors).item()



def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)

    if args.data != "random":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device)
            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr, d.mean(), m.mean(),
                                                                                               (d - m).mean()))

    else:
        model.W.data = torch.randn_like(model.W.data) * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None

    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir+str('/')+str(args.data)), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    probabilities = gt_samples.mean(dim=0)

    # Step 3: Plot the probabilities
    """
    plt.figure(figsize=(15, 5))
    plt.bar(range(probabilities.size(0)), probabilities.numpy(),color='skyblue')
    plt.xlabel('Column Index')
    plt.ylabel('Probability of Success')
    plt.title('Probability of Success for Each Column')
    plt.grid(True)
    plt.show()
    """

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    # temps = [ 'dula','dmala','edula','edmala','edula-glu','edmala-glu','bg-1','hb-10-1','gwg-1']
    temp = args.sampler
    if temp == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(args.n_visible)
    elif temp == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
    elif "bg-" in temp:
        block_size = int(temp.split('-')[1])
        sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
    elif "hb-" in temp:
        block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
    elif temp == "gwg":
        sampler = samplers.DiffSampler(args.n_visible, 1,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif "gwg-" in temp:
        n_hops = int(temp.split('-')[1])
        sampler = samplers.MultiDiffSampler(args.n_visible, 1,
                                            approx=True, temp=2., n_samples=n_hops)

    elif temp == "dmala":
        sampler = samplers.LangevinSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.08,
                                           mh=True)
    elif temp == "edmala":
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.08, alpha_a=args.alpha_a,mh=True, eta=args.eta)

    #MNIST:<(0.01,1),>

    #EMNIST:<0.00001,0.1>      (0.00001,0.8)>
    #FASHION:<0.000001, 0.4>
    #KMNIST:<0.1,1>
    elif temp == "dula":
        sampler = samplers.LangevinSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1,
                                           mh=False)
    elif temp == "edula":
        sampler = samplers.ELangevinSampler(model.data_dim, 1,
                                            fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                            alpha=0.1, alpha_a=args.alpha_a, mh=False, eta=args.eta)

        # MNIST:<(0.01,1),>
        # EMNIST:
        # FASHION:
        # KMNIST:
    elif temp == "edula-glu":
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 1,
                                                   fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                   alpha=0.1, mh=False, eta=5)
        # <58,115,120>
    elif temp == "edmala-glu":
        sampler = samplers.ELangevinSamplerwithGLU(model.data_dim, 1,
                                                   fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                                   alpha=0.2, mh=True, eta=args.eta)
    elif temp=='fbp':
        array = np.load(f"FBP/rbm_fbp_{str(0)}.npy")

        j=0
        for sample in array:
            sample = (sample + 1) / 2
            x_reshaped = sample.reshape( 28, 28)  # 100 images, each 28x28

            # Loop through the 100 images and save each one individually
            fig = plt.figure(figsize=(28, 28), dpi=1)  # Set to 28x28 pixels
            plt.imshow(x_reshaped, cmap='gray', interpolation='none')
            plt.axis('off')  # Turn off axes

            # Define the save path for each image
            save_path = f"{args.save_dir}/{args.data}/{args.sampler}/{j}_img.png"
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

            j += 1  # Increment the counter for unique filenames
            plt.close(fig)  # Close the figure to save memory
        print("Done")

        exit()



    else:
        raise ValueError("Invalid sampler...")

    x = x0.clone().detach()
    x_a = x
    ars[temp] = []
    hops[temp] = []
    times[temp] = []
    tracking_time = []
    logmmd = []
    chain = []
    SE = []
    E = []
    E_e = []
    Surr_E = []
    Surr_E_e = []
    HD = []
    HD_e = []
    j=0
    cur_time = 0.
    st = time.time()
    for i in range(args.n_steps):
        # do sampling and time it

        if (sampler.entropy == False):
            xhat = sampler.step(x.detach(), model).detach()
        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            xahat = sample_tuple[1].detach()
        cur_time += time.time() - st

        # compute hamming dist
        cur_hops = (x != xhat).float().sum(-1).mean().item()
        cur_hops_std = torch.std(torch.sum((x != xhat).float(),dim=1)).item()/np.sqrt(100)

        # update trajectory
        x = xhat
        if (sampler.entropy == True):
            x_a = xahat

        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(x.cpu().numpy()[0][None])
            else:
                xc = x[0][None]
                h = (xc != gt_samples).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])



        if args.burnin<i:
            if i % args.viz_every == 0 and plot is not None:
                x_reshaped = x.view(100, 28, 28)  # 100 images, each 28x28

                # Loop through the 100 images and save each one individually
                for idx, image in enumerate(x_reshaped):
                    fig = plt.figure(figsize=(28, 28), dpi=1)   # 28x28 pixels image
                    plt.imshow(image.cpu().numpy(), cmap='gray', interpolation='none')
                    plt.axis('off')  # Turn off axes

                    save_path = f"{args.save_dir}/{args.data}/{args.sampler}/{i}_img_{idx}.png"
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    print(j)
                    j+=1
                    plt.close(fig)

            """
            p, p_e = neighbourhood_prob_generation(model, x,1)
            e = torch.mean((model(x))).item()
            e_e = torch.std((model(x))).item() / np.sqrt(100)

            h = cur_hops
            h_e = cur_hops_std

            print(i)
            print("Average Energy: " + str(e) + " " + str(e_e))
            print("Average Surrounding Energy: " + str(p) + " " + str(p_e))
            print("Average Hamming Distance: " + str(h) + " " + str(h_e))
            print("\n")

            E.append(e)
            E_e.append(e_e)

            HD.append(h)
            HD_e.append(h_e)

            Surr_E.append(p)
            Surr_E_e.append(p_e)

    E = np.array(E)
    E_e = np.array(E_e)

    N = range(0, len(E))
    plt.plot(N, E, marker='', label='Energy')
    plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)

    E = np.array(Surr_E)
    E_e = np.array(Surr_E_e)

    plt.plot(N, E, marker='', label='Surrounding Energy')
    plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)

    plt.xlabel('Iterations')
    plt.ylabel('Average Energy')
    plt.title('Energy Changes in RBM')
    plt.legend()
    plt.show()
    plt.clf()

    E = np.array(HD)
    E_e = np.array(HD_e)

    plt.plot(N, E, marker='')
    plt.fill_between(N, E - E_e, E + E_e, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel('Average Hamming Distance')
    plt.title('Consecutive Hamming Distance Changes in RBM')
    plt.legend()
    plt.show()
    plt.clf()
    """
    """
        if i % args.print_every == 0:
            hard_samples = x

            stat = kmmd.compute_mmd(hard_samples, gt_samples)
            log_stat = stat.log().item()
            running_time=time.time() - st
            #ste=bootstrap(hard_samples, gt_samples,kmmd,hard_samples.shape[0],1000)
            #ste1=batched_bootstrap(hard_samples, gt_samples,kmmd,hard_samples.shape[0],1000)

            tracking_time.append(running_time)
            logmmd.append(log_stat)
            times[temp].append(cur_time)
            hops[temp].append(cur_hops)
            #SE.append(ste)
            print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(temp, i, log_stat, cur_hops))

    chain = np.concatenate(chain, 0)
    #ess[temp] = get_ess(chain, args.burn_in)
    chains[temp] = chain
    #print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
    np.save("{}/rbm_sample_times_{}_{}_{}.npy".format(args.save_dir,args.seed, args.data, temp), tracking_time)
    np.save("{}/rbm_sample_logmmd_{}_{}_{}.npy".format(args.save_dir,args.seed, args.data, temp), logmmd)
    #np.save("{}/rbm_sample_logmmd_se_{}.npy".format(args.save_dir, temp), SE)

    """
    """
    plt.clf()
    for temp in temps:
        plt.plot(log_mmds[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/logmmd.png".format(args.save_dir))
    plt.clf()
    
    
    
    for temp in temps:
        plt.plot(times[temp],log_mmds[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/runtime.png".format(args.save_dir))
    """

if __name__ == "__main__":
    potential_datasets = [
        "mnist",
        "fashion",
        "emnist",
        "caltech",
        "omniglot",
        "kmnist",
        "random",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/rbm_sample")
    parser.add_argument('--data', choices=potential_datasets, type=str, default='mnist')

    parser.add_argument('--sampler', type=str, default='fbp')
    parser.add_argument('--alpha_a', type=float, default=0.001)
    parser.add_argument('--eta', type=float, default=0.4)

    parser.add_argument('--n_steps', type=int, default=52000)
    parser.add_argument('--burnin', type=int, default=1000)

    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--gt_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    # rbm def
    parser.add_argument('--n_hidden', type=int, default=500)
    parser.add_argument('--n_visible', type=int, default=784)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    args = parser.parse_args()

    main(args)
