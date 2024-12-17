import julia
from julia import BinaryCommitteeMachineFBP
import argparse

from numpy.f2py.crackfortran import quiet

import rbm
import torch
import numpy as np
import samplers
import os
import torchvision

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools


# Initialize Julia environment
j = julia.Julia(compiled_modules=False)

j.eval("""
function get_all_configurations(patterns)
    configurations = getfield(patterns, :X)  # Access all configurations stored in `X`
    return configurations                    # Return the entire list of configurations
end
""")
# Import the helper function from Julia's main namespace
from julia.Main import get_all_configurations
def main(args):
    # Define your parameters
    task=args.task
    seed = args.seed
    max_iters = args.max_iters
    K = args.K
    patterns_spec = args.pattern_spec
    damping = args.damping

    if task=='ising':
        N = 25# Number of input nodes
        samples=5000

    elif task=='tsp':
        print('tsp')
        N=24
        samples = 8000
    elif task=='rbm':
        print('rbm')
        N = 784# Number of input nodes
        K=7
        samples=5000

    else:
        samples = 1000+1

        print("bbnn")
        d=args.dataset
        print(d)

        if d=='compas':
            K = 7
            N=1501
        elif d=='news':
            K = 5
            N=6001
        elif d=='adult':
            K = 5
            N=9001
        else:
            K = 3
            N=27801


    max_steps=10
    try:
        # Run fBP to get patterns
        output = []
        i = 0
        while True:
            if len(output) >= samples:
                break
            try:
                # Run the fBP function and catch NaN values
                errors, messages, patterns = BinaryCommitteeMachineFBP.focusingBP(N, K, patterns_spec,
                                                                                  max_iters=max_iters,max_steps=max_steps, seed=seed + i,
                                                                                  damping=damping)

                # Process patterns if no NaN values are found
                all_configurations = get_all_configurations(patterns)
                output.extend(all_configurations)
                i += 1
                print(len(output))
            except Exception as e:
                print(f"Error running fBP: {e}")
                i += 1  # Increment to try a new seed or configuration
                continue

        print("Done")

        # Save the output, trimmed to the required number of samples
        output = output[:samples]
        if task!='bbnn':
            np.save("{}/{}_{}_{}.npy".format(args.save_dir, args.task, 'fbp', seed), output)
        else:
            np.save("{}/{}_{}_{}_{}.npy".format(args.save_dir, args.task, 'fbp', d, seed), output)

    except Exception as final_error:
        print(f"Final error: {final_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./FBP")
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--sampler', type=str, default='fbp')
    parser.add_argument('--dataset', type=str, default='news')
    parser.add_argument('--N', type=int, default=25)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--pattern_spec', type=float, default=0.2)
    parser.add_argument('--damping', type=float, default=0.8)

    parser.add_argument('--seed', type=int, default=0)
    # model def
    parser.add_argument('--task', type=str, default='bbnn')


    args = parser.parse_args()

    main(args)


