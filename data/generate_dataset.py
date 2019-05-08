from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
import multiprocessing
import math


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)

def wrapper(args):
    """To be pickled, we need to define it here and manually unstar the arguments."""
    t = time.time()
    res = sim.sample_trajectory(T=args[1], sample_freq=args[2])
    if args[0] % 100 == 0:
        print("Iter: {}, Simulation time: {}".format(args[0], time.time() - t))
    return res

def generate_dataset(num_sims, length, sample_freq):

    # It is recommended to use num(cpu) / 2 to use skip hyperthreading (a bit dirty).
    pool = multiprocessing.Pool(math.ceil(multiprocessing.cpu_count() / 2))

    def arguments(num_sims):
        """Iterator that returns for every sim the args (step, length, sample_freq)
        to be fed to the wrapper.
        """
        for i in range(num_sims):
            yield i, length, sample_freq

    loc_all, vel_all, edges_all = zip(*list(pool.imap(wrapper, arguments(num_sims))))
    pool.close()
    pool.join()

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all

print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} test simulations".format(args.num_test))
loc_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                  args.length_test,
                                                  args.sample_freq)

np.save('loc_train' + suffix + '.npy', loc_train)
np.save('vel_train' + suffix + '.npy', vel_train)
np.save('edges_train' + suffix + '.npy', edges_train)

np.save('loc_valid' + suffix + '.npy', loc_valid)
np.save('vel_valid' + suffix + '.npy', vel_valid)
np.save('edges_valid' + suffix + '.npy', edges_valid)

np.save('loc_test' + suffix + '.npy', loc_test)
np.save('vel_test' + suffix + '.npy', vel_test)
np.save('edges_test' + suffix + '.npy', edges_test)
