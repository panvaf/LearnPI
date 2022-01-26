"""
Run simulation and store results.
"""

# imports
import time
import numpy as np
import fly_rec as rec
import utilities as util
import os
from pathlib import Path

# Simulation options
sim_run = "2Enough"
t_run = util.sim_time(sim_run)
store_f = False
train = True
adapt_gain = False

# Parameters
params = {
    'dt': 5*10**(-4),    # euler integration step size
    'n_neu': 60,         # number of HD neurons
    'v0': 2,             # vestibular input offset
    'v_max': 720,        # maximum angular velocity
    'M': 4,              # visual receptive field magnitude
    'sigma': .15,        # visual receptive field width
    'inh': - 1,          # global inhibitory input to HD neurons
    'inh_rot': - 1.5,    # global inhibitory input to rotation neurons
    'every_perc': 1,     # store value and show update this often
    'avg_err': 10,       # segment is s in which to compute current error of network
    'n_sigma': 0,        # input noise standard deviation
    'exc': 4,            # global excitation to soma of HD cells
    'run': 0,            # number of run for many runs of same simulation
    'tau_s': 65,         # synaptic delay in the network, in ms
    't_run': t_run,      # real time for which the network is run
    'sim_run': sim_run,  # designation corresponding to above time
    'gain': 1,           # neural velocity gain to entrain
    'sigma_v': 225,      # standard deviation of velocity noise in OU process
    'vary_w_rot': False, # whether to add variability in the HD to HR connections
    'adj': False,        # whether HD neurons project to adjacent HR neurons
    'rand_w_rot': False, # whether to randomly generate the HD to HR connections
    'filt': True,        # whether to filter the learning dynamics
    'tau_d': 100,        # synaptic plasticity time constant
    'x0': 1,             # input level for 50% of activation function
    'beta': 2.5,         # steepness of activation function
    'gD': 2,             # coupling conductance
    'gL': 1,             # leak conductance
    'fmax': .15,         # maximum firing rate in kHz (if saturated)
    'eta': 5e-2          # learning rate
    }

# Directory in which to load and save networks
data_path = str(Path(os.getcwd()).parent) + '\\savefiles\\trained_networks\\'

# Gain adaptation simulations
if adapt_gain:
    filename = "fly_rec" + util.filename(params)
    # Load weights from stored network
    data = np.load(data_path + filename + '.npz')
    w = data['w'][:,:,-1]
    try:
        w_rot = data['w_rot']
    except:
        w_rot = None
    # Specify options for gain adaptation simulation
    params['gain'] = adapt_gain
    sim_run = "4Medium"
    t_run = util.sim_time(sim_run)
    params['sim_run'] = sim_run
    params['t_run'] = t_run
    params['run'] = 0
else:
    w = None
    w_rot = None

filename = "fly_rec" + util.filename(params) 

# run simulation

start = time.time()

# Generate movement trajectory
theta0, v_ang_true, t = util.gen_theta0_OU(t_run,sigma=params['sigma_v'])

# Call main simulation routine
w, w_rot, f, f_rot, err = rec.simulate(t_run,theta0,params,store_f,train,w,w_rot)

# Save results
np.savez(data_path+filename,w=w,w_rot=w_rot,f=f,f_rot=f_rot,err=err,
         params=params,allow_pickle=True)

end = time.time()
print("The simulation ran for {} hours".format(round((end-start)/3600,2)))