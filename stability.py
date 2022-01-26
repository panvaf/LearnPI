"""
Check stable states of the network and their robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import fly_rec as rec
import utilities as util
from astropy.stats.circstats import circmean
import os
from pathlib import Path
from random import choice

# Simulation options
sim_run = "2Enough"
t_run = util.sim_time(sim_run)
store_f = True
train = False
stab = True
d_coeff = False
n_levels = np.arange(0,1.1,.1)

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

# load network
data_path = str(Path(os.getcwd()).parent) + '\\savefiles\\trained_networks\\'
filename = 'fly_rec' + util.filename(params)
network = np.load(data_path+filename+'.npz')

# Change parameters for stimulation
params['M'] = 16
params['sigma'] = .25

w = network['w'][:,:,-1]
try:
    w_rot = network['w_rot']
except:
    w_rot = None

n_neu = np.size(w,0)
n_pos = int(n_neu/2)
n_dir = 6                 # for all possible bump positions set n_dir = n_pos
pref_dir = [36, 108, 288, 216, 276, 144]  # Stimulation locations in Fig. 2D
# Stimulate randomly
# pref_dir = np.random.choice(360*np.linspace(0,n_pos-1,n_pos)/n_pos,n_dir,replace=False)
dphi = 360/n_pos
f_max = 150; f_min = 0

# Create artificial data to feed the network with
sec_per_dir = 5
dur_light = 2
reps = int(sec_per_dir/params['dt'])
light = int(dur_light/params['dt'])

# Position of landmark
theta0 = np.repeat(pref_dir,reps)
n_th = len(theta0)

# Indicator of when there is light and when not
day = np.zeros(reps)
day[0:light] = 1
day = np.repeat(day[np.newaxis,...], n_dir, axis=0).flatten().astype(int)

# Simulate
t_run = sec_per_dir*n_dir
xi = np.linspace(0,t_run/params['dt'],n_dir+1)
x = np.linspace(0,t_run,n_dir+1).astype(np.int)

w,w_rot,f,f_rot,err = rec.simulate(t_run,theta0,params,store_f,train,w,w_rot,day,stab)

# find position of the bump at every instant as the center of mass of neural activity
pos = 360*np.linspace(0,n_pos-1,n_pos)/n_pos
pos = np.repeat(pos,2)

bump_pos = util.bump_COM(f,pos)
theta = theta0 % 360; theta = 360 - theta
theta_f = theta * n_neu / 360

# check if the bump is in the same position at the end of the period as at the start

n_same = 0
avg = 100     # how many snapshots to average position of bump on
offset = 100  # offset from end of position period
margin = dphi

for i in range(n_dir):
    true_dir = pref_dir[i]
    avg_dir = circmean(bump_pos[((i+1)*reps - avg - offset):((i+1)*reps - offset)]/180*np.pi)
    diff = 2*circmean(np.array([avg_dir, true_dir/180*np.pi])); diff = abs(diff*180/np.pi)
    if (diff > 360 - margin) or (diff < margin):
        n_same +=1

print('The number of stable locations in the network is {}'.format(n_same))

# Fontsize appropriate for plots
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# Stimulation plot
ticks_HD = np.linspace(0,n_neu,3)
t_ticks = [-180,0,180]

true_col = 'limegreen'
dur = .0475
offst = .002
vert_width = .75
stim_col = 'red'
line_sz = .0001

fig, ax = plt.subplots(figsize=(6.5,1))
im = ax.imshow(f*1000, interpolation='nearest', aspect='auto', cmap = 'Greys', origin='lower')
ax.scatter(range(n_th),theta_f, color=true_col, s = line_sz)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Heading ($\degree$) \n HD')
ax.set_xlim((0,sec_per_dir*n_dir/params['dt']))
ax.set_ylim((0,n_neu))
plt.xticks(xi,x)
plt.yticks(ticks_HD,t_ticks)
ax.yaxis.set_minor_locator(MultipleLocator(15))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -500))
ax.spines['bottom'].set_position(('data', -1))
ax_pos = ax.get_position()
ax_cond = fig.add_axes([0,ax_pos.y1,1,.1])
plt.axis('off')
end_pos = ax_pos.x0 + offst
for i in range(n_dir):
    start_pos = end_pos
    end_pos = ax_pos.x0 + (i+1)/n_dir*(ax_pos.x1-ax_pos.x0) + offst
    ax_cond.axhline(y=.4, xmin=start_pos, xmax=start_pos+dur, color=stim_col, linewidth = 2)
    ax.axvline(x=xi[i],color=stim_col,linestyle='--',linewidth=vert_width)
ax.plot([],ls='-',color=stim_col,linewidth=2,label='Stimulation')
ax.plot([],ls='-',color=true_col,linewidth=.75,label='True heading')
fig.legend(loc='upper center',  bbox_to_anchor=(0.55, 1.6),\
           ncol=2,prop={'size': SMALL_SIZE},markerscale=2.3,frameon=False)
ax_bar = fig.add_axes([.07,ax.get_position().y0,1,ax.get_position().y1-ax.get_position().y0])
fig.colorbar(im,label='Firing rate\n(spikes/s)',ticks=[0, 50, 100, 150])
plt.axis('off')


# Quantify diffusion coefficient

if d_coeff:
    
    t_sim = 10      # Simulation time, in seconds
    n_sim = 1000    # Number of simulations per noise level
    seg_dur = np.arange(10,20,10); seg_n = (seg_dur/params['dt']).astype(int)
    n_t = int(t_sim/params['dt'])
    init_dirs = 360*np.linspace(0,n_pos-1,n_pos)/n_pos
    day = np.zeros(n_t)
    day[0:light] = 1
    
    Dxs = np.zeros((len(n_levels),n_sim,len(seg_dur)))
    
    for i, n_level in enumerate(n_levels):
        
        params['n_sigma'] = n_level
        
        for j in range(n_sim):
            print('Simulation {} out of {}'.format(j+1,n_sim))
            print('Test noise sigma_n = {}'.format(n_level))
            
            init_dir = choice(init_dirs)
            theta0 = np.repeat(init_dir,n_t)
            w,w_rot,f,f_rot,err = rec.simulate(t_sim,theta0,params,store_f,train,w,w_rot,day)
            bump_pos = util.bump_COM(f,pos)
            Dx = np.diff(bump_pos)
            
            # Correct jumps
            bump_size = int(n_neu/12); bump_size = bump_size*dphi 
            Dx[Dx<-bump_size] = Dx[Dx<-bump_size] + 360
            Dx[Dx>bump_size] = Dx[Dx>bump_size] - 360
            
            for k, seg in enumerate(seg_n):
                # Total change in position
                Dxs[i,j,k] = np.sum(Dx[light+5:seg])
    
    Dxs[np.abs(Dxs)>1e6] = np.nan
    D_coeff = np.divide(np.nanvar(Dxs,axis=1),seg_dur-dur_light)