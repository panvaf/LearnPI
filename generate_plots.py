"""
Produce figures for a single trained network.
"""

# imports
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import fly_rec as rec
import utilities as util
import copy
from cycler import cycler
from astropy.stats.circstats import circmean
import os
from pathlib import Path

# Simulation options
sim_run = "2Enough"
t_run = util.sim_time(sim_run)
vel_gain = True                # Whether to compute and plot the neural velocity gain plot
vel_hist = False               # Whether to plot velocity histograms
PI_err = True                  # Whether to produce histogram of errors after PI in darkness
cut_exc = False                # Whether to exterminate extended positive sidelobes or not
perturb_conn = False           # Add noise to final connectivity to account for biological irregularities
save = False                   # Whether to save results
n_hist = 1001                  # Number of bins for angular velocity histograms
prop_std = .2                  # Noise level as a fraction of variability in connectivity
width = 13                     # Width of extermination
offset = 4                     # Offset for start of extermination
n_PI = 1000                    # Number of examples for PI in darkness error
PI_dur = np.arange(10,70,10)   # Duration of PI in darkness segments, in sec
avg = 20                       # How many snapshots to average position of bump on
data_dir = '\\savefiles\\trained_networks\\'
PI_example_dir = '\\savefiles\\PI_example.npz'

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

tau_tot = params['tau_s']/1000 + 0.01
if params['gain']<0:
    inv = True
else:
    inv = False

# load simulation results
data_path = str(Path(os.getcwd()).parent) + data_dir
filename = "fly_rec" + util.filename(params)
network = np.load(data_path+filename+'.npz',allow_pickle=True)
w = network['w'][:,:,-1]
try:
    w_rot = network['w_rot']
except:
    w_rot = None
W = network['w']
error = network['err']
stored = []

try:
    hd_v = network['hd_v']; neu_v_dark = network['neu_v_dark']
    neu_v_day = network['neu_v_day']; stored.append('neu_v')
except:
    hd_v = np.nan; neu_v_dark = np.nan; neu_v_day = np.nan

try:
    PI_errors = network['PI_errors']; 
    if not np.all(np.isnan(PI_errors)):
        stored.append('PI_err')
except:
    PI_errors = np.nan

# Constants
n_neu = np.size(w,0)
bump_size = int(n_neu/12)
dphi = 720/n_neu
theor_lim = 80/tau_tot
actual_lim = 1100
colormap = 'seismic'
fly_max = 500

# Fontsize appropriate for plots
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title

# Visual input
th = np.linspace(-np.pi,np.pi,1000)
theta0 = 0
r = util.vis_in(th,theta0,params['M'],params['sigma'],-params['exc']-1,day=True)

fig, ax = plt.subplots(figsize=(1.5,1))
plt.plot(th*180/np.pi,r,c='dodgerblue',linewidth=2)
plt.ylabel('Visual input')
plt.xlabel('Offset from current HD ($\degree$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -190))
ax.spines['bottom'].set_position(('data', -5.2))
ax.xaxis.set_major_locator(MultipleLocator(180))
ax.xaxis.set_minor_locator(MultipleLocator(90))
ax.set_yticks([-1,-3,-5])
ax.yaxis.set_minor_locator(MultipleLocator(1))
plt.xlim([-180,180])

# Vestibular input
vel = np.array([720,0,-360])
col_cycle = cycler('color', ['green','dodgerblue','darkorange'])
th = np.linspace(-180,180,int(n_neu/2))
k = params['v0']/params['v_max']
sign = np.ones(n_neu)
sign[int(n_neu/2):n_neu] = -1
v = k*np.dot(vel[:,np.newaxis],sign[np.newaxis,:])

fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(3,1.5))
ax1.set_prop_cycle(col_cycle)
ax1.plot(th,v[:,0:int(n_neu/2)].T,linewidth=2)
ax1.set_xlabel('L-HR')
ax1.set_ylabel('Velocity input')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_position(('data', -190))
ax1.spines['bottom'].set_position(('data', -2.2))
ax1.xaxis.set_major_locator(MultipleLocator(180))
ax1.xaxis.set_minor_locator(MultipleLocator(90))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.set_xlim([-180,180])
ax2.set_prop_cycle(col_cycle)
ax2.plot(th,v[:,int(n_neu/2):].T,linewidth=2)
ax2.set_xlabel('R-HR')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_position(('data', -190))
ax2.spines['bottom'].set_position(('data', -2.2))
fig.text(0.55, 0.04, 'Heading ($\degree$)', ha='center', va='center')
fig.text(0.55, .93, '720 $\degree$/s', ha='right', va='center', fontsize = SMALL_SIZE)
fig.text(0.55, .68, '0 $\degree$/s', ha='right', va='center', fontsize = SMALL_SIZE)
fig.text(0.55, .57, '-360 $\degree$/s', ha='right', va='center', fontsize = SMALL_SIZE)
plt.tight_layout()

# Average error history
t = np.linspace(0,t_run,100)/3600
error_hist = np.mean(error,axis = 0)*1000

fig, ax = plt.subplots(figsize=(3,2))
plt.plot(t,error_hist,color = 'dodgerblue',linewidth=2)
plt.ylabel('Absolute error (spikes/s)')
plt.xlabel('Training time (hours)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.4))
ax.spines['bottom'].set_position(('data', -.2))
plt.xlim([0,t[-1]])
plt.ylim([0,np.max(error_hist)])

# Plot weight matrices
w_hr = w[:,0:n_neu]; w_rec = w[:,n_neu:2*n_neu]
ticks = np.array([0,int(n_neu/2),n_neu-1])
ticks = ticks.astype(np.int)
r_ticks = [ticks[0]+1,ticks[1]+1,ticks[2]+1]

# Recurrent connections
vmax = np.max(w_rec); vmin = np.min(w_rec)
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)

fig, ax = plt.subplots(figsize = (3.5,2.5))
im = ax.imshow(w_rec, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm)
ax.set_title('$W^{rec}$')
ax.set_ylabel('# of postsynaptic neuron \n HD')
ax.set_xlabel('HD \n # of presynaptic neuron')
ax.set_xticks(ticks)
ax.set_xticklabels(r_ticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.yticks(ticks,r_ticks)
ax2 = fig.add_axes([.1,ax.get_position().y0,.85,ax.get_position().y1-ax.get_position().y0])
fig.colorbar(im,ticks = [np.ceil(vmin),0,np.floor(vmax)])
plt.axis('off')
fig.text(.95, (ax.get_position().y0+ax.get_position().y1)/2, 
        'Synaptic strength', ha='center', va='center', rotation='vertical')

# HR to HD connections
vmax = np.max(w_hr); vmin = np.min(w_hr)
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)

fig, ax = plt.subplots(figsize = (3.5,2.5))
im = ax.imshow(w_hr, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm)
ax.set_title('$W^{HR}$')
ax.set_ylabel('# of postsynaptic neuron \n HD')
ax.set_xlabel('L-HR               R-HR \n # of presynaptic neuron')
ax.set_xticks(ticks)
ax.set_xticklabels(r_ticks)
plt.yticks(ticks,r_ticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax2 = fig.add_axes([.1,ax.get_position().y0,.85,ax.get_position().y1-ax.get_position().y0])
fig.colorbar(im,ticks = [np.ceil(vmin),0,np.floor(vmax)])
plt.axis('off')
fig.text(.95, (ax.get_position().y0+ax.get_position().y1)/2, 
        'Synaptic strength', ha='center', va='center', rotation='vertical')

# HD to HR connections
if w_rot is not None:    
    vmax = np.max(w_rot); vmin = np.min(w_rot)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize = (3.5,2.5))
    im = ax.imshow(w_rot, cmap = 'Reds',vmax = vmax, vmin = vmin)
    ax.set_title('$W^{HD}$')
    ax.set_ylabel('# of postsynaptic neuron \nL-HR                R-HR')
    ax.set_xlabel('HD \n # of presynaptic neuron')
    ax.set_xticks(ticks)
    ax.set_xticklabels(r_ticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.yticks(ticks,r_ticks)
    ax2 = fig.add_axes([.1,ax.get_position().y0,.85,ax.get_position().y1-ax.get_position().y0])
    fig.colorbar(im,ticks = [np.ceil(vmin),0,np.floor(vmax)])
    plt.axis('off')
    fig.text(.95, (ax.get_position().y0+ax.get_position().y1)/2, 
            'Synaptic strength', ha='center', va='center', rotation='vertical')


# Plot weight profile history
percs = np.arange(1,100)
w_rec_hist = np.zeros((n_neu,percs.size)); w_rec_hist_norm = np.zeros((n_neu,percs.size))
w_l_hr_hist = np.zeros((n_neu,percs.size)); w_r_hr_hist = np.zeros((n_neu,percs.size));
w_l_hr_hist_norm = np.zeros((n_neu,percs.size)); w_r_hr_hist_norm = np.zeros((n_neu,percs.size))

for i, perc in enumerate(percs):
    # Get weight matrices at specific time in training
    temp1 = copy.deepcopy(W[:,n_neu:(2*n_neu),perc])
    temp2 = copy.deepcopy(W[:,0:n_neu,perc])
    # Average weights over neurons
    for j in range(n_neu):    
        # Displace the rows so that the weight matrices can be averaged across 
        # receptive field difference
        if j%2:
            temp1[:,j] = np.roll(temp1[:,j],int(n_neu/2-j+1))
        else:
            temp1[:,j] = np.roll(temp1[:,j],int(n_neu/2-j))
        temp2[:,j] = np.roll(temp2[:,j],int(n_neu/2-2*j))
        
    # Mean and std of weights
    hd_slice = np.mean(temp1,axis=1); l_hr_slice = np.mean(temp2[:,0:int(n_neu/2)],axis=1)
    r_hr_slice = np.mean(temp2[:,int(n_neu/2):n_neu],axis=1)
    hd_sd = np.mean(np.std(temp1,axis=1)); l_hr_sd = np.mean(np.std(temp2[:,0:int(n_neu/2)],axis=1))
    r_hr_sd = np.mean(np.std(temp2[:,int(n_neu/2):n_neu],axis=1))
    
    w_rec_hist[:,i] = hd_slice; w_l_hr_hist[:,i] = l_hr_slice
    w_r_hr_hist[:,i] = r_hr_slice

# History of recurrent connections
x_ticks_hist = np.round(t_run*np.array([1,10,100])/100/3600,1)
ticks_HD = np.linspace(0,n_neu,5)
ticks_HD_less = np.linspace(0,n_neu,3)
t_ticks_more = [-180,-90,0,90,180]
ticks_HR = np.linspace(0,int(n_neu/2),3)
t_ticks_less = [-180,0,180]

vmax = np.max(w_rec_hist); vmin = np.min(w_rec_hist)
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
fig, ax = plt.subplots(figsize=(8/3,2))
im = ax.imshow(w_rec_hist, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm, aspect='auto')
ax.set_title('$W^{rec}$')
ax.set_ylabel('Receptive field difference ($\degree$)')
ax.set_xlabel('Training time (hours)')
plt.yticks(ticks_HD,t_ticks_more)
ax.set_xscale('log')
ax.set_xlim([1,100])
ax.set_xticks([1,10,100])
ax.set_xticklabels(x_ticks_hist)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', 61))
ax2 = fig.add_axes([.1,.1,1,.8])
fig.colorbar(im,ticks = [np.ceil(vmin),0,np.floor(vmax)])
plt.axis('off')
fig.text(1.1, (ax.get_position().y0+ax.get_position().y1)/2, 
         'Synaptic strength', ha='center', va='center', rotation='vertical')

# History of HR to HD connections
vmax = np.max(np.concatenate((w_l_hr_hist,w_r_hr_hist))); vmin = np.min(np.concatenate((w_l_hr_hist,w_r_hr_hist)))
norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
fig = plt.figure(figsize=(8/3,2))
ax1 = plt.subplot(211)
im = ax1.imshow(w_l_hr_hist, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm, aspect='auto')
ax1.set_title('$W^{HR}$')
ax1.set_ylabel('L-HR')
plt.yticks(ticks_HD_less,t_ticks_less)
ax1.set_xscale('log')
ax1.set_xlim([1,100])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('data', 62))
ax1.set_xticklabels([])
ax1.yaxis.set_minor_locator(MultipleLocator(15))
ax2 = plt.subplot(212)
im = ax2.imshow(w_r_hr_hist, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm, aspect='auto')
ax2.set_xlabel('Training time (hours)')
ax2.set_ylabel('R-HR')
plt.yticks(ticks_HD_less,t_ticks_less)
ax2.yaxis.set_minor_locator(MultipleLocator(15))
ax2.set_xscale('log')
ax2.set_xlim([1,100])
ax2.set_xticks([1,10,100])
ax2.set_xticklabels(x_ticks_hist)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position(('data', 62))
plt.subplots_adjust(hspace=0.25)
ax3 = fig.add_axes([.1,.1,1,.8])
fig.colorbar(im,ticks = [np.ceil(vmin),0,np.floor(vmax)])
plt.axis('off')
fig.text(-.13, (ax1.get_position().y0+ax2.get_position().y1)/2, 
         'Receptive field difference ($\degree$)', ha='center', va='center', rotation='vertical')
fig.text(1.1, (ax1.get_position().y0+ax2.get_position().y1)/2, 
         'Synaptic strength', ha='center', va='center', rotation='vertical')


# Weight profiles

deg_HR = np.linspace(-180,180,int(n_neu/2))
deg_HD = np.linspace(-180,180,n_neu)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(3,2))
ax1.plot(deg_HD,w_rec_hist[:,-1],color='dodgerblue',linewidth=3)
ax1.set_title('$W^{rec}$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_position(('data', -190))
ax1.spines['bottom'].set_position(('data', -11))
ax1.set_xlim([-180,180])
ax1.set_xticks([-180,-90+dphi/4,0+dphi/2,90+dphi/4,180])
ax1.set_xticklabels([])
ax1.set_yticks([-10,0,10])
ax2.plot(deg_HD,w_l_hr_hist[:,-1],color='green',linewidth=3)
ax2.plot(deg_HD,w_r_hr_hist[:,-1],color='darkorange',linewidth=3)
plt.legend(['L-HR','R-HR'],prop={'size': SMALL_SIZE},frameon=False, \
           bbox_to_anchor=(.3, 1.45))
ax2.set_title('$W^{HR}$')
ax2.set_ylabel('Synaptic strength', y=1.1)
ax2.set_xlabel('Receptive field difference ($\degree$)')
plt.subplots_adjust(hspace=0.5)
ax2.set_xticks([-180,-90+dphi/4,0+dphi/2,90+dphi/4,180])
ax2.set_xticklabels([-180,-90,0,90,180])
ax2.set_yticks([-20,0,20])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_position(('data', -190))
ax2.spines['bottom'].set_position(('data', -24))
ax2.set_xlim([-180,180])
plt.show()

print('SD for recurrent connections is {}'.format(round(hd_sd,2)))
print('SD for L-HR connections is {}'.format(round(l_hr_sd,2)))
print('SD for R-HR connections is {}'.format(round(l_hr_sd,2)))
    

# Add noise to connectivity

if perturb_conn:
    data_path += 'Parallel\\Perturb_Conn\\'
    params['run'] = 0
    filename = "fly_rec" + util.filename(params)
    np.random.seed(params['run'])
    
    std = np.std(w)
    w += np.random.normal(0,prop_std*std,(n_neu,2*n_neu))
    W[:,:,-1] = w
    w_hr = w[:,0:n_neu]; w_rec = w[:,n_neu:2*n_neu]
    
    # plot new weight matrices
    vmax = np.max(w); vmin = np.min(w)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)

    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize = (10,20))
    im1 = ax1.imshow(w_rec, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm)
    ax1.set_title('$W^{rec}$')
    ax1.set_ylabel('# of postsynaptic neuron \n HD')
    ax1.set_xlabel('HD')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(r_ticks)
    plt.yticks(ticks,r_ticks)
    
    im2 = ax2.imshow(w_hr, cmap = colormap,vmax = vmax, vmin = vmin, norm = norm)
    ax2.set_title('$W^{HR}$')
    plt.xlabel('L-HR               R-HR')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(r_ticks)
    plt.subplots_adjust(hspace=2)
    fig.text(0.5, .37, '# of presynaptic neuron', ha='center', va='center', rotation='horizontal')
    ax3 = fig.add_axes([.1,.415,1,.175])
    fig.colorbar(im2,ticks = [np.ceil(vmin),0,np.floor(vmax)],label = 'Synaptic strength')
    plt.axis('off')
    plt.show()
    

# Integration in light and darkness

# Load example simulation
PI_example = np.load(str(Path(os.getcwd()).parent) + PI_example_dir)
t_example = 60
dt = 5*10**(-4)
theta0 = PI_example['theta0']
v_ang = PI_example['v_ang']
theta = params['gain'] * theta0 % 360; theta = 360 - theta
theta_h = theta * n_neu / 720; theta_f = theta * n_neu / 360

xi = np.linspace(0,t_example/dt,7)
x = np.linspace(0,t_example,7).astype(np.int)
n_th = len(theta)

# Day initially, then dark, then day again
dark_dur = t_example*1/3
day_dur = t_example*1/2
dark_num = int(dark_dur/dt)
day_num = int(day_dur/dt)
day = np.ones(n_th)
day[dark_num:dark_num+day_num] = 0

w, w_rot, f, f_rot, err = rec.simulate(t_example,theta0,params,True,False,w,w_rot,day)


# find position of the bump at every instant as the center of mass of neural activity
n_pos = int(n_neu/2)
pos = 360*np.linspace(0,n_pos-1,n_pos)/n_pos
pos = np.repeat(pos,2)
bump_pos = util.bump_COM(f,pos)

# Path integration error
PI_error = bump_pos - theta
PI_error[np.abs(PI_error)>180] = PI_error[np.abs(PI_error)>180] - np.sign(PI_error[np.abs(PI_error)>180])*360
PI_error[0] = 0   # to avoid jump in the beginning

# PI example plot

f_rot_r = f_rot[0:int(n_neu/2),:]; f_rot_l = f_rot[int(n_neu/2):n_neu,:]
vmax = np.max(f_rot[:,int(0.5/dt):-1]*1000)

f_max = params['fmax']*1e3; f_min = 0
true_col = 'limegreen'
line_sz = .0001
true_alpha = .5
light_col = 'gold'
light_alpha = .6
dark_col = 'darkslateblue'
err_lim = 90
vel_lim = 720
fr_col = 'Greys'
vert_width = .75
markerscale = 5

fig, axs = plt.subplots(5,1,sharex=True,figsize=(6.5,5),gridspec_kw={
                           'width_ratios': [1],'height_ratios': [5,5,5,4,4]})
im = axs[0].imshow(f*1000, interpolation='nearest', aspect='auto', cmap = fr_col, vmax = f_max, vmin = f_min, origin='lower')
axs[0].scatter(range(n_th),theta_f, color=true_col, s = line_sz)
axs[0].set_ylabel('HD')
axs[0].set_ylim((0,n_neu))
axs[0].set_yticks(ticks_HD_less)
axs[0].set_yticklabels(t_ticks_less)
axs[0].yaxis.set_minor_locator(MultipleLocator(15))
axs[0].set_xlim((0,t_example/dt))
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_position(('data', -500))
axs[0].spines['bottom'].set_position(('data', -1))
axs[0].plot([],ls='-',color=light_col,linewidth=2,alpha=light_alpha,label='Light')
axs[0].plot([],ls='-',color=dark_col,linewidth=2,label='Dark')
axs[0].plot([],ls='-',color=true_col,linewidth=.75,label='True heading')
fig.legend(loc='upper center', bbox_to_anchor=(0.6, 1.03),markerscale=markerscale,ncol=3,frameon=False)
axs[0].axvline(x=n_th/3,color=dark_col,linestyle='--',linewidth=vert_width)
axs[0].axvline(x=n_th*5/6,color=dark_col,linestyle='--',linewidth=vert_width)

axs[1].scatter(range(n_th),theta_h, color=true_col, s = line_sz, alpha=true_alpha)
axs[1].imshow(f_rot_r*1000, interpolation='nearest', aspect='auto', cmap = fr_col, vmax = f_max, vmin = f_min, origin='lower')
axs[1].axvline(x=n_th/3,color=dark_col,linestyle='--',linewidth=vert_width)
axs[1].axvline(x=n_th*5/6,color=dark_col,linestyle='--',linewidth=vert_width)
axs[1].set_ylabel('L-HR')
axs[1].set_yticks(ticks_HR)
axs[1].set_yticklabels(t_ticks_less)
axs[1].yaxis.set_minor_locator(MultipleLocator(7.5))
axs[1].set_ylim((0,n_neu/2))
axs[1].set_xlim((0,t_example/dt))
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_position(('data', -500))
axs[1].spines['bottom'].set_position(('data', -1))
fig.text(0.03, axs[1].get_position().y1-0.01, 
         'Heading ($\degree$)', ha='center', va='center', rotation='vertical')

axs[2].scatter(range(n_th),theta_h, color=true_col, s = line_sz,alpha=true_alpha)
axs[2].imshow(f_rot_l*1000, interpolation='nearest', aspect='auto', cmap = fr_col, vmax = f_max, vmin = f_min, origin='lower')
axs[2].axvline(x=n_th/3,color=dark_col,linestyle='--',linewidth=vert_width)
axs[2].axvline(x=n_th*5/6,color=dark_col,linestyle='--',linewidth=vert_width)
axs[2].set_ylabel('R-HR')
axs[2].set_yticks(ticks_HR)
axs[2].set_yticklabels(t_ticks_less)
axs[2].yaxis.set_minor_locator(MultipleLocator(7.5))
axs[2].set_ylim((0,n_neu/2))
axs[2].set_xlim((0,t_example/dt))
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_position(('data', -500))
axs[2].spines['bottom'].set_position(('data', -1))

axs[3].plot(range(n_th),np.zeros(n_th), color='grey', linewidth = .5)
axs[3].plot(range(n_th),PI_error, color='firebrick', linewidth = 1)
axs[3].axvline(x=n_th/3,color=dark_col,linestyle='--',linewidth=vert_width)
axs[3].axvline(x=n_th*5/6,color=dark_col,linestyle='--',linewidth=vert_width)
axs[3].set_ylabel('PI error\n($\degree$)')
axs[3].set_yticks([-err_lim,0,err_lim])
axs[3].set_yticklabels([-err_lim,0,err_lim])
axs[3].yaxis.set_minor_locator(MultipleLocator(err_lim/2))
axs[3].set_ylim((-err_lim,err_lim))
axs[3].set_xlim((0,t_example/dt))
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)
axs[3].spines['left'].set_position(('data', -500))
axs[3].spines['bottom'].set_position(('data', -err_lim*1.1))

axs[4].plot(range(n_th),v_ang, color='dodgerblue', linewidth = .5)
axs[4].axvline(x=n_th/3,color=dark_col,linestyle='--',linewidth=vert_width)
axs[4].axvline(x=n_th*5/6,color=dark_col,linestyle='--',linewidth=vert_width)
axs[4].set_ylabel('Velocity\n($\degree$/s)')
axs[4].set_xlabel('Time (s)')
axs[4].set_yticks([-vel_lim,0,vel_lim])
axs[4].set_yticklabels([-vel_lim,0,vel_lim])
axs[4].yaxis.set_minor_locator(MultipleLocator(vel_lim/2))
axs[4].set_ylim((-vel_lim,vel_lim))
axs[4].set_xlim((0,t_example/dt))
axs[4].spines['top'].set_visible(False)
axs[4].spines['right'].set_visible(False)
axs[4].spines['left'].set_position(('data', -500))
axs[4].spines['bottom'].set_position(('data', -vel_lim*1.1))
axs[4].set_xticks(xi)
axs[4].set_xticklabels(x)
plt.tight_layout()

# Light overbars
dur0 = .275
dur1 = .135
axs_cond = []
plot_i = [0,3]
counter = 0
for i, ax in enumerate(axs):
    if i in plot_i:
        ax_pos = ax.get_position()
        axs_cond.append(fig.add_axes([0,ax_pos.y1,1,.1]))
        plt.axis('off')
        axs_cond[counter].axhline(y=.2, xmin=ax_pos.x0, xmax=ax_pos.x0+dur0, color=light_col, linewidth = 2, alpha = light_alpha)
        axs_cond[counter].axhline(y=.2, xmin=ax_pos.x1-dur1, xmax=ax_pos.x1 , color=light_col, linewidth = 2, alpha = light_alpha)
        axs_cond[counter].axhline(y=.2, xmin=ax_pos.x0+dur0, xmax=ax_pos.x1-dur0/2, color=dark_col, linewidth = 2)
        counter += 1
# Colorbar
ax_bar = fig.add_axes([.1,axs[2].get_position().y0,1.05,axs[0].get_position().y1-axs[2].get_position().y0])
fig.colorbar(im,label='Firing rate (spikes/s)', ticks = [round(f_min),round(f_max/3),round(2*f_max/3),round(f_max)])
plt.axis('off')

# Error plots
xi_err = np.linspace(0,t_example/dt/2,7)
x_err = np.linspace(0,t_example/2,7).astype(np.int)
plt_range = int(n_th/2)
_, _, f_light, _, err_light = rec.simulate(t_example/2,theta0[0:plt_range],params,True,False,w,w_rot,True)
_, _, f_dark, _, err_dark = rec.simulate(t_example/2,theta0[0:plt_range],params,True,False,w,w_rot,False)

bump_pos_light = util.bump_COM(f_light,pos)
bump_pos_dark = util.bump_COM(f_dark,pos)

vmax_light = np.max(np.abs(err_light[:,int(0.5/dt):-1]*1000)); vmin_light = -vmax_light
vmax_dark = np.max(np.abs(err_dark[:,int(0.5/dt):-1]*1000)); vmin_dark = -vmax_dark

fig, axs = plt.subplots(2,1,sharey=True,sharex=True,figsize=(6.5,3))
im0 = axs[0].imshow(err_light*1000, interpolation='nearest', aspect='auto', cmap = colormap, vmax = vmax_light, vmin = vmin_light, origin='lower')
axs[0].scatter(range(n_th)[0:plt_range],bump_pos_light/int(360/n_neu), color='black', s = .000001, alpha=.5)
axs[0].set_ylabel('HD')
axs[0].set_ylim((0,n_neu))
plt.xticks(xi_err,x_err)
plt.yticks(ticks_HD_less,t_ticks_less)
axs[0].yaxis.set_minor_locator(MultipleLocator(15))
axs[0].set_xlim((0,t_example/dt/2))
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_position(('data', -200))
axs[0].spines['bottom'].set_position(('data', -1))

im1 = axs[1].imshow(err_dark*1000, interpolation='nearest', aspect='auto', cmap = colormap, vmax = vmax_dark, vmin = vmin_dark, origin='lower')
axs[1].scatter(range(n_th)[0:plt_range],bump_pos_dark/int(360/n_neu), color='black', s = .000001, alpha=.5)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('HD')
axs[1].set_ylim((0,n_neu))
plt.xticks(xi_err,x_err)
plt.yticks(ticks_HD_less,t_ticks_less)
axs[1].yaxis.set_minor_locator(MultipleLocator(15))
axs[1].set_xlim((0,t_example/dt/2))
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_position(('data', -200))
axs[1].spines['bottom'].set_position(('data', -1))
fig.tight_layout()
# Colorbars
ax_bar0 = fig.add_axes([.14,axs[0].get_position().y0,1,axs[0].get_position().y1-axs[0].get_position().y0])
fig.colorbar(im0)
plt.axis('off')
ax_bar1 = fig.add_axes([.14,axs[1].get_position().y0,1,axs[1].get_position().y1-axs[1].get_position().y0])
fig.colorbar(im1)
plt.axis('off')
# Text
axs[0].plot([],ls='-',color=light_col,linewidth=2,alpha=light_alpha)
axs[0].plot([],ls='-',color=dark_col,linewidth=2)
axs[0].plot([],ls='-',color='black',linewidth=.4)
fig.legend(['Light','Dark','PVA'],loc='upper center', bbox_to_anchor=(0.55, 1.05), \
           ncol=3,markerscale=10,frameon=False)
fig.text(0, (axs[1].get_position().y1+axs[0].get_position().y0)/2, 
         'Heading ($\degree$)', ha='center', va='center', rotation='vertical')
fig.text(1.07, (axs[1].get_position().y1+axs[0].get_position().y0)/2, 
         'Learning error (spikes/s)', ha='center', va='center', rotation='vertical')
# Light overbars
ax_cond0 = fig.add_axes([0,axs[0].get_position().y1,1,.1])
plt.axis('off')
ax_cond0.axhline(y=.3, xmin=axs[0].get_position().x0, xmax=axs[0].get_position().x1, color=light_col, linewidth = 2, alpha = light_alpha)
ax_cond1 = fig.add_axes([0,axs[1].get_position().y1,1,.1])
plt.axis('off')
ax_cond1.axhline(y=.3, xmin=axs[1].get_position().x0, xmax=axs[1].get_position().x1, color=dark_col, linewidth = 2)

# Have a look at the bump and the visual input
t = 11.5
t_idx = int(t/dt)
f_inst = f_light[:,t_idx]*1000
err_inst = err_light[:,t_idx]*1000
th = np.linspace(0,360,int(n_neu))
th0 = theta0[t_idx] % 360
r = util.vis_in(th/180*np.pi,th0/180*np.pi,params['M'],params['sigma'],-params['exc']-1,day=True)
f_r = util.logistic(r+params['exc'])*1000

fig, ax = plt.subplots(figsize=(3,2))
plt.plot(th-180,f_inst,c='green',linewidth=2)
plt.plot(th-180,f_r,c='dodgerblue',linewidth=2)
plt.plot(th-180,err_inst,c='darkorange',linewidth=2)
plt.ylabel('Firing rate (spikes/s)')
plt.xlabel('Heading ($\degree$)')
plt.title('t = {} s'.format(t),fontsize=MEDIUM_SIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -190))
ax.spines['bottom'].set_position(('data', -10))
ax.xaxis.set_major_locator(MultipleLocator(90))
ax.yaxis.set_major_locator(MultipleLocator(75))
ax.yaxis.set_minor_locator(MultipleLocator(37.5))
plt.xlim([-180,180])
plt.legend(['Bump','Visual bump','Learning error'],loc='upper right',bbox_to_anchor=(1.2, 1.03), frameon=False)


# Velocity gain plot

if vel_gain:
    
    # Remove excitatory sidelobes
    if cut_exc:
        for i in range(n_neu):
            idx1 = int((i+n_pos)/2 - offset) % n_pos
            idx2 = int((i+n_pos)/2 + width - offset) % n_pos
            if idx1<idx2:
                w[i,idx1:idx2] = 0
            else:
                w[i,idx1:n_pos] = 0
                w[i,0:idx2] = 0
            idx3 = int((i+n_pos)/2 - width + offset + 1) % n_pos + n_pos
            idx4 = int((i+n_pos)/2 + offset + 1) % n_pos + n_pos
            if idx3<idx4:
                w[i,idx3:idx4] = 0
            else:
                w[i,idx3:n_neu] = 0
                w[i,n_pos:idx4] = 0
        
        # Plot W^HR profiles
        temp = copy.deepcopy(w[:,0:n_neu])
        # Average weights over neurons
        for j in range(n_neu):    
            temp[:,j] = np.roll(temp[:,j],int(n_neu/2-2*j))
        l_hr_slice = np.mean(temp[:,0:int(n_neu/2)],axis=1)
        r_hr_slice = np.mean(temp[:,int(n_neu/2):n_neu],axis=1)        
        
        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(deg_HD,l_hr_slice,color='green',linewidth=3,zorder=1)
        ax.plot(deg_HD,r_hr_slice,color='darkorange',linewidth=3,zorder=1)
        ax.plot(deg_HD,w_l_hr_hist[:,-1],color='green',linewidth=3,linestyle='dashed',zorder=0)
        ax.plot(deg_HD,w_r_hr_hist[:,-1],color='darkorange',linewidth=3,linestyle='dashed',zorder=0)
        plt.legend(['L-HR','R-HR'],prop={'size': SMALL_SIZE},frameon=False,ncol=1,bbox_to_anchor=(1.1, 1.2))
        ax.set_title('$W^{HR}$')
        ax.set_ylabel('Synaptic strength')
        ax.set_xlabel('Receptive field difference ($\degree$)')
        plt.subplots_adjust(hspace=0.5)
        ax.set_xticks([-180,-90+dphi/4,0+dphi/2,90+dphi/4,180])
        ax.set_xticklabels([-180,-90,0,90,180])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -190))
        ax.spines['bottom'].set_position(('data', -27))
        ax.set_xlim([-180,180])
        plt.show()
        
    if 'neu_v' not in stored or cut_exc:
        # Run velocity gain simulation if data is not already available
        hd_v, neu_v_day, neu_v_dark = util.vel_gain(w,w_rot,t_each=5,
                v_max=params['v_max'],inv=inv,bump_size=bump_size,params=params,start=180)
    
    ticks = np.linspace(int(min(hd_v)),int(max(hd_v)),5).astype(np.int)
    
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    ax.plot(hd_v,neu_v_day,linewidth=2,color=light_col,alpha=light_alpha,label='Light')
    ax.plot(hd_v,neu_v_dark,linewidth=2,color=dark_col,label='Dark')
    ax.plot(hd_v,params['gain']*hd_v,linestyle='dotted',linewidth=2,color='gray',label='Gain {}'.format(params['gain']))
    plt.axhline(y=theor_lim, color='b', linestyle='-.',linewidth=1)
    plt.axhline(y=-theor_lim, color='b', linestyle='-.',linewidth=1)
    plt.axhline(y=fly_max, color='green', linestyle='dashed',linewidth=1.5,label='Fly maximum')
    plt.axhline(y=-fly_max, color='green', linestyle='dashed',linewidth=1.5)
    plt.legend(ncol = 1, loc='upper center', bbox_to_anchor=(.9, .65) ,frameon=False)
    plt.ylabel('Neural angular velocity ($\degree$/s)')
    plt.xlabel('Head angular velocity ($\degree$/s)')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim([-params['v_max'],params['v_max']])
    plt.ylim([-params['v_max'],params['v_max']])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', -750))
    ax.spines['bottom'].set_position(('data', -750))
    plt.show()
    
    # Velocity histogram
    
    if vel_hist:
    
        _, v_ang, _ = util.gen_theta0_OU(t_run=40000)
        
        # fit gaussian
        dist = getattr(scipy.stats, 'norm')
        param = dist.fit(v_ang)
        mean = 0; sigma = util.round_multiple(param[-1],5).astype(int)
        x = np.linspace(-params['v_max']*1.2,params['v_max']*1.2,n_hist)
        pdf_fitted = dist.pdf(x, loc=mean, scale=sigma)
        
        fig, ax = plt.subplots(figsize=(3,2))
        plt.plot(x,pdf_fitted,c='darkorange',linewidth=2,linestyle='--')
        _ = plt.hist(v_ang,n_hist,density=True)
        plt.xlabel('Angular velocity ($\degree$/s)')
        plt.ylabel('Probability density')
        plt.xticks(ticks)
        plt.xlim((-params['v_max']*1.2,params['v_max']*1.2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', -params['v_max']*1.25))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
        plt.legend(['$N({},{})$'.format(mean,sigma)],loc='upper right', \
                    bbox_to_anchor=(1.08, 1),frameon=False)
        plt.show()
        
    
# Path integration in darkness error plot
    
if PI_err:
    
    if 'PI_err' not in stored:
        # Store errors for each simulation
        PI_errors = np.zeros((n_PI,len(PI_dur)))
            
        for i in range(n_PI):
                
            print('Path integration in darkness for {} s, simulation {} out of {}'.format(PI_dur[-1],i+1,n_PI))
            theta0_dark, _, _ = util.gen_theta0_OU(PI_dur[-1],bound_vel=True,v_max=500)
            _, _, f, _, _ = rec.simulate(PI_dur[-1],theta0_dark,params,True,False,w,w_rot,day=False)
            bump_pos = util.bump_COM(f,pos)
                
            for j, dur in enumerate(PI_dur):
                # Convert to rads to average and back to degrees
                loc = int(dur/dt)
                bump_pos_end = circmean(bump_pos[(loc-avg):loc] * (np.pi/180)) / (np.pi/180)
                true_pos_end = circmean(-theta0_dark[(loc-avg):loc] * (np.pi/180)) / (np.pi/180)
                
                diff = bump_pos_end - true_pos_end
                
                if diff>180:
                    diff = diff - 360
                elif diff<-180:
                    diff = 360 + diff
                    
                PI_errors[i,j] = diff
                
    # PI error plot
    n_y = 3; n_x = int(len(PI_dur)/n_y)
    bins = np.linspace(-180,180,50)
    fig, axs = plt.subplots(n_y,n_x,sharex=True,sharey=True,figsize=(3.5,2.5))
    
    count = 0
    for j in range(n_x):
        for i in range(n_y):
            axs[i,j].hist(PI_errors[:,count],bins)
            axs[i,j].set_title('t = {} s'.format(PI_dur[count]),fontdict = {'fontsize':SMALL_SIZE})
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['left'].set_position(('data', -190))
            axs[i,j].spines['bottom'].set_position(('data', 0))
            count+=1
    
    plt.xlim([-180,180])
    axs[0,0].xaxis.set_major_locator(MultipleLocator(180))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(90))    
    axs[0,0].yaxis.set_major_locator(MultipleLocator(150))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(75))
    fig.text(0, 0.5, 'Count', ha='center', va='center', rotation='vertical')
    fig.text(0.55,0,'Path integration error ($\degree$)',ha='center', va='center')
    plt.tight_layout()

# Save results
if save:
    np.savez(data_path+filename + '.npz',w=W,w_rec=w_rec_hist[:,-1],
             w_l_hr=w_l_hr_hist[:,-1],w_r_hr=w_r_hr_hist[:,-1],w_rot=w_rot,
             params=params,hd_v=hd_v,neu_v_dark=neu_v_dark,neu_v_day=neu_v_day,
             PI_dur=PI_dur,PI_errors=PI_errors,err=error)