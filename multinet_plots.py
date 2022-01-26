"""
Produces figures that involve more than one networks.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import utilities as util
import os
from pathlib import Path
import scipy.stats as st
import fly_rec as rec
from sklearn.metrics import mean_squared_error
from matplotlib.legend import Legend

# Simulation options
sim_run = "2Enough"
t_run = util.sim_time(sim_run)
# Directories
load_dir = str(Path(os.getcwd()).parent) + '\\savefiles\\'
parallel_dir = load_dir + 'trained_networks\\Parallel\\'
main_net_dir = parallel_dir + 'Main_Net\\'
theor_lim_dir = parallel_dir + 'Theor_Lim\\'
adapt_gain_dir = parallel_dir + 'Adapt_Gain\\'

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

# Error history plot

override_run = np.arange(0,12)
error_hists = np.empty((100,len(override_run)))

for i in override_run:
    params['run'] = i
    filename = 'fly_rec' + util.filename(params)
    data = np.load(main_net_dir+filename+'.npz',allow_pickle=True)
    error = data['err']
    error_hist = np.mean(error,axis = 0)*1000
    error_hists[:,i] = error_hist    

avg_error_hist = np.mean(error_hists,axis=1)

params['run'] = 0
t = np.linspace(0,params['t_run'],100)/3600

fig, ax = plt.subplots(figsize=(3,2))
plt.plot(t,error_hists,color = 'dodgerblue',linewidth=1,alpha=.3)
plt.plot(t,avg_error_hist,color = 'dodgerblue',linewidth=2)
plt.ylabel('Absolute error (spikes/s)')
plt.xlabel('Training time (hours)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.4))
ax.spines['bottom'].set_position(('data', -.2))
plt.xlim([0,t[-1]])
plt.ylim([0,np.max(error_hist)])

# Theoretical velocity limit plot

override_tau_s = np.linspace(100,260,17).astype(int)
avail_run = [0,1,2,3,4,5]
max_abs_vel = np.zeros((len(override_tau_s),len(avail_run)))
max_abs_vel_mean = np.zeros(len(override_tau_s))
max_abs_vel_025 = np.zeros(len(override_tau_s))
max_abs_vel_975 = np.zeros(len(override_tau_s))

for i, tau_s in enumerate(override_tau_s):
    params['tau_s'] = tau_s
    if tau_s <= 150:
        params['sigma_v'] = 400; params['v_max'] = 1080
    else:
        params['sigma_v'] = 225; params['v_max'] = 720
    for j in avail_run:
        params['run'] = j
        filename = 'fly_rec' + util.filename(params)
        data = np.load(theor_lim_dir+'run'+str(j)+'\\'+filename+'.npz',allow_pickle=True)
        hd_v = data['hd_v']
        neu_v_dark = data['neu_v_dark']
        max_abs_vel[i,j] = util.max_velocity_PI(neu_v_dark,hd_v)
    data = max_abs_vel[i,:]
    max_abs_vel_mean[i] = np.mean(data)
    (max_abs_vel_025[i], max_abs_vel_975[i]) = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))

params['tau_s'] = 65
params['run'] = 0
override_tau_s_fine = np.linspace(override_tau_s[0],override_tau_s[-1],100)

# fit line to points
coeff_ln = np.polyfit(override_tau_s,max_abs_vel_mean,1)
coeff_inv,_,_,_ = np.linalg.lstsq(1/override_tau_s[:,np.newaxis],max_abs_vel_mean)
line = coeff_ln[0]*override_tau_s_fine + coeff_ln[1]
inv = coeff_inv/override_tau_s_fine

fig, ax = plt.subplots(figsize=(3.5,2.5))
plt.plot(override_tau_s_fine,inv,color='b', linestyle='-.',linewidth=.75,label='Neural Velocity Limit')
plt.scatter(override_tau_s,max_abs_vel_mean,color = 'green',s=10)
plt.errorbar(override_tau_s,max_abs_vel_mean,[max_abs_vel_mean-max_abs_vel_025,max_abs_vel_975-max_abs_vel_mean],color = 'green',linestyle='')
plt.ylabel('Neural angular velocity ($\degree$/s)')
plt.xlabel('$τ_{s}$ (ms)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', override_tau_s[0]-15))
ax.spines['bottom'].set_position(('data', 240))
plt.xlim([override_tau_s[0]-10,override_tau_s[-1]+5])
plt.ylim([250,750])
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(100))
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1),markerscale=10,frameon=False)


# Plot that shows what happens at the theoretical limit

params['tau_s'] = 190
filename = 'fly_rec' + util.filename(params)
data = np.load(theor_lim_dir+'run0'+'\\'+filename+'.npz',allow_pickle=True)
w = data['w'][:,:,-1]
try:
    w_rot = data['w_rot']
except:
    w_rot = None

# Simulate different velocities in darkness and light
vels = [390,420,450]
t_each = 2
n_t = int(t_each/params['dt'])
f_day = np.empty((len(vels),params['n_neu'],n_t))
f_dark = np.empty((len(vels),params['n_neu'],n_t))
theta0 = np.empty((len(vels),n_t))

for i,vel in enumerate(vels):
    theta0[i,:] = np.linspace(180,180-t_each*vel,n_t)
    _,_,f_day[i,:],_,_ = rec.simulate(t_each,theta0[i,:],params,True,False,w,w_rot,True)
    _,_,f_dark[i,:],_,_ = rec.simulate(t_each,theta0[i,:],params,True,False,w,w_rot,False)
    
# Plot
f_min = 0; f_max = 150
true_col = 'limegreen'
true_sz = .01
light_col = 'gold'
light_alpha = .6
dark_col = 'darkslateblue'
fr_col = 'Greys'
ticks_HD = np.linspace(0,params['n_neu'],3)
t_ticks = [-180,0,180]
xi = np.linspace(0,t_each/params['dt'],3)
x = np.linspace(0,t_each,3).astype(np.int)
theta = params['gain'] * theta0 % 360; theta = 360 - theta
theta_f = theta * params['n_neu'] / 360

fig, axs = plt.subplots(2,len(vels),sharex=True,sharey=True,figsize = (6.5,3.25))

for i in range(2):
    for j, vel in enumerate(vels):
        if i==0:
            im = axs[i,j].imshow(f_day[j,:]*1000, interpolation='nearest', aspect='auto', cmap = fr_col, vmax = f_max, vmin = f_min, origin='lower')
        else:
            im = axs[i,j].imshow(f_dark[j,:]*1000, interpolation='nearest', aspect='auto', cmap = fr_col, vmax = f_max, vmin = f_min, origin='lower')
        axs[i,j].scatter(range(n_t),theta_f[j,:], color=true_col, s = true_sz,label='_')
        axs[i,j].spines['top'].set_visible(False)
        axs[i,j].spines['right'].set_visible(False)
        axs[i,j].spines['left'].set_position(('data', -15))
        axs[i,j].spines['bottom'].set_position(('data', -1))
        if i==0:
            axs[i,j].set_title('v = {} $\degree$/s'.format(vel),fontsize=MEDIUM_SIZE,y=1.2)
        if j==0:
            axs[i,j].set_ylabel('HD')
        
plt.xlim((0,t_each/params['dt']))
plt.xticks(xi,x)
plt.ylim((0,params['n_neu']))
plt.yticks(ticks_HD,t_ticks)
axs[0,0].yaxis.set_minor_locator(MultipleLocator(15))  
fig.text(0, 0.5, 'Heading ($\degree$)', ha='center', va='center', rotation='vertical')
fig.text(0.55,0,'Time (s)',ha='center', va='center')
plt.tight_layout()

# Condition overbars
axs_cond = []; counter = 0
for i in range(2):
    for j, vel in enumerate(vels):
        ax_pos = axs[i,j].get_position()
        axs_cond.append(fig.add_axes([0,ax_pos.y1,1,.1]))
        plt.axis('off')
        if i==0:
            axs_cond[counter].axhline(y=.3, xmin=axs[i,j].get_position().x0, xmax=axs[i,j].get_position().x1, color=light_col, linewidth = 2, alpha = light_alpha)
        else:
            axs_cond[counter].axhline(y=.3, xmin=axs[i,j].get_position().x0, xmax=axs[i,j].get_position().x1, color=dark_col, linewidth = 2)
        counter += 1 
        
# Colorbar
ax_bar = fig.add_axes([.1,axs[1,2].get_position().y0,1.05,axs[0,2].get_position().y1-axs[1,2].get_position().y0])
fig.colorbar(im,label='Firing rate (spikes/s)', ticks = [round(f_min),round(f_max/3),round(2*f_max/3),round(f_max)])
plt.axis('off')
# legend
axs[0,1].plot([],ls='-',color=light_col,linewidth=2,alpha=light_alpha,label='Light')
axs[0,1].plot([],ls='-',color=dark_col,linewidth=2,label='Dark')
axs[0,1].plot([],ls='-',color=true_col,linewidth=.75,label='True heading')
fig.legend(loc='upper center', bbox_to_anchor=(0.6, 1.1),ncol=3,markerscale=10,frameon=False)

# Velocity Gain plot
neu_v_dark = data['neu_v_dark']
neu_v_day = data['neu_v_day']
ticks = np.linspace(int(min(hd_v)),int(max(hd_v)),5).astype(np.int)
theor_lim = coeff_inv/(params['tau_s']+10)

fig, ax = plt.subplots(figsize=(3,2))
ax.plot(hd_v,neu_v_day,linewidth=2,color=light_col,alpha=light_alpha)
ax.plot(hd_v,neu_v_dark,linewidth=2,color=dark_col)
ax.plot(hd_v,params['gain']*hd_v,linestyle='dotted',linewidth=2,color='gray')
plt.axhline(y=500, color='green', linestyle='dashed',linewidth=1.5)
plt.axhline(y=theor_lim, color='b', linestyle='-.',linewidth=.75)
plt.axhline(y=-500, color='green', linestyle='dashed',linewidth=1.5)
plt.axhline(y=-theor_lim, color='b', linestyle='-.',linewidth=.75)
plt.legend(['Light','Dark','Gain {}'.format(params['gain']),'Fly maximum','Neural velocity limit'], \
            ncol = 2, loc='upper center', bbox_to_anchor=(.75, 1.3) ,frameon=False,columnspacing=0.2)
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

# Adapt gain plots
params['tau_s'] = 65
gains = [0.25,0.5,1,1.5,2]
blue_colors = util.get_sat_col('dodgerblue')
green_colors = ['limegreen','green','darkgreen']
orange_colors = ['gold','darkorange','saddlebrown']
purple_colors = util.get_sat_col(dark_col)
grey_colors = ['darkgrey','grey','black']
weight_plot = [0,2,4]
n_net = len(gains)
neu_v_dark = np.empty((n_net,len(hd_v)))
w_rec = np.empty((n_net,params['n_neu']))
w_l_hr = np.empty((n_net,params['n_neu']))
w_r_hr = np.empty((n_net,params['n_neu']))
avg_error_hist = np.empty((n_net,100))

sim_run_adapt = '4Medium'
t_run_adapt = util.sim_time(sim_run_adapt)

for i in range(n_net):
    # navigate to correct directory
    params['gain'] = gains[i]
    if i==2:
        curr_dir = main_net_dir
        params['sim_run'] = '2Enough'
        params['t_run'] = util.sim_time(params['sim_run'])
    else:
        params['t_run'] = t_run_adapt
        params['sim_run'] = sim_run_adapt
        curr_dir = adapt_gain_dir + 'Gain' + str(gains[i]) + '\\'
    error_hists_temp = np.empty((len(override_run),100))
    for j in override_run:
        params['run'] = j
        filename = 'fly_rec' + util.filename(params)
        data = np.load(curr_dir+filename+'.npz',allow_pickle=True)
        
        # get connectivity and gain plot from run0
        if j==0:
            w_rec[i,:] = data['w_rec']
            w_l_hr[i,:] = data['w_l_hr']
            w_r_hr[i,:] = data['w_r_hr']
            neu_v_dark[i,:] = data['neu_v_dark']
        error = data['err']
        error_hist = np.mean(error,axis = 0)*1000
        error_hists_temp[j,:] = error_hist
        
    avg_error_hist[i,:] = np.mean(error_hists_temp, axis = 0)
    
# Plots

# Error history plot
t = np.linspace(0,params['t_run'],100)/3600
fig,ax = plt.subplots(figsize=(3,2))
for i in range(n_net):
    if i==2:
        continue
    ax.plot(t,avg_error_hist[i,:],linewidth=1,color = blue_colors[i],label='$g={}$'.format(gains[i]))
    
plt.ylabel('Absolute error (spikes/s)')
plt.xlabel('Training time (hours)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.1))
ax.spines['bottom'].set_position(('data', -.1))
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=2,bbox_to_anchor=(1, 1.1))
plt.xlim([0,t[-1]])
plt.ylim([0,np.max(error_hist)])

# Velocity gain plot
ticks = np.linspace(int(min(hd_v)),int(max(hd_v)),5).astype(np.int)
fig, ax = plt.subplots(figsize=(3,2))

for i in reversed(range(n_net)):
    ax.plot(hd_v,gains[i]*hd_v,linewidth=.5,color='gray')
    ax.plot(hd_v,neu_v_dark[i,:],linewidth=1.25,color=purple_colors[i],
            label='{}'.format(gains[i]))

plt.ylabel('Neural angular velocity ($\degree$/s)')
plt.xlabel('Head angular velocity ($\degree$/s)')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim([-params['v_max'],params['v_max']])
plt.ylim([-900,900])
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -params['v_max']*1.05))
ax.spines['bottom'].set_position(('data', -950))
plt.legend(title = 'Gain $g$', ncol = 1, fontsize = SMALL_SIZE,
           title_fontsize = SMALL_SIZE,loc='upper center', bbox_to_anchor=(1.4, 1.2) ,frameon=False)

# Weight profile plots
deg_HR = np.linspace(-180,180,int(params['n_neu']/2))
deg_HD = np.linspace(-180,180,params['n_neu'])
dphi = 720/params['n_neu']

fig, ax = plt.subplots(figsize=(3,2))
for i in weight_plot:
    ax.plot(deg_HD,w_rec[i,:],color=blue_colors[i],linewidth=1,label='${}$'.format(gains[i]))
ax.set_title('$W^{rec}$')
ax.set_ylabel('Synaptic strength')
ax.set_xlabel('Receptive field difference ($\degree$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -190))
ax.spines['bottom'].set_position(('data', -12))
ax.set_xlim([-180,180])
ax.set_xticks([-180,-90+dphi/4,0+dphi/2,90+dphi/4,180])
ax.set_xticklabels([-180,-90,0,90,180])
plt.legend(title = 'Gain $g$', ncol = 1, fontsize = SMALL_SIZE,
           title_fontsize = SMALL_SIZE,loc='upper center', bbox_to_anchor=(.9, 1.1) ,frameon=False)

fig, ax = plt.subplots(figsize=(3,2))
for j,i in enumerate(weight_plot):
    ax.plot(deg_HD,w_l_hr[i,:],color=green_colors[j],linewidth=1,label='${}$'.format(gains[i]))
    ax.plot(deg_HD,w_r_hr[i,:],color=orange_colors[j],linewidth=1,label='${}$'.format(gains[i]))
handles, labels = plt.gca().get_legend_handles_labels()
even = [0,2,4]; odd = [1,3,5]
plt.legend([handles[idx] for idx in even],[labels[idx] for idx in even],
           title='$g$ (L-HR)',fontsize=SMALL_SIZE,title_fontsize=SMALL_SIZE,frameon=False,ncol=1,bbox_to_anchor=(.32, .5))
leg = Legend(ax, [handles[idx] for idx in odd],[labels[idx] for idx in odd],
      title='$g$ (R-HR)',fontsize=SMALL_SIZE,title_fontsize=SMALL_SIZE,frameon=False,ncol=1,bbox_to_anchor=(.75, .5))
ax.add_artist(leg);
ax.set_title('$W^{HR}$')
ax.set_ylabel('Synaptic strength')
ax.set_xlabel('Receptive field difference ($\degree$)')
plt.subplots_adjust(hspace=0.5)
ax.set_xticks([-180,-90+dphi/4,0+dphi/2,90+dphi/4,180])
ax.set_xticklabels([-180,-90,0,90,180])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -190))
ax.spines['bottom'].set_position(('data', -32))
ax.set_xlim([-180,180])


# Diffusion coefficients plot

data = np.load(load_dir + 'Diff1000tr.npz'); D = data['D'][:,0]; sigma = data['n_levels']
data = np.load(load_dir + 'Diff1000trNoise.npz'); D_n = data['D'][:,0]

PI_dur = np.arange(10,70,10)
PIerr = np.load(load_dir + 'PIerr.npy'); D_PI = np.mean(np.divide(np.nanvar(PIerr,axis=0),PI_dur))
PIerrNoise = np.load(load_dir + 'PIerrNoise.npy'); D_PI_n = np.mean(np.divide(np.nanvar(PIerrNoise,axis=0),PI_dur))

offset = 0

fig, ax = plt.subplots(figsize=(3.5,2.5))
plt.scatter(sigma-offset,D,color='dodgerblue',s=10,label='Train noise $\sigma_n = 0$')
plt.scatter(sigma+offset,D_n,color='darkorange',s=10,label='Train noise $\sigma_n = 0.7$')
plt.scatter([],[],color='black',s=10,marker="^",label='PI errors included')
plt.scatter(0,D_PI,color='dodgerblue',s=10,marker="^")
plt.scatter(0.7,D_PI_n,color='darkorange',s=10,marker="^")
plt.yscale('log')
plt.ylabel('Diffusion coefficient ($deg^2$/s)')
plt.xlabel('Test noise $\sigma_n$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.05))
plt.xlim(-.03,1.03)
ax.xaxis.set_major_locator(MultipleLocator(.2))
ax.xaxis.set_minor_locator(MultipleLocator(.1)) 
plt.legend(loc='lower right',markerscale=1,frameon=False)

# PI performance for different gains plot

gains = [1/4, 1/2, 3/4, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5]
params['run'] = 1
params['sim_run'] = '2Enough'
theor_lim = coeff_inv/(params['tau_s']+10)
fly_max = 500

nrmse = np.zeros(len(gains))

curr_dir = adapt_gain_dir + 'Broad\\'

for i, gain in enumerate(gains):
    
    params['gain'] = gain
    filename = 'fly_rec' + util.filename(params)
    data = np.load(curr_dir+filename+'.npz',allow_pickle=True)
    hd_v = data['hd_v']; neu_v_dark = data['neu_v_dark']
    
    maxvel = np.min([fly_max,theor_lim/np.abs(params['gain'])])
    n_maxvel = np.argmin(np.abs(hd_v-maxvel))
    
    nrmse[i] = np.sqrt(mean_squared_error(params['gain']*hd_v[len(hd_v)-n_maxvel:n_maxvel],
                       neu_v_dark[len(hd_v)-n_maxvel:n_maxvel]))/(params['gain']*hd_v[n_maxvel])

fig, ax = plt.subplots(figsize=(3,2))
plt.scatter(gains,nrmse,color = 'green',s=10)
plt.ylabel('NRMSE')
plt.xlabel('Gain $g$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.1))
ax.spines['bottom'].set_position(('data', -.01))
plt.ylim([0,.17])
plt.xlim([0,4.7])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(.5)) 
ax.yaxis.set_major_locator(MultipleLocator(.05))
ax.yaxis.set_minor_locator(MultipleLocator(.025)) 
plt.legend(loc='lower right',markerscale=1,frameon=False)