"""
Simulate development of a recurrent circuit model of the Drosophila central
complex for angular path integration.
"""

# imports
import numpy as np
import utilities as util

# Main simulation function

def simulate(t_run,theta0,params,store_f=False,train=True,
             w=None,w_rot=None,day=True,stab=False):
    
    # Get parameters from dictionary
    dt = params['dt']                       # euler integration step size
    dt_ms = dt*1e03                         # step size in ms
    n_neu = params['n_neu']                 # number of associative neurons
    n_rot = n_neu                           # number of rotation neurons
    v0 = params['v0']                       # vestibular input offset
    v_max = params['v_max']                 # maximum angular velocity
    M = params['M']                         # visual receptive field magnitude
    sigma = params['sigma']                 # visual receptive field width
    k = v0/v_max                            # vestibular input angular gain
    inh = params['inh']                     # global inhibitory input to associative neurons
    inh_rot = params['inh_rot']             # global inhibitory input to rotation neurons
    every_perc = params['every_perc']       # store value and show update this often
    avg_err = params['avg_err']             # segment is s in which to compute current error of network
    n_sigma = params['n_sigma']             # input noise standard deviation
    exc = params['exc']                     # global excitation to soma of HD cells
    tau_s = params['tau_s']                 # synaptic delays in the network
    gain = params['gain']                   # neural velocity gain to entrain
    vary_w_rot = params['vary_w_rot']       # whether to add variability in the HD to HR connections
    adj = params['adj']                     # whether HD neurons project to adjacent HR neurons
    rand_w_rot = params['rand_w_rot']       # whether to randomly generate the HD to HR connections
    filt = params['filt']                   # whether to filter the learning dynamics
    tau_d = params['tau_d']                 # synaptic plasticity time constant
    x0 = params['x0']                       # input level for 50% of activation function
    beta = params['beta']                   # steepness of activation function
    gD = params['gD']                       # coupling conductance
    gL = params['gL']                       # leak conductance
    fmax = params['fmax']                   # maximum firing rate in kHz
    eta = params['eta']                     # learning rate
    
    bump_size = int(n_neu/12)
    
    # Shift baseline of visual input to make it purely inhibitory 
    r0 = - exc - 1
    
    # Initialize variables
    time = np.linspace(0,t_run,int(t_run/dt))
    # Convert all angles to radians
    n_dir = int(n_neu/2)
    theta_temp = 2*np.pi*np.linspace(0,n_dir-1,n_dir)/n_dir
    theta = np.zeros(n_neu); theta[::2] = theta_temp; theta[1::2] = theta_temp
    # Random initialization of firing rates
    f_old = np.random.uniform(0,.15,n_neu)
    f_old_rot = np.random.uniform(0,.15,n_rot)
    batch_size = int(every_perc/100*time.size)
    samples_perc = int(100/every_perc)
    
    if store_f:
        # Store entire simulation
        f = np.zeros((n_neu,time.size)); f[:,0] = f_old
        f_rot = np.zeros((n_rot,time.size)); f_rot[:,0] = f_old_rot
        err = np.zeros((n_neu,time.size))
    else:
        # Store history of average error
        f = 0; f_rot = 0;
        samples_st = int(avg_err/dt)
        # Store all errors in a time window
        err = np.zeros((n_neu,samples_st))
        # Stores average error history
        error_hist = np.zeros((n_neu,samples_perc))
        curr = -1
    
    # Angular velocity input
    v_ang = -np.diff(theta0)/dt
    theta0 = gain*theta0
    theta0_rad = theta0/180*np.pi
    sign = np.ones(n_rot)
    sign[int(n_rot/2):n_rot] = -1
    
    # Initialize weights
    w_his = np.zeros((n_neu,n_rot+n_neu,samples_perc))
    if w is None:
        N = n_rot + n_neu
        # initialization with Gaussian (0,1/N)
        w = np.random.normal(0,np.sqrt(1/N),(n_neu,N))
    
    # Hardwired HD to HR connections
    if w_rot is None:
        w_rot_ampl = 2/fmax
        if vary_w_rot:
            # Variable yet local weights
            mult_l = 1/2
            w_rot_diag = np.random.uniform(low=mult_l*w_rot_ampl,high=w_rot_ampl,size=n_neu)
            w_rot = np.diag(w_rot_diag)
            if adj:
                # Also project to adjacent HR neurons in PB
                mult_h = 3/4
                w_rot_diag_up = np.random.uniform(low=0,high=mult_h*w_rot_ampl,size=n_neu-1)
                w_rot_diag_lo = np.random.uniform(low=0,high=mult_h*w_rot_ampl,size=n_neu-1)
                w_rot += np.diag(w_rot_diag_up,1) + np.diag(w_rot_diag_lo,-1)
                # L1 and R1 are next to each other
                w_rot[[int(n_neu/2),0],[int(n_neu/2-1),int(n_neu/2)]] = w_rot[[0,int(n_neu/2)],[int(n_neu/2),int(n_neu/2-1)]]
                w_rot[[int(n_neu/2-1),int(n_neu/2)],[int(n_neu/2),0]] = w_rot[[int(n_neu/2),int(n_neu/2-1)],[0,int(n_neu/2)]]
                # R-HR neurons have a clockwise progression
                w_rot[int(n_neu/2),[int(n_neu/2+1),-1]] = w_rot[int(n_neu/2),[-1,int(n_neu/2+1)]]
                w_rot[[int(n_neu/2+1),-1],int(n_neu/2)] = w_rot[[-1,int(n_neu/2+1)],int(n_neu/2)]
        elif rand_w_rot:
            # Completely random weights
            w_rot = np.abs(np.random.normal(scale=w_rot_ampl*np.sqrt(np.pi/2)/(4*bump_size),size=(n_neu,n_neu)))
        else:
            # 1-1 HD to HR connections
            w_rot_diag = w_rot_ampl*np.ones(n_neu)
            w_rot = np.diag(w_rot_diag)
    
    # initialize voltages, currents and weight updates
    u = np.random.uniform(0,1,n_neu); V = np.random.uniform(0,1,n_neu)
    Iden = np.zeros(n_neu); Delta = np.zeros((n_neu,n_rot+n_neu))
    PSP = np.zeros(n_rot+n_neu); I_PSP = np.zeros(n_rot+n_neu); x = np.zeros(n_neu)
    
    # initialize network with a bump of activity
    u = np.zeros(n_neu); V = np.zeros(n_neu);
    diff = 360/n_neu
    peak = n_neu - int(theta0[0]/diff)
    # make sure initialization will happen no matter where the initial position of the bump is
    start = (peak - bump_size) % n_neu; end = peak + bump_size % n_neu
    if start > peak:
        u[start:n_neu] = 100; u[0:end] = 100; V[start:n_neu] = 100; V[0:end] = 100
        f_old_rot = np.zeros(n_rot); f_old_rot[start:n_neu] = .15; f_old_rot[0:end] = .15
        f_old = np.zeros(n_neu); f_old[start:n_neu] = .15; f_old[0:end] = .15
    else:
        u[start:end] = 100; V[start:end] = 100
        f_old_rot = np.zeros(n_rot); f_old_rot[start:end] = .15
        f_old = np.zeros(n_neu); f_old[start:end] = .15
    
    # Check whether light conditions are constant during simulation
    if type(day) == bool:
        DayOrNight = True
    else:
        DayOrNight = False
    
    # Simulate network
    batch_num = 0  # history counter
    
    for i in range(time.size-1):
        # Compute visual and vestibular inputs
        
        # If light conditions change day is a vector specifying current conditions
        if DayOrNight:
            state = day
        else:
            state = day[i]
        
        # Visual input
        r = util.vis_in(theta,theta0_rad[i],M,sigma,r0,state)
            
        # If it is dark discard extra excitatory input, else keep it
        try: 
            if r==0:
                exc_temp = 0
        except:
            exc_temp = exc
        
        # For stability analysis we want to change the position of the bump
        # without a corresponding vestibular input being created (artificial conditions)
        if not stab:
            v = sign*k*v_ang[i]
        else:
            v = 0
        
        # Compute the one step ahead network dynamics
        f_cur,f_cur_rot,u,Iden,V,Delta,w,PSP,I_PSP,x,error = network(f_old,
              f_old_rot,u,w,w_rot,v,r,Iden,V,Delta,PSP,I_PSP,x,train,eta,
              x0,beta,fmax,dt_ms,inh,inh_rot,exc_temp,n_sigma,
              filt,tau_s,tau_d,gD,gL)
        
        # Show progress and store weights
        if (i % batch_size == 0):
            print('{} % of the simulation complete'.format(round(i/time.size*100)))
            if not store_f:
                # Reset buffer of recent error history
                curr = samples_st
            if train:
                w_his[:,:,batch_num] = w
            batch_num += 1
        
        # Store firing rates
        if store_f:
            # store everything
            err[:,i+1] = error
            f[:,i+1] = f_cur
            f_rot[:,i+1] = f_cur_rot
        else:
            if curr > 0:
                # Fill up buffer of recent errors
                err[:,-curr] = error
                curr -= 1
            elif curr == 0:
                # Compute and report average error once buffer is filled 
                err = np.abs(err)
                error_hist[:,batch_num-1] = np.average(err,1)
                print('Average error is {} Hz'.format(round(1000*np.average(err),2)))
                curr -= 1
        
        f_old = f_cur
        f_old_rot = f_cur_rot
    
    # Return error history instead of errors
    if not(store_f):
        err = error_hist
        
    # Store final weights
    if train:
        w_his[:,:,-1] = w
        w = w_his
    
    return w, w_rot, f, f_rot, err


# Network dynamics

def network(f,f_rot,u,w,w_rot,v,r,Iden,V,Delta,PSP,I_PSP,x,train,eta,x0,beta,fmax,
            dt,inh,inh_rot,exc,n_sigma,filt=True,tau_s=65,tau_d=100,gD=2,gL=1,
            tau_l=10):
    # units in ms or ms^-1, C is considered unity and the unit is embedded in g
    
    f_all = np.concatenate((f_rot,f))
    n_hd = np.size(f); n_hr = np.size(f_rot)
    
    # Create noise that will be added to all origins of input
    N = np.random.normal(0,n_sigma,n_hd)
    N_d = np.random.normal(0,n_sigma,n_hd)
    N_HR = np.random.normal(0,n_sigma,n_hr)
    
    # Input current to the dendrites of HD cells
    Iden += (- Iden + np.dot(w,f_all) + inh + N_d) * dt/tau_s
    
    # Dentritic potential is a low-pass filtered version of the dentritic current
    V += (-V+Iden)*dt/tau_l
    
    # Compute PSP for each neuron
    I_PSP += (- I_PSP + f_all) * dt/tau_s
    PSP += (-PSP+I_PSP) * dt/tau_l
    
    # Delayed copy of firing rate of associative neurons as neurotransmitter level
    x += (f-x)/tau_s
    x_l = x[::2]; x_r = x[1::2]
    x_rot = np.concatenate((x_l,x_r))
    
    # firing rate of rotation neurons
    f_new_rot = util.logistic(np.dot(w_rot,x_rot) + v + inh_rot + N_HR,x0,beta,fmax)
    
    # feed visual signal as input to the soma (teacher signal)
    u += (-gL*u + gD*(V-u) + r + N + exc)*dt
    
    # Firing rate of HD neurons
    f_new = util.logistic(u,x0,beta,fmax)
    
    # Dendritic prediction of somatic voltage
    V_ss = V*gD/(gD+gL)
    
    # Error between actual firing rate and dendritic prediction
    error = f_new - util.logistic(V_ss,x0,beta,fmax)
    
    # update the weights
    if train:
        PI = np.outer(error,PSP)
        if filt:
            # Low-pass filter weight change
            Delta += (PI - Delta)*dt/tau_d
        else:
            Delta = PI
        w += eta*Delta*dt
            
    f = f_new
    f_rot = f_new_rot
    
    return f, f_rot, u, Iden, V, Delta, w, PSP, I_PSP, x, error