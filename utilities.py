"""
Various utilities.
"""

import numpy as np
import fly_rec as rec
import matplotlib.colors as mc
import colorsys

# Visual input

def vis_in(theta,theta0,M,sigma,r0,day):
    if day:
        return M*np.exp(-(np.sin((theta+theta0)/2))**2/(2*sigma**2)) + r0
    else:
        # During darkness, the visual input is zero
        return 0
    
# Activation function

def logistic(x,x0=1,b=2.5,s=.15):
    # s: maximum firing rate in kHz
    # x0: 50 % firing rate point
    # b: steepness of gain
    
    return s/(1+np.exp(-b*(x-x0)))


# Generates training trials via an Ornstein-Uhlenbeck process for angular velocity

def gen_theta0_OU(t_run,sigma=225,tau=.5,bound_vel=False,v_max=720):
    # constants
    dt = 5*10**(-4) # euler integration step size
    
    # Initialization
    time = np.linspace(0,t_run,int(t_run/dt))
    theta0 = 180*np.ones(time.size)
    
    # Generate velocities from Ornstein-Uhlenbeck process
    v_ang = Ornstein_Uhlenbeck(0,dt,time.size,sigma,tau)
    # restrict angular velocity
    if bound_vel:
        v_ang[np.abs(v_ang)>v_max] = np.sign(v_ang[np.abs(v_ang)>v_max])*v_max
    
    # Compute landmark position vector theta0
    for i in range(1,time.size):
        # minus sign since theta0 is not head but landmark position
        theta0[i] = theta0[i-1] - v_ang[i]*dt
    
    return theta0, v_ang, time


# Ornstein-Uhlenbeck process to generate velocities that decay to zero

def Ornstein_Uhlenbeck(v0,dt,num,sigma,tau):
    # constants
    sigma_bis = sigma*np.sqrt(2./tau)
    sqrtdt = np.sqrt(dt)
    alpha = dt/tau
    
    # initialization
    v = np.zeros(num); v[0] = v0
    n = np.random.randn(num-1)
    
    # OU process
    for i in range(num-1):
        v[i+1] = (1-alpha)*v[i] + sigma_bis*sqrtdt*n[i]
    
    return v


# Computes neural velocity for given angular velocity range

def vel_gain(w,w_rot,t_each,bump_size,params=0,dv=5,
             v_max=720,inv=False,start=180):
    
    # Initialize
    dt = 5*10**(-4) # euler integration step size
    hd_v = np.linspace(-v_max,v_max,int(2*v_max/dv)+1) # array of angular velocities, in deg/s
    neu_v_day = np.zeros(hd_v.size)
    neu_v_dark = np.zeros(hd_v.size)
    n_neu = len(w)
    
    # Adjust to work for inverted direction of movement
    if inv:
        sign = -1
    else:
        sign = 1
    
    # Define numbers of neurons and angular positions of neurons
    n_ring = n_neu
    n_pos = int(n_ring/2)
    pos = 360*np.linspace(0,n_pos-1,n_pos)/n_pos
    pos = np.repeat(pos,2)
    dphi = 360/n_ring
    
    # Run simulations of one angular velocity at a time
    
    for i in range(hd_v.size):
        print('Evaluating PI performance for {} deg/s'.format(round(hd_v[i],2)))
        # Location of landmark
        theta0 = np.linspace(start,start-t_each*hd_v[i],int(t_each/dt))
        # Light conditions
        w,w_rot,f_day,f_rot_day,err = rec.simulate(t_each,theta0,params,True,False,w,w_rot,True)
        # Darkness conditions
        w,w_rot,f_dark,f_rot_dark,err = rec.simulate(t_each,theta0,params,True,False,w,w_rot,False)
            
        # find where the bump is by estimating the population vector average (PVA)
        bump_day = bump_COM(f_day,pos)
        bump_day[bump_day<0] = 360 + bump_day[bump_day<0]
        bump_day = bump_day/dphi
        bump_dark = bump_COM(f_dark,pos)
        bump_dark[bump_dark<0] = 360 + bump_dark[bump_dark<0]
        bump_dark = bump_dark/dphi
            
        # compute bump velocity
        diff_day = np.diff(bump_day) 
        v_day = diff_day*dphi/dt
        diff_dark = np.diff(bump_dark)
        v_dark = diff_dark*dphi/dt
        # direction of movement
        mv_dir = sign*np.sign(hd_v[i])
        
        # correct for 360 phase transitions when the bump does of a full circle
        temp1 = (mv_dir*n_ring+diff_day)*dphi/dt     # Jumps to side of propagation
        temp2 = (-mv_dir*n_ring+diff_day)*dphi/dt    # Jumps to opposite side (due to noise)
        v_day[mv_dir*diff_day<-bump_size] = temp1[mv_dir*diff_day<-bump_size]
        v_day[mv_dir*diff_day>bump_size] = temp2[mv_dir*diff_day>bump_size]
        
        temp1 = (mv_dir*n_ring+diff_dark)*dphi/dt    # Jumps to side of propagation
        temp2 = (-mv_dir*n_ring+diff_dark)*dphi/dt   # Jumps to opposite side (due to noise)
        v_dark[mv_dir*diff_dark<-bump_size] = temp1[mv_dir*diff_dark<-bump_size]
        v_dark[mv_dir*diff_dark>bump_size] = temp2[mv_dir*diff_dark>bump_size]
        
        # receive average neural velocity for each case
        neu_v_day[i] = np.mean(v_day[10:])
        neu_v_dark[i] = np.mean(v_dark[10:])
        
    return hd_v, neu_v_day, neu_v_dark


# Finds center of mass of bump activity

def bump_COM(f,pos):
    # f: firing rate of cells
    # pos: position that each cell corresponds to, in degrees
    
    pos = pos/180*np.pi
    # transform angle to cartesian coordinate on unitary circle
    pos_z = angle2z(pos)
    # find PVA    
    COM_z = np.dot(f.T,pos_z)
    # angle of PVA is the bump location
    COM = z2angle(COM_z) % 360
    
    return COM

def angle2z(theta):
    return np.exp(1j * theta)

def z2angle(z):
    return np.angle(z,deg = True)


# Returns filename from network parameters

def filename(params):
    
    filename = params['sim_run'] +'v0'+ str(params['v0']) + \
        'inh' + str(abs(params['inh'])).replace(".","") + 'rot'+ \
        str(abs(params['inh_rot'])).replace(".","") + \
        (('taus' + str(round(params['tau_s'],1))) if params['tau_s'] != 65 else '') + \
        (('taud' + str(round(params['tau_d'],1))) if params['tau_d'] != 100 else '') + \
        (('n' + str(params['n_sigma']).replace(".","")) if params['n_sigma'] != 0 else '') + \
        'NoClipOUsigma' + str(round(params['sigma_v'])) + 'tau05NoBound' + \
        'x' + str(params['x0']).replace('.','') + 'k1' + 'b' + str(params['beta']).replace('.','') + \
        (('vmax' + str(params['v_max'])) if params['v_max'] != 720 else '') + \
        (('A' + str(params['M']).replace(".","")) if params['M'] != 4 else '') + \
        (('s' + str(params['sigma']).replace(".","")) if params['sigma'] != .25 else '') + \
        (('exc' + str(params['exc'])) if params['exc'] != 0 else '') + \
        (('N' + str(params['n_neu'])) if params['n_neu'] != 120 else '') + \
        (('gain' + str(params['gain'])) if params['gain'] != 1 else '') + \
        'Init' + 'NoAnneal05' + \
        (('run' + str(params['run'])) if params['run'] else '') + \
        ('VaryWrot' if params['vary_w_rot'] else '') + \
        ('Adj' if params['adj'] else '') + ('RandWrot' if params['rand_w_rot'] else '') + \
        ('NoFilt' if not params['filt'] else '') + \
        (('gL' + str(params['gL']).replace(".","")) if params['gL'] != 1 else '') + \
        (('gD' + str(params['gD']).replace(".","")) if params['gD'] != 2 else '') + \
        (('eta' + str(params['eta']).replace(".","")) if params['eta'] != 5e-2 else '')
        
        
    return filename


# Returns simulation time from word description

def sim_time(t_run):
    
    sim_time =	{
            "Look": 5,
            "Short": 2e2,
            "Medium": 4e3,
            "2Medium": 8e3,
            "4Medium": 1.6e4,
            "Enough": 4e4,
            "2Enough": 8e4,
            "Long": 2e5
            }
    
    return sim_time[t_run]


# Estimate maximum angular velocity that the network integrates smoothly

def max_velocity_PI(v_dark,hd_v,prop=.5):
    
    # Pick preferred direction of PI for networks with high delays
    zero = np.where(hd_v==0)[0][0]
    sign = np.sign(v_dark[np.argmax(np.abs(v_dark))])
    if sign == 1:
        v_dark_pref = v_dark[zero:]
    elif sign == -1:
        v_dark_pref = np.flip(np.abs(v_dark[:zero]))
        
    # Demand that PI velocity remains above prop times the max encountered
    max_v = v_dark_pref[10]
    for i in np.arange(10,len(v_dark_pref)):
        if v_dark_pref[i] >= max_v:
            max_v = v_dark_pref[i]
        elif v_dark_pref[i] < prop*max_v:
            break
        
    return max_v

# Round to nearest multiple
    
def round_multiple(num, divisor):
    rem = num % divisor
    
    if rem>=divisor/2:
        num = num + divisor
        
    return num - rem

# Function to modify color saturation for plotting

def saturation_mult(color, mult=0.5):
    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], np.max(1 - mult * (1 - c[1]),0), c[2])

# Color saturation helper
    
def get_sat_col(color):
    
    colors = [saturation_mult(color,.25),saturation_mult(color,.5),
              saturation_mult(color,.75),saturation_mult(color,1),
              saturation_mult(color,1.25)]
    
    return colors