"""
Simulates reduced network in mathematical appendix, and generates figures.

Author: Tiziano D'Albis
"""


import numpy as np
import pylab as pl
import matplotlib.colors as colors
from scipy.stats import norm
from matplotlib import ticker


pl.rc('font',size=13)


# parameters
f_max=150.
beta=2.5
M=4
I_vis_0=-1
sigma=.15
I_vis_min=I_vis_0
I_vis_max=M+I_vis_0
I_inhib_hd=-1
I_inhib_hr=-1.5
gl=1.
gd=2.
tau_s=0.065
tau_l=0.01
tau_d=0.1
A_active=2.
T=2*np.pi 
dt=0.001
delta_v=0.5

v_sigma=225/180*np.pi
average_speeds=True

rho_hd=60/(2*np.pi)
rho_hr=30/(2*np.pi)

# functions
f= lambda x: f_max/(1+np.exp(-beta*(x-1)))
f_prime = lambda x: beta*f(x)*(1-f(x)/f_max)
vis= lambda t: M*np.exp(-1/(2*sigma**2)*np.sin(t/2.)**2)+I_vis_0
h = lambda t: (np.exp(-t/tau_l)-np.exp(-t/tau_s))/(tau_l-tau_s)
h_d = lambda t: np.exp(-t/tau_d)/tau_d 
h_s = lambda t: np.exp(-t/tau_s)/tau_s 
   


def conv(a,b,dt,T):
    """
    Convolution
    """
    a_dft=np.fft.fft(np.fft.fftshift(a))*dt/T
    b_dft=np.fft.fft(np.fft.fftshift(b))*dt/T
    corr_dft=a_dft*b_dft
    corr=np.fft.fftshift(np.fft.ifft(corr_dft)*T/dt)
    return corr.real

def corr(a,b,dt,T):
    """
    Cross-correlation
    """
    a_dft=np.fft.fft(np.fft.fftshift(a))*dt/T
    b_dft=np.fft.fft(np.fft.fftshift(b))*dt/T
    corr_dft=a_dft.conj()*b_dft
    corr=np.fft.fftshift(np.fft.ifft(corr_dft)*T/dt)
    return corr.real



class SimLearn(object):
    """
    Class to simulate learning of synaptic weights
    """
    
    def __init__(self,paramMap):
        
        for param,value in paramMap.items():
            setattr(self,param,value)
                    
        self.theta_ran=np.arange(-T/2,T/2,dt)
        
        self.title=f'Learning with rotation input, v={self.v_abs}, t={self.num_steps-1}'

                
    def simulate(self):
        
        # angles
        theta_ran=self.theta_ran
    
        # visual input
        vis0=vis(theta_ran)
        
        # recurrent weights
        w=np.zeros_like(theta_ran)
        
        # weight changes
        dw_avg=np.zeros_like(theta_ran)
        dw_r_avg=np.zeros_like(theta_ran)
        dw_l_avg=np.zeros_like(theta_ran)
        
        # rotation weights
        w_r=np.zeros_like(theta_ran)
        w_l=np.zeros_like(theta_ran)
        
        # axo-proximal voltage and rotation-cell activity
        v_a_pos=np.zeros_like(theta_ran)
        rot_r_pos=np.zeros_like(theta_ran)
        rot_l_pos=np.zeros_like(theta_ran)
               
        # vector for evolution plots
        self.w_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.w_r_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.w_l_vect=np.zeros((self.num_steps,len(theta_ran)))

        self.f_va_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.f_vss_pos_vect=np.zeros((self.num_steps,len(theta_ran)))

        self.dw_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.dw_r_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.dw_l_pos_vect=np.zeros((self.num_steps,len(theta_ran)))

        self.p_pos_vect=np.zeros((self.num_steps,len(theta_ran)))        
        self.p_r_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.p_l_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        
        self.rot_r_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        self.rot_l_pos_vect=np.zeros((self.num_steps,len(theta_ran)))
        
   
        if not average_speeds:
            v_steps=[self.v_abs,]
            v_probs=[1,]
            dv=1
        else:
            dv=delta_v
            v_steps=np.arange(dv/2,14,dv)
            v_probs=0.5*(2*norm.pdf(v_steps+dv/2,0,v_sigma)+2*norm.pdf(v_steps-dv/2,0,v_sigma))

            v_steps=v_steps[::-1]
            v_probs=v_probs[::-1]

            va_pos_vel=np.zeros((len(v_steps),len(theta_ran)))
            rot_r_pos_vel=np.zeros((len(v_steps),len(theta_ran)))
            rot_l_pos_vel=np.zeros((len(v_steps),len(theta_ran)))
            

        # simulation loop
        for t_idx in range(self.num_steps):
            
            print(f'\rSimulating time step {t_idx}/{self.num_steps}',end="", flush=True)
                
            # clear dw of previous time step
            dw_avg*=0
            dw_l_avg*=0
            dw_r_avg*=0

            # loop over different speed            
            for v_idx,(v,v_prob) in enumerate(zip(v_steps,v_probs)):
            
                # vestibular input
                I_ves=2*v/(4*np.pi)

                # note that we are mirroring the filters to adehere to the convention of the main text, i.e., clockwise movement means v negative
                h_theta_pos=np.fft.fftshift(h((theta_ran+T/2)/abs(v))[::-1])/abs(v)                
                h_s_theta_pos=np.fft.fftshift(h_s((theta_ran+T/2)/abs(v)))[::-1]/abs(v)
        
                # retrieve firing rates from previous step
                if average_speeds:                
                    v_a_pos=va_pos_vel[v_idx,:]
                    rot_r_pos=rot_r_pos_vel[v_idx,:]
                    rot_l_pos=rot_l_pos_vel[v_idx,:]
                    
                # dendritic input
                d_pos=rho_hd*corr(w,f(v_a_pos),dt,T)*T+rho_hr*corr(w_r,rot_r_pos,dt,T)*T+rho_hr*corr(w_l,rot_l_pos,dt,T)*T+I_inhib_hd
                    
                # steady-state voltage            
                v_ss_pos=gd/(gd+gl)*conv(d_pos,h_theta_pos,dt,T)*T
                
                # axonal voltage            
                v_a_pos=v_ss_pos+vis0/(gd+gl)
    
                # rotation cell firing rate
                rot_r_pos=f(A_active/f_max*conv(h_s_theta_pos,f(v_a_pos),dt,T)*T+I_ves+I_inhib_hr)            
                rot_l_pos=f(A_active/f_max*conv(h_s_theta_pos,f(v_a_pos),dt,T)*T-I_ves+I_inhib_hr)
                            
                # save firing rates for the next step
                if average_speeds:                
                    va_pos_vel[v_idx,:]=v_a_pos
                    rot_r_pos_vel[v_idx,:]=rot_r_pos
                    rot_l_pos_vel[v_idx,:]=rot_l_pos

                # error
                err_pos=f(v_a_pos)-f(v_ss_pos)
    
                # recurrent weights changes
                p_pos=conv(f(v_a_pos),h_theta_pos,dt,T)*T
                dw_pos=self.eta*corr(err_pos,p_pos,dt,T)
                            
                # rotation weights changes
                p_r_pos=conv(rot_r_pos,h_theta_pos,dt,T)*T
                p_l_pos=conv(rot_l_pos,h_theta_pos,dt,T)*T
                dw_r_pos=self.eta*corr(err_pos,p_r_pos,dt,T)
                dw_l_pos=self.eta*corr(err_pos,p_l_pos,dt,T)

                # total weight changes in both directions
                dw=(dw_pos+dw_pos[::-1])  
                dw_r=(dw_r_pos+dw_l_pos[::-1])
                dw_l=dw_r[::-1]      

                # update averge weight changes
                dw_avg+=dw*v_prob*dv
                dw_r_avg+=dw_r*v_prob*dv
                dw_l_avg+=dw_l*v_prob*dv
                
                # save results   
                self.dw_pos_vect[t_idx,:]+=dw_pos*v_prob*dv      
                self.dw_r_pos_vect[t_idx,:]+=dw_r_pos*v_prob*dv      
                self.dw_l_pos_vect[t_idx,:]+=dw_l_pos*v_prob*dv      
    
                self.f_va_pos_vect[t_idx,:]+=f(v_a_pos)*v_prob*dv
                self.f_vss_pos_vect[t_idx,:]+=f(v_ss_pos)*v_prob*dv
                   
                self.p_pos_vect[t_idx,:]+=p_pos*v_prob*dv
                self.p_r_pos_vect[t_idx,:]+=p_r_pos*v_prob*dv
                self.p_l_pos_vect[t_idx,:]+=p_l_pos*v_prob*dv
                
                self.rot_r_pos_vect[t_idx,:]+=rot_r_pos*v_prob*dv
                self.rot_l_pos_vect[t_idx,:]+=rot_l_pos*v_prob*dv
                
               
            # update weights    
            if self.learn_rot_weights:
                w_r+=dw_r_avg
                w_l+=dw_l_avg
            if self.learn_rec_weights:
                w+=dw_avg
                
            # save weights    
            self.w_vect[t_idx,:]=w
            self.w_r_vect[t_idx,:]=w_r
            self.w_l_vect[t_idx,:]=w_l

        
        # compute symmetrized vectors
        self.err_pos_vect=self.f_va_pos_vect-self.f_vss_pos_vect
        self.err_vect=(self.err_pos_vect+self.err_pos_vect[:,::-1])/2
        self.f_va_vect=(self.f_va_pos_vect+self.f_va_pos_vect[:,::-1])/2
        self.f_vss_vect=(self.f_vss_pos_vect+self.f_vss_pos_vect[:,::-1])/2
                
    
    
    
class SimPlots(object):
    """
    Class for plotting results
    """
    def __init__(self,simObject):
        for param,value in simObject.__dict__.items():
            setattr(self,param,value)
        
    def plot_evo(self):
        
        def plot_time_markers():
            pl.axvline(self.t1,color='k',ls=':')
            pl.axvline(self.t3,color='k',ls=':')
        
        stride=2
        strided_steps=np.arange(self.num_steps)[::stride]
        theta_ran_deg=self.theta_ran*180/np.pi

        pl.figure(figsize=(10,10))


        pl.subplots_adjust(hspace=0.3,top=0.95,bottom=0.07)
        
        pl.subplot(611)
        pl.pcolormesh(strided_steps,theta_ran_deg,self.f_va_vect[::stride,:].T,rasterized=True,cmap='binary')
        pl.colorbar()
        pl.yticks([-180,-90,0,90,180])    
        pl.ylabel('Firing rate\n[spikes/s]')
        pl.xlim([0,self.num_steps])
        pl.xticks(np.arange(0,self.num_steps,50),[])
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
        pl.subplot(612)
        divnorm = colors.DivergingNorm(vmin=self.err_vect.min(), vcenter=0, vmax=self.err_vect.max())
        pl.pcolormesh(strided_steps,theta_ran_deg,self.err_vect[::stride,:].T,rasterized=True,cmap='seismic',norm=divnorm)
        cb=pl.colorbar()
        cb.locator=ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        pl.yticks([-180,-90,0,90,180])    
        pl.ylabel('Error \n[spikes/s]')
        pl.xlim([0,self.num_steps])
        pl.xticks(np.arange(0,self.num_steps,50),[])
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
        pl.subplot(613)
        pl.plot(np.arange(self.num_steps),np.abs(self.err_vect).mean(axis=1),'-k')
        cb=pl.colorbar()
        cb.remove()
        pl.xlim([0,self.num_steps])
        pl.xticks(np.arange(0,self.num_steps,50),[])
        pl.ylabel('Average\nabsolute error\n[spikes/s]')
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
    
        pl.subplot(614)
        divnorm = colors.DivergingNorm(vmin=self.w_vect.min(), vcenter=0, vmax=self.w_vect.max())
        pl.pcolormesh(strided_steps,theta_ran_deg,self.w_vect[::stride].T,rasterized=True,cmap='seismic',norm=divnorm)
        cb=pl.colorbar()
        cb.formatter.set_powerlimits((0, 0))
        cb.locator=ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        pl.yticks([-180,-90,0,90,180])    
        pl.ylabel('w')
        pl.xlim([0,self.num_steps])
        pl.xticks(np.arange(0,self.num_steps,50),[])
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
        pl.subplot(615)
        divnorm = colors.DivergingNorm(vmin=self.w_r_vect.min(), vcenter=0, vmax=self.w_r_vect.max())
        pl.pcolormesh(strided_steps,theta_ran_deg,self.w_r_vect[::stride].T,rasterized=True,cmap='seismic',norm=divnorm)
        cb=pl.colorbar()
        cb.formatter.set_powerlimits((0, 0))
        cb.locator=ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        pl.yticks([-180,-90,0,90,180])    
        pl.ylabel('$w^R$')
        pl.xlim([0,self.num_steps])
        pl.xticks(np.arange(0,self.num_steps,50),[])
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
        pl.subplot(616)
        divnorm = colors.DivergingNorm(vmin=self.w_l_vect.min(), vcenter=0, vmax=self.w_l_vect.max())
        pl.pcolormesh(strided_steps,theta_ran_deg,self.w_l_vect[::stride].T,rasterized=True,cmap='seismic',norm=divnorm)
        cb=pl.colorbar()
        cb.locator=ticker.MaxNLocator(nbins=3)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        pl.yticks([-180,-90,0,90,180])    
        pl.xlabel('Time step')
        pl.ylabel('$w^L$')
        pl.xticks(np.arange(0,self.num_steps,50))
        pl.gca().tick_params(length=5)
        plot_time_markers()
        
        
    def plot_rec_profiles(self):
        theta_ran_deg=self.theta_ran*180/np.pi
        
        pl.figure(figsize=(10,10))
        pl.subplots_adjust(left=0.1,right=0.98,hspace=0.5,wspace=0.2,top=0.95,bottom=0.06)
                        
        offset=2
        
        for idx,t in enumerate([self.t1,self.t3]):
            
            pl.subplot(4,2,idx+1,)  
            pl.plot(theta_ran_deg,(self.f_va_pos_vect[t,:]),color='C0',label='f(v$^{a+}$)')
            pl.plot(theta_ran_deg,(self.f_va_pos_vect[t,::-1]),color='C1',label='f(v$^{a-}$)')
            pl.plot(theta_ran_deg,(self.f_vss_pos_vect[t,:]),color='C0',ls='--',label='f(v$^{ss+}$)')
            pl.plot(theta_ran_deg,(self.f_vss_pos_vect[t,::-1]),color='C1',ls='--',label='f(v$^{ss-}$)')
            pl.title(f't={t}')
            pl.xticks([-180,-90,0,90,180])    
            if idx==0:
                pl.ylabel('Firing rate')
                pl.legend(loc=1)
                
            pl.subplot(4,2,offset+idx+1,)              
            pl.plot(theta_ran_deg,self.err_pos_vect[t,:],color='C0',ls='-',label='$\epsilon^+$')
            pl.plot(theta_ran_deg,self.err_pos_vect[t,::-1],color='C1',ls='-',label='$\epsilon^-$')
            pl.plot(theta_ran_deg,self.p_pos_vect[t,:],color='C0',ls='--',label='p$^+$')
            pl.plot(theta_ran_deg,self.p_pos_vect[t,::-1],color='C1',ls='--',label='p$^-$')
            pl.xticks([-180,-90,0,90,180])    
            if idx==0:
                pl.ylabel('Firing rate')
                pl.legend(loc=1)
        
            pl.subplot(4,2,2*offset+idx+1,)  
            pl.plot(theta_ran_deg,self.dw_pos_vect[t,:],label='dw$^+$')
            pl.plot(theta_ran_deg,self.dw_pos_vect[t,::-1],label='dw$^-$')
            pl.plot(theta_ran_deg,(self.dw_pos_vect[t,::-1]+self.dw_pos_vect[t]),color='k',label='dw')
            pl.xticks([-180,-90,0,90,180])    
            pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if idx==0:
                pl.ylabel('Weight change')
                pl.legend(loc=1)
        
            pl.subplot(4,2,3*offset+idx+1,)  
            pl.plot(theta_ran_deg,self.w_vect[t,:],color='k',label='w')
            pl.xticks([-180,-90,0,90,180])    
            pl.xlabel('Head direction [deg]')
            pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if idx==0:
                pl.ylabel('Synaptic weight')
                pl.legend(loc=1)


    def plot_rot_profiles(self):
        
        theta_ran_deg=self.theta_ran*180/np.pi
          
        pl.figure(figsize=(10,10))
        pl.subplots_adjust(left=0.1,right=0.98,hspace=0.5,wspace=0.2,top=0.95,bottom=0.06)
                        
        offset=2
        
        for idx,t in enumerate([self.t1,self.t3]):
            
            pl.subplot(4,2,idx+1,)  
            pl.plot(theta_ran_deg,self.rot_r_pos_vect[t,:],color='C0',label='f(v$^{R+}$)')
            pl.plot(theta_ran_deg,self.rot_l_pos_vect[t,::-1],color='C1',label='f(v$^{R-}$)')
            pl.plot(theta_ran_deg,self.rot_l_pos_vect[t,:],color='C0',ls='--',label='f(v$^{L+}$)')
            pl.plot(theta_ran_deg,self.rot_r_pos_vect[t,::-1],color='C1',ls='--',label='f(v$^{L-}$)')
            pl.title(f't={t}')
            pl.xticks([-180,-90,0,90,180])    
            if idx==0:
                pl.ylabel('Firing rate')
                pl.legend(loc=1)
                
            pl.subplot(4,2,offset+idx+1,)              
            pl.plot(theta_ran_deg,self.err_pos_vect[t,:],color='C0',ls='-',label='$\epsilon^+$')
            pl.plot(theta_ran_deg,self.err_pos_vect[t,::-1],color='C1',ls='-',label='$\epsilon^-$')        
            pl.plot(theta_ran_deg,self.p_r_pos_vect[t,:],color='C0',ls='--',label='p$^R+$')
            pl.plot(theta_ran_deg,self.p_l_pos_vect[t,::-1],color='C1',ls='--',label='p$^R-$')        
            pl.plot(theta_ran_deg,self.p_l_pos_vect[t,:],color='C0',ls=':',label='p$^L+$')
            pl.plot(theta_ran_deg,self.p_r_pos_vect[t,::-1],color='C1',ls=':',label='p$^L-$')            
            pl.xticks([-180,-90,0,90,180])    
            if idx==0:
                pl.ylabel('Firing rate')
                pl.legend(loc=1,fontsize=10)
        
            pl.subplot(4,2,2*offset+idx+1,)          
            pl.plot(theta_ran_deg,self.dw_r_pos_vect[t,:],color='C0',label='dw$^R+$')
            pl.plot(theta_ran_deg,self.dw_l_pos_vect[t,::-1],color='C1',label='dw$^R-$')
            pl.plot(theta_ran_deg,self.dw_l_pos_vect[t,:],color='C0',ls='--',label='dw$^L+$')
            pl.plot(theta_ran_deg,self.dw_r_pos_vect[t,::-1],color='C1',ls='--',label='dw$^L-$')
            pl.xticks([-180,-90,0,90,180])    
            pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if idx==0:
                pl.ylabel('Weight change')
                pl.legend(loc=1)
        
            pl.subplot(4,2,3*offset+idx+1,)  
            pl.plot(theta_ran_deg,self.w_r_vect[t,:],color='k',label='w$^R$')
            pl.plot(theta_ran_deg,self.w_l_vect[t,:],color='k',ls='--',label='w$^L$')            
            pl.xticks([-180,-90,0,90,180])    
            pl.xlabel('Head direction [deg]')
            pl.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if idx==0:
                pl.ylabel('Synaptic weight')
                pl.legend(loc=1)
                
    def plot_filters(self,v):
        
        theta_ran_deg=self.theta_ran*180/np.pi

        h_theta_pos=np.fft.fftshift(h((self.theta_ran+T/2)/abs(v))[::-1])/abs(v)                
        #h_s_theta_pos=np.fft.fftshift(h_s((self.theta_ran+T/2)/abs(v)))[::-1]/abs(v)

        #pl.figure()
        pl.plot(theta_ran_deg,h_theta_pos)       
        #pl.plot(theta_ran_deg,h_s_theta_pos)       
        pl.xticks([-180,-90,0,90,180])    
        pl.xlabel('Head direction [deg]')
        
        
#%%

if __name__ == '__main__':

    # parameters
    params={
            'eta':1e-6,
            'num_steps':401,
            'v_abs':1.7,
            'learn_rec_weights':True,
            'learn_rot_weights':True,
             }
    
    # run sumulation
    res=SimLearn(params)
    res.simulate()        


    # plot results
    p=SimPlots(res)        
    p.t1=25
    p.t2=90
    p.t3=350
            
    p.plot_evo()
    p.plot_rec_profiles()
    p.plot_rot_profiles()


    # plot speed distribution
    v_ran=np.arange(0,14,0.01)
    theta_ran_deg=p.theta_ran*180/np.pi
    v_steps_plot=np.linspace(0.5,13,5)[::-1]
    
    dv=delta_v
    v_steps=np.arange(dv/2,13,dv)
    v_probs=0.5*(2*norm.pdf(v_steps+dv/2,0,v_sigma)+2*norm.pdf(v_steps-dv/2,0,v_sigma))
    
    pl.figure(figsize=(10,4))
    pl.subplots_adjust(left=0.1,wspace=0.3,bottom=0.2)
    
    pl.subplot(121) 
    pl.bar(v_steps*180/np.pi,v_probs,width=dv*180/np.pi,facecolor='lightgray',edgecolor='gray')
    pl.plot(v_ran*180/np.pi,2*norm.pdf(v_ran,0,v_sigma),'-k')
    for idx,v in enumerate(v_steps_plot):
        pl.axvline(v*180/np.pi,color=f'C{idx}')
    pl.xlabel('Head turning speed [deg/s]')
    pl.ylabel('Probability density')
       
    pl.subplot(122)
    for v in v_steps_plot:
        h_theta_pos=(h((p.theta_ran+T/2)/abs(v))[::-1])/abs(v)                
        pl.plot(np.arange(-T,0,dt)*180/np.pi,h_theta_pos/h_theta_pos.max())
    pl.ylabel('Normalized filter $h^+$')
    pl.xlabel('Head direction [deg/s]')