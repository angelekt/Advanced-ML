import numpy as np
import random 
import matplotlib.pyplot as plt
from math import cos, sin, exp, tanh
import math
from tqdm import tqdm_notebook as tqdm


x0 = 0
RandomWalk = []
RandomWalk.append(x0)
u0 = 0
Control = [u0]
Tfinal = np.arange(0.5,2,0.5)
#v = np.concatenate(np.arange([np.asarray([1]),np.arange(5, 100, 5)]))
var = np.asarray([1,10,15,50,100])
#times = np.concatenate(np.asarray([np.arange(0, 2-dt, dt),np.arange(2-dt, 2-dt1, dt1),np.arange(2-dt1, 2, dt2)]))
x_max = 1
x_min = -1

# time step
dt = 0.001

fig1, ax1 = plt.subplots(len(Tfinal),len(var))
fig2, ax2 = plt.subplots(len(Tfinal),len(var))

# loop over the different final objective times 
for i,T in enumerate(Tfinal):
    times = np.arange(dt,T,dt)
    plot_times =np.arange(0,T,dt)
    
    # loop over the different variances
    for j,v in enumerate(var): 
        
        # compute the optimal control (Control) in each time step with Gaussian noise~N(0,v). Save the Random Walk
        for t in times:            
            u_star = (tanh(RandomWalk[-1]/(v*(T-t))) - RandomWalk[-1])/(T-t)
            dxi = np.random.normal(.0,v,size=None)
            x = RandomWalk[-1] + u_star * dt + dxi
            RandomWalk.append(x) 
            Control.append(u_star)
        # end loop of random walk calculation
        
        # plot the results
        
        ax1[i,j].plot(plot_times,RandomWalk,linewidth=0.5, c='blue')
        ax1[i,j].plot(plot_times,np.ones((len(RandomWalk),1)), c='orange')
        ax1[i,j].plot(plot_times,(-1)*np.ones((len(RandomWalk),1)), c='orange')
        ax1[i,j].set_xlabel('Time ' + r'$t$')
        ax1[i,j].set_ylabel('Random Walk ' + r'$x$')
        ax1[i,j].set_ylim([-4,4])
        ax1[i,j].set_title(r'$T$' + '=' + str(Tfinal[i]) +' '  r'$\nu$'+ '='  + str(var[j]))

        ax2[i,j].plot(plot_times,Control,linewidth=0.5, c='blue')
        ax2[i,j].set_xlabel('Time ' + r'$t$')
        ax2[i,j].set_ylabel('Control u*')
        ax2[i,j].set_ylim([-4,4])
        ax2[i,j].set_title(r'$T$' + '=' + str(Tfinal[i]) +' '  r'$\nu$'+ '='  + str(var[j]))  
        
        # initialize for the next loop
        RandomWalk = [x0]
        Control = [u0]          
        # end loop variances
    # end loop objective times
    plt.draw()

# Loop end repetition conditions
fig1.suptitle('Random walk with dynamics ' + r'$dx = udt+d\xi$' + '\n' + 'Optimal control is ' + r'$u*(x,t)=\frac{tanh(\frac{x}{\nu(T-t)})-x}{T-t}$' + '\n' + 'Target location at ' + r'$t=T$' + ' is ' + r'$x = \pm 1$'
, fontsize=14)
fig2.suptitle('Control of dynamics ' + r'$dx = udt+d\xi$' + '\n' + 'Optimal control is ' + r'$u*(x,t)=\frac{tanh(\frac{x}{\nu(T-t)})-x}{T-t}$' + '\n' + 'Target location at ' + r'$t=T$' + ' is ' + r'$x = \pm 1$'
, fontsize=14)            
            