import numpy as np
import matplotlib.pyplot as plt
import yaml
from math import exp
import random
import math
import sys
import tqdm

class nJointArm:
    
    def __init__(self,  thetas, ax, n_joints = 3, T =2, v =0.5, alpha =100, lr = 0.0001,  joint_length =1, target_flag = True):
        self.n_joints = n_joints
        self.axis = ax
        self.x = [0]*(n_joints+1) #x position of each joint
        self.y = [0]*(n_joints+1) #y position of each joint
        self.joint_angles = thetas 
        self.joint_length = joint_length 
        self.m = np.zeros(self.n_joints) # initialization of mu_i
        self.s = np.ones(self.n_joints) # initialization of sigma_i
        self.target_flag = target_flag        
        self.x_target, self.y_target = self.get_target_location()  
        self.x_exp, self.y_exp = self.expected_Values() 
        self.total_length = self.joint_length*self.n_joints #total length of the arm
        self.variance = v #variance noise
        self.T = T # Time to reach the target
        self.alpha = alpha #alpha parameter
        self.lr = lr # learning rate
        self.update_position()
        
    def update_position(self):
        # update the position of the joints
        for i in range(1,self.n_joints+1):
            self.x[i]=self.x[i-1]+np.cos(self.joint_angles[i-1]);
            self.y[i]=self.y[i-1]+np.sin(self.joint_angles[i-1]);

    def expected_Values(self):
        # <x_i>, <y_i>
        # mu, sigma : (3x1) , (3x1)
        # x_expected(0) = 0
        # y_expected(0) = 0
        # x_expected.shape = (4x1) points 
        # y_expected.shape = (4x1) points

        x_expected = np.zeros(self.n_joints+1)
        y_expected = np.zeros(self.n_joints+1)
        x_expected[1] = self.joint_length*np.cos(self.m[0])*np.exp(-self.s[0]/2);
        y_expected[1] = self.joint_length*np.sin(self.m[0])*np.exp(-self.s[0]/2);
        for i in range(2,self.n_joints+1):
            x_expected[i] = x_expected[i-1]+self.joint_length*np.cos(self.m[i-1])*exp(-self.s[i-1]/2);
            y_expected[i] = y_expected[i-1]+self.joint_length*np.sin(self.m[i-1])*exp(-self.s[i-1]/2);
            
        return x_expected, y_expected

    def get_target_location(self):
        # get the coordinates (x,y) of the target location
        if self.target_flag:
            x_target = self.n_joints/2
            y_target = self.n_joints/2
        else:
            # randomly chosen targets from a uniform distribution 
            x_target = np.random.uniform(-self.n_joints/1.5,self.n_joints/1.5,1)
            y_target = np.random.uniform(-self.n_joints/1.5,self.n_joints/1.5,1)
        return x_target, y_target  
    
    def update_angles(self,new_angles,t):
        # Update of the angles
        self.joint_angles = new_angles
        self.update_position()        

    def plot_position(self,ax,t):
        # creates the plots of the position of the n-joint arm
        self.x_exp, self.y_exp = self.expected_Values()
         
        for i in range(self.n_joints+1):
            if i is not self.n_joints:
                ax.plot([self.x[i], self.x[i + 1]],
                         [self.y[i], self.y[i + 1]], 'b-')
                ax.plot([self.x_exp[i], self.x_exp[i+1]], [
                 self.y_exp[i], self.y_exp[i+1]], 'k--')
            ax.plot(self.x[i], self.y[i], 'go')

        ax.plot(self.x_target, self.y_target, 'k*', label = 'Target')
        ax.plot(self.x[0], self.y[0], 'rx', label = 'Start')

        ax.set_xlim([-1*self.total_length, 1*self.total_length])
        ax.set_ylim([-1*self.total_length, 1*self.total_length])        
        ax.set_title('t = '+str(t), fontsize=10)
        ax.set_xlabel('x'); plt.ylabel('y')
        ax.legend(loc="upper left", fontsize=10)
        plt.draw()

  

    ############################## FUNCTIONS #########################################
    ##################################################################################


    def update_parameters( self, t,  mu_old, sigma_old):

        # vdt = var(noise)+mu(noise)
        # a : parameter
        # mu_old and sigma_old 
        # returns : mu, sigma and 0 for convergence 1 otherwise
        
        x_n = np.cos(mu_old).dot(np.exp(-sigma_old.T/2)); # mean end point
        y_n = np.sin(mu_old).dot(np.exp(-sigma_old.T/2)); # mean end point
        x_target, y_target = self.get_target_location()
        mu = self.joint_angles + self.alpha*(self.T-t)*(np.sin(mu_old)*np.exp(-sigma_old/2)*(x_n - x_target) - \
                              (np.cos(mu_old)*np.exp(-sigma_old/2)*(y_n - y_target)))
        sigma_inv = (1/self.variance)*((1/(self.T-t)) + self.alpha*np.exp(-sigma_old) - \
                                       self.alpha*(np.sin(mu_old)*np.exp(-sigma_old/2.)*(y_n - y_target)) - \
                                           self.alpha*(np.cos(mu_old)*np.exp(-sigma_old/2.)*(x_n - x_target))) 

        dm = mu - mu_old
        mu = mu_old + self.lr * dm
        if sigma_inv.all():
            sigma_inv += sys.float_info.epsilon
        dsigma = 1/(sigma_inv) - sigma_old
        sigma = sigma_old + self.lr *dsigma
        #if sigma.all():
        #    sigma += sys.float_info.epsilon
        diff = np.maximum(np.absolute(dm),np.absolute(dsigma))
        d = np.where(diff > 0.01, diff,False) # if dm or dsigma <0.01 then the update has converge
        diff = all(d)

        return mu,sigma,diff 


    def new_angles(self, t, dt_temp ,  theta, theta_expected):
        # u: controller
        # t: time point
        # T: total time
        # dt: time_new-time_old (time step)
        # dksi: noise
        # theta_expected = mu 
        # formula : dtheta = udt + dxi
        dksi = np.sqrt(self.variance*dt_temp)*np.random.randn(1,self.n_joints) #
        u = self.update_controller( theta, theta_expected, t)
        theta_new = theta + u*dt_temp + dksi
        return theta_new[0]

    def update_controller(self, theta, theta_expected, t):
        # u: controller
        # t: time point
        # T: total time
        return (1/(self.T-t))*(theta_expected - theta)

    
    def compute_ELBO(self,t):
        # compute the ELBO
        theta_old = self.joint_angles
        x_target,y_target = self.x_target, self.y_target
        
        x_exp = np.cos(self.m).dot(np.exp(-self.s.T/2)); # mean end
        
        y_exp = np.sin(self.m).dot(np.exp(-self.s.T/2)); # mean end

        Phi_expected = (self.alpha/2) *( (1-np.exp(-self.s)).sum() + (x_exp - x_target)**2+ (y_exp - y_target)**2 )
        elbo1 = (1/(2*self.variance*(self.T-t)))*(self.s +np.square(self.m - theta_old)).sum()
        elbo2 = (1/self.variance)*Phi_expected
        entropy = -(np.log(np.sqrt(2*np.pi*self.s)) + 0.5).sum()
        ELBO = - entropy  - elbo1 - elbo2 
        return ELBO.sum()

    def plot_diagr(self, ELBO, ax_elbo):
        # Creates the figure of the ELBO
        ax_elbo.plot(ELBO, color='b', lw=2.0, label='ELBO')
        ax_elbo.set_title('Variational Inference for the '+str(self.n_joints)+'-joint arm')
        ax_elbo.set_xlabel('iterations'); plt.ylabel('ELBO objective')
        ax_elbo.legend(loc='upper left')
        plt.draw()
        plt.savefig('ELBO_random_'+str(self.n_joints)+'.eps', format='eps')
        
    def learning(self, times, elbo_flag = True):
        # Learn the parameters
        ELBO = []
        fig_num = 0
        t0 = 0
        for j in tqdm.tqdm(range(len(times))):
                t = times[j]
                dt_temp = t-t0
                diff = 1
                mu_new, sigma_new = self.m,self.s

                while diff:
                    mu_new, sigma_new,diff= self.update_parameters(t, mu_new, sigma_new)
                    theta_new = self.new_angles(t, dt_temp, self.joint_angles, mu_new)

                self.m = mu_new
                self.s = sigma_new
                ELBO.append(self.compute_ELBO(t))
                self.update_angles(theta_new,t)
                t0 = t
                if ((t == 0.05) or (t == 0.55) or (t ==1.8) or (t==times[-1])):
                    self.plot_position(self.axis[fig_num],t)
                    fig_num+=1
                
        fig_elbo, ax_elbo = plt.subplots(1)
        if elbo_flag:
            self.plot_diagr(ELBO, ax_elbo)

def main():
    
    
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f)
     
    times = np.arange(0, 2, cfg['dt'])
    
    if cfg['Example1']:
        fig, ax = plt.subplots(2,4,figsize=(10,10))  
              
        thetas = [np.pi/3 for i in range(cfg['n_joints_3'])]        
        n_arm3 = nJointArm(thetas, ax[0,:], cfg['n_joints_3'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'])
        
        thetas = [np.pi/3 for i in range(cfg['n_joints_100'])]
        n_arm100 = nJointArm(thetas, ax[1,:], cfg['n_joints_100'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'])    

        
        n_arm3.learning(times)
        n_arm100.learning(times)
        
        plt.show()
        fig.savefig('PathIntegral.eps', format='eps')
        #fig.savefig('PathIntegral.png')#, format='eps')
        
    elif cfg['Example2']:
         
        fig, ax = plt.subplots(2,4,figsize=(10,10))  
              
        thetas = [random.uniform(math.radians(-180),math.radians(+180)) for i in range(cfg['n_joints_3'])]        
        n_arm3 = nJointArm(thetas, ax[0,:], cfg['n_joints_3'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'],cfg['target_flag'])
        
        thetas = [random.uniform(math.radians(-180),math.radians(+180)) for i in range(cfg['n_joints_100'])]
        n_arm100 = nJointArm(thetas, ax[1,:], cfg['n_joints_100'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'],cfg['target_flag'])    

        
        n_arm3.learning(times)
        n_arm100.learning(times)
        
        plt.show()
        #fig.savefig('PathIntegral_example2.eps', format='eps') 
        
        
    elif not cfg['target_flag']:
        # targets randomly chosen
                
        fig, ax = plt.subplots(1,4,figsize=(10,10))
        thetas = [random.uniform(math.radians(-180),math.radians(+180)) for i in range(cfg['n_joints'])]
        n_arm = nJointArm(thetas, ax, cfg['n_joints'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'],cfg['target_flag'])
        
        n_arm.learning(times)
        plt.show()
        #fig.savefig('PathIntegral_randomTarget.eps', format='eps')

                
    else:

        fig, ax = plt.subplots(1,4,figsize=(10,10))
        thetas = [random.uniform(math.radians(-180),math.radians(+180)) for i in range(cfg['n_joints'])]
        n_arm = nJointArm(thetas, ax, cfg['n_joints'], cfg['T'], cfg['v'], cfg['alpha'] , cfg['lr'], cfg['joint_length'])
        
        n_arm.learning(times)
        plt.show()
        

    
if __name__ == '__main__':
    main()
