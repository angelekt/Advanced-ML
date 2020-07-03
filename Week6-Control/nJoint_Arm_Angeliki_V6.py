import numpy as np
import random 
import matplotlib.pyplot as plt
from math import cos, sin, exp
import math
from tqdm import tqdm_notebook as tqdm


n_joints = 100
joint_length = 1
dt = 0.05
#dt1 = dt/10
#dt2 = dt1/10
T = 2
lr = 0.001/n_joints
alpha = 100# the bigger the better ( I tried 50, 100 and 10)
             # whenever I try alpha = 1000 it crushes
            # it affects the variance, the bigger it is, the smallest the variance
v = 0.5



class nJointArm:
    
    def __init__(self, n_joints, thetas, joint_length =1):
        self.n_joints = n_joints
        self.x = [0]*(n_joints+1) #x position of each joint
        self.y = [0]*(n_joints+1) #y position of each joint
        self.joint_angles = thetas
        self.joint_length = joint_length 
        self.m = np.zeros(self.n_joints)
        self.s = np.ones(self.n_joints)
        self.x_target, self.y_target = self.get_target_location() 
        self.x_exp, self.y_exp = self.expected_Values()
        self.total_length = self.joint_length*self.n_joints
        self.update_position()
        
    def update_position(self):

        for i in range(1,n_joints+1):
            self.x[i]=self.x[i-1]+np.cos(self.joint_angles[i-1]);
            self.y[i]=self.y[i-1]+np.sin(self.joint_angles[i-1]);

    def expected_Values(self):
        # <x_i>, <y_i>
        # mu, sigma : (3x1) , (3x1)
        # x_expected(0) = 0
        # y_expected(0) = 0
        # x_expected.shape = (4x1) points 
        # y_expected.shape = (4x1) points
        ## I think this should be the calculated without sigma as follows:

        global n_joints, joint_length
        mu, sigma = self.m, self.s
        x_expected = np.zeros(n_joints+1)
        y_expected = np.zeros(n_joints+1)
        x_expected[1] = np.cos(mu[0])*np.exp(-sigma[0]/2);
        y_expected[1] = np.sin(mu[0])*np.exp(-sigma[0]/2);
        for i in range(2,n_joints+1):
            x_expected[i] = x_expected[i-1]+np.cos(mu[i-1])*exp(-sigma[i-1]/2);
            y_expected[i] = y_expected[i-1]+np.sin(mu[i-1])*exp(-sigma[i-1]/2);
            
        return x_expected, y_expected

    def get_target_location(self):
        x_target = self.n_joints/2
        y_target = self.n_joints/2
        return x_target, y_target  
    
    def update_angles(self,new_angles):
        self.joint_angles = new_angles#(new_angles + np.pi) % (2 * np.pi) - np.pi
        self.update_position()        

    def plot(self):  # pragma: no cover
        plt.cla()
        # for stopping simulation with the esc key.
        self.x_exp, self.y_exp = self.expected_Values()
         
        for i in range(self.n_joints+1):
            if i is not self.n_joints:
                plt.plot([self.x[i], self.x[i + 1]],
                         [self.y[i], self.y[i + 1]], 'b-' )
                plt.plot([self.x_exp[i], self.x_exp[i+1]], [
                 self.y_exp[i], self.y_exp[i+1]], 'k--')
            plt.plot(self.x[i], self.y[i], 'go')
            #plt.plot(self.x_target[i], self.y_target[i], 'ro')

        plt.plot(self.x_target, self.y_target, 'k*', label = 'Target')
        plt.plot(self.x[0], self.y[0], 'rx', label = 't = ')

        plt.xlim([-1*self.total_length, 1*self.total_length])
        plt.ylim([-1*self.total_length, 1*self.total_length])        
        plt.draw()
        plt.title('N-joint arm progress')
        plt.xlabel('x'); plt.ylabel('y')
          
        plt.legend(loc='upper right')
  

    ############################## FUNCTIONS #########################################
    ##################################################################################


    def update_parameters( self, t,  mu_old, sigma_old):
        # NOT ARRAYS everything a number
        ## answer(xuan): no, vdt = var(noise)+mu(noise)
        # a : parameter
        # mu_old and sigma_old (sigma_old = sigma**2)

        # returns : mu and sigma 
        global alpha, v, n_joints, joint_length, T
        #print("Update Parameters")
        theta_old = self.joint_angles
        a = alpha
        x_n = np.cos(mu_old).dot(np.exp(-sigma_old.T/2)); # mean end point
        #print(x_n)
        y_n = np.sin(mu_old).dot(np.exp(-sigma_old.T/2)); # mean end point
        #print(y_n)
        x_target, y_target = self.get_target_location()
        mu = theta_old + a*(T-t)*(np.sin(mu_old)*np.exp(-sigma_old/2)*(x_n - x_target) - \
                              (np.cos(mu_old)*np.exp(-sigma_old/2)*(y_n - y_target)))

        sigma_inv = (1/v)*((1/(T-t)) + a*np.exp(-sigma_old) - \
                           a*(np.sin(mu_old)*np.exp(-sigma_old/2)*(y_n - y_target)) - \
                           a*(np.cos(mu_old)*np.exp(-sigma_old/2)*(x_n - x_target))) 

        dm = mu - mu_old
        mu = mu_old + lr * dm
        dsigma = 1/(sigma_inv) - sigma_old
        sigma = sigma_old + lr *dsigma
        diff = np.maximum(np.absolute(dm),np.absolute(dsigma))
        d = np.where(diff > 0.01, diff,0)
        if all(d) == False:            
            diff = 0
        else:
            diff = 1

        return mu,sigma,diff 


    def new_angles(self, t, dt_temp ,  theta, theta_expected):
        # u: controller
        # t: time point
        # T: total time
        # dt: time_new-time_old (time step)
        # dksi: noise
        # theta_expected = mu 
        # formula
        global T, v
        #print("New angles")
        dksi = np.sqrt(v*dt_temp)*np.random.randn(1,self.n_joints) #
        u = self.update_controller( theta, theta_expected, t)
        theta_new = theta + u*dt_temp + dksi
        return theta_new[0]

    def update_controller(self, theta, theta_expected, t):
        # u: controller
        # t: time point
        # T: total time
        global T
        ## what is the definition of theta_expected
        ### ANSWER ANGELIKI : the definition of theta_expected is the mu after convergence
        return (1/(T-t))*(theta_expected - theta)

    
    def compute_ELBO(self,t):
        global alpha, v, T
        
        sigma = self.s # sigma = sigma**2
        mu = self.m
        theta_old = self.joint_angles
        x_target,y_target = self.x_target, self.y_target
        
        x_exp = np.cos(mu).dot(np.exp(-sigma.T/2)); # mean end
        
        y_exp = np.sin(mu).dot(np.exp(-sigma.T/2)); # mean end

        Phi_expected = (alpha/2) *( (1-np.exp(-sigma)).sum() + (x_exp - x_target)**2+ (y_exp - y_target)**2 )
        elbo1 = (1/(2*v*(T-t)))*(sigma +np.square(mu - theta_old)).sum()
        elbo2 = (1/v)*Phi_expected
        entropy = -(np.log(np.sqrt(2*np.pi*sigma)) + 0.5).sum()
        ELBO = - entropy  - elbo1 - elbo2 
        #print(ELBO)
        return ELBO.sum()

    def plot_diagr(self, ELBO):#,Hx_mean,F):    
        plt.figure()
        plt.plot(ELBO, color='b', lw=2.0, label='ELBO')
        plt.title('Variational Inference for the n-joint arm')
        plt.xlabel('iterations'); plt.ylabel('ELBO objective')
        plt.legend(loc='upper left')
        
    def learning(self, times):
        global n_joints, joint_length, alpha, v
        
        targets = self.x_target, self.y_target
        ELBO = []
        mu = self.m
        sigma = self.s
        t0 = 0
        with tqdm(total=len(times)) as pbar:
            for j in range(len(times)):
                print(j)
                t = times[j]
                dt_temp = t-t0
                expected = self.expected_Values()

                diff = 1
                #mu_new, sigma_new,diff= self.update_parameters(t, mu, sigma)
                mu_new, sigma_new = self.m,self.s

                while (diff>0.001):
                    mu_new, sigma_new,diff= self.update_parameters(t, mu_new, sigma_new)                
                    #print('diff = ',diff)
                theta_new = self.new_angles(t, dt_temp, self.joint_angles, mu_new)

                self.m = mu_new
                self.s = sigma_new
                ELBO.append(self.compute_ELBO(t))
                self.update_angles(theta_new)
                t0 = t
                pbar.update(1)
                if j%10 == 0 :#((t == 0.05) or (t == 0.55) or (t ==1.8) or (t==2)):
                    plt.figure(j)
            self.plot()

  
            
        self.plot_diagr(np.asarray(ELBO))



def main():
    print('we are starting')
    #times = np.concatenate(np.asarray([np.arange(0, 2-dt, dt),np.arange(2-dt, 2-dt1, dt1),np.arange(2-dt1, 2, dt2)]))
    times = np.arange(0, 2, dt)
    #thetas = [random.uniform(math.radians(-180),math.radians(+180)) for i in range(n_joints)]
 
    thetas = [np.pi/4 for i in range(n_joints)]
    narm = nJointArm(n_joints, thetas)

    # Initialization of parameters

    narm.learning(times)
    plt.show()
    
    
if __name__ == '__main__':
    main()
