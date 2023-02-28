import random

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt

# python
import enum
import numpy as np

def get_tensor_to_array(tensor, ind):
    cn = torch.zeros(len(tensor), ind)

    for i in range(len(tensor)):
        cn[i] = tensor[i]
    right_shape = cn.cpu().numpy()

    return right_shape

def plot_sub(torques,foot_contacts,which_vel):

    foot_contacts = np.array(foot_contacts)

    x_axis1 = np.arange(0,len(torques),1)
    x_axis2 = np.arange(0,len(foot_contacts[:,0]),1)

    f, (ax1, ax2) = plt.subplots(2, 1, 
                             sharey=False)
    
   
    ax1.plot(x_axis1,torques)
    ax1.set(xlabel='Time (s)', ylabel='Torques (Nm/s)', title='Isaac Unfilter Torques for Velocity 0.5 m/s')
    ax1.set_ylim([-50, 50])

    

    ax2.scatter(x_axis2,foot_contacts[:,0], marker="_",color='red')
    ax2.scatter(x_axis2,foot_contacts[:,1]-0.05,marker="_",color='blue')
    ax2.scatter(x_axis2,foot_contacts[:,2]-0.1,marker="_", color='magenta')
    ax2.scatter(x_axis2,foot_contacts[:,3]-0.15,marker="_",color='green')
    ax2.set(xlabel='Time (s)', ylabel='Conatct Points', title='Isaac Foothold Graph of Velocity 0.5 m/s')
    ax2.legend(('Front Left','Front Right','Rare Left','Rare Right'))
    ax2.set_ylim([0.8,1.1])

    
    plt.show()




home_path =  '/home/robohike/IsaacGymEnvs/saved_data'

#forces = torch.load(home_path+/)
position = torch.load(home_path+'/fc0.5.pt')
torque = torch.load(home_path+'/trq0.5.pt')

positions = get_tensor_to_array(position,4)
torques = get_tensor_to_array(torque,12)

plot_sub(torques,positions,which_vel=0.5)





