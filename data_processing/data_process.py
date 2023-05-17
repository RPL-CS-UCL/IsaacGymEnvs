import random

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
import 

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

def load_data:

    for i in range(1, 1):
        # if (i == 5): continue # No 5 for some reason
        filepaths = [
            f'base_position{i}.pth',
            f'base_orientation{i}.npz',
            f'joint_angles{i}.npz',
            f'actions{i}.npz',
            f'period{i}.npz',
            f'foot{i}.npz',
            f'calf{i}.npz',
            f'hip{i}.npz',
            f'foot_contacts{i}.npz'
        ]
        filepaths = [path + fp for fp in filepaths]

        base_position = np.load(filepaths[0])['base_position']
        base_orientation = np.load(filepaths[1])['base_orientation']
        joint_angles = np.load(filepaths[2])['joint_angles']
        actions = np.load(filepaths[3])['actions']
        periods = np.load(filepaths[4])['period']
        foots = np.load(filepaths[5])['foot_pos']
        calfs = np.load(filepaths[6])['calf_pos']
        hips = np.load(filepaths[7])['hip_pos']
        foot_contacts = np.load(filepaths[8])['feet_contacts']





