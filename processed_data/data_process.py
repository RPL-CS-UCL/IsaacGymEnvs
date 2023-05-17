import random

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
import os.path

# python
import enum
import numpy as np

class Data_Processing:
    def __init__(self):
        return

    def get_tensor_to_array(self,tensor, ind):
        cn = torch.zeros(len(tensor), ind)

        for i in range(len(tensor)):
            cn[i] = tensor[i]
        right_shape = cn.cpu().numpy()

        return right_shape

    def plot_sub(self,torques,foot_contacts,which_vel):

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

    def load_data(self):

        self.save_ref_period = []
        self.save_period =[]
        self.save_ref_joint_angles =[]
        self.save_joint_angles = []
        self.ref_com_vel = []
        self.save_com_vel = []
        self.save_torques = []

        directory = os.path.join(f'save_data/')


        for i in range(1, 1):
            # if (i == 5): continue # No 5 for some reason
            filepaths = [
                f'ref_period_{i}.pt',
                f'period_{i}.pt',
                f'ref_joint_angles_{i}.pt',
                f'joint_angles_{i}.pt',
                f'ref_com_vel_{i}.pt',
                f'com_vel_{i}.pt',
                f'torques_{i}.pt'
            ]

            filepaths = [directory + fp for fp in filepaths]
        

        ref_period = np.load(filepaths[0])['ref_period']
        period = np.load(filepaths[1])['period']
        ref_joint_angles = np.load(filepaths[2])['ref_joint_angles']
        joint_angles = np.load(filepaths[3])['joint_angles']
        ref_com_vel = np.load(filepaths[4])['ref_com_vel']
        com_vel = np.load(filepaths[5])['com_vel']
        torques = np.load(filepaths[6])['torques']
      

        self.save_ref_period.append(ref_period)
        self.save_period.append(period)
        self.save_ref_joint_angles.append(ref_joint_angles)
        self.save_joint_angles.append(joint_angles)
        self.ref_com_vel.append(ref_com_vel)
        self.save_com_vel.append(com_vel)
        self.save_torques.append(torques)

        return self.save_ref_period



# home_path =  '/home/robohike/IsaacGymEnvs/saved_data'

# #forces = torch.load(home_path+/)
# position = torch.load(home_path+'/fc0.5.pt')
# torque = torch.load(home_path+'/trq0.5.pt')

# positions = get_tensor_to_array(position,4)
# torques = get_tensor_to_array(torque,12)

# plot_sub(torques,positions,which_vel=0.5)

if __name__=='main':
    data_process = Data_Processing()
    test = data_process.load_data()
    print(test)






