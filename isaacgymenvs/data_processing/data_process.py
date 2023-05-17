import random

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from tabulate import tabulate
# python
import enum
import numpy as np

class Data_Processing:
    def __init__(self,max_time,velocity):
        self.data_dict = {"ref_period":[],"period":[],"ref_joint_angles":[],
        "joint_angles":[],"ref_com_vel":[],"com_vel":[],"torques":[],"ref_base_pose":[],"base_pose":[]}
        self.max_time = max_time
        self.velocity=velocity

     

    def get_tensor_to_array(self,tensor):

        tensor_format = tensor[0]
        ind = tensor_format.shape[-1]
        cn = torch.zeros(len(tensor), ind)


        for i in range(len(tensor)):
            cn[i] = tensor[i]
        right_shape = cn.cpu().numpy()

        return right_shape



    def load_data(self):

        self.save_ref_period = []
        self.save_period =[]
        self.save_ref_joint_angles =[]
        self.save_joint_angles = []
        self.ref_com_vel = []
        self.save_com_vel = []
        self.save_torques = []
        self.save_period = []
        self.save_ref_joint_angles = []

        directory = os.path.join(f'save_data/')


        # for i in range(7,7):
        i=7
            # if (i == 5): continue # No 5 for some reason
        filepaths = [
            f'ref_period_{i}.pt',
            f'period_{i}.pt',
            f'ref_joint_angles_{i}.pt',
            f'joint_angles_{i}.pt',
            f'ref_com_vel_{i}.pt',
            f'com_vel_{i}.pt',
            f'torques_{i}.pt',
            f'ref_base_position_{i}.pt',
            f'base_position_{i}.pt'
        ]

        filepaths = [directory + fp for fp in filepaths]



        ref_period = self.get_tensor_to_array(torch.load(filepaths[0]))
        period = self.get_tensor_to_array(torch.load(filepaths[1]))
        ref_joint_angles = self.get_tensor_to_array(torch.load(filepaths[2]))
        joint_angles = self.get_tensor_to_array(torch.load(filepaths[3]))
        ref_com_vel = self.get_tensor_to_array(torch.load(filepaths[4]))
        com_vel = self.get_tensor_to_array(torch.load(filepaths[5]))
        torques = self.get_tensor_to_array(torch.load(filepaths[6]))
        ref_base_pos = self.get_tensor_to_array(torch.load(filepaths[7]))
        base_pos = self.get_tensor_to_array(torch.load(filepaths[8]))


        self.data_dict["ref_period"].append(ref_period)
        self.data_dict["period"].append(period)
        self.data_dict["ref_joint_angles"].append(ref_joint_angles)
        self.data_dict["joint_angles"].append(joint_angles)
        self.data_dict["ref_com_vel"].append(ref_com_vel)
        self.data_dict["com_vel"].append(com_vel)
        self.data_dict["torques"].append(torques)
        self.data_dict["ref_base_pose"].append(ref_base_pos)
        self.data_dict["base_pose"].append(base_pos)

        return self.data_dict



    def process_velocity(self):

        velocity = int(10*self.velocity)
        names = ["Forward Linear Velocity", "Time (s)","Velocity (m/s)","Error:Forward Linear Velocity", "Time (s)","Error Velocity (m/s)" ]

        self.sim_vel = np.squeeze(np.array(self.data_dict['com_vel']))[:, 0]
        self.target_vel = [self.velocity, ] * len(self.sim_vel)

        errors,self.error_vel = self.percent_error(self.target_vel, self.sim_vel)
        self.plot_data(self.target_vel, self.sim_vel,names,errors)

    def process_period(self):

        velocity = int(10*self.velocity)
        names = ["Gait Period Matching", "Time (s)","Gait Period ","Error: Gait Period Matching", "Time (s)","Error Gait Period Matching " ]

        self.sim_period = np.mean(np.squeeze(np.array(self.data_dict['period'])), axis=-1)
        self.target_period = np.mean(np.squeeze(np.array(self.data_dict['ref_period'])), axis=-1)

        errors,self.error_period = self.percent_error(self.sim_period ,self.target_period)
        self.plot_data(self.target_period, self.sim_period,names,errors)

    def process_joint_angles(self):
        velocity = int(10 * self.velocity)
        names = ["Joint Angles Matching", "Time (s)", "Joint Angles Matching", "Error: Joint Angles Matching", "Time (s)",
                 "Error Joint Angles Matching"]

        joint_angle_index = 1

        self.sim_joint_angles = np.squeeze(np.array(self.data_dict['joint_angles']))[joint_angle_index]
        self.target_joint_angles = np.squeeze(np.array(self.data_dict['ref_joint_angles']))[joint_angle_index]

        errors, self.error_ja = self.percent_error(self.sim_joint_angles, self.target_joint_angles)
        self.plot_data(self.target_joint_angles, self.sim_joint_angles, names, errors)

    def process_base_position(self):
        velocity = int(10 * self.velocity)
        names = ["Base Position", "Time (s)", "Base Position", "Error: Base Position", "Time (s)",
                 "Error Base Position"]

        joint_angle_index = 1

        self.sim_base_pose = np.squeeze(np.array(self.data_dict['base_pose']))
        self.target_base_pose= np.squeeze(np.array(self.data_dict['ref_base_pose']))

        errors, self.error_bp = self.percent_error(self.sim_base_pose, self.target_base_pose)
        self.plot_data(self.target_base_pose, self.sim_base_pose, names, errors)


    def percent_error(self, val1, val2):
        errors = np.square(val1 -  val2)
        error= np.sum(val1)
        return errors, error

    def plot_data(self,target,sim,names,errors):

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)), sim, color='black', linestyle="-",
                    label="Measured")
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)), target, color='black', linestyle="--",
                    label="Desired")
        axs[0].legend()
        axs[0].set_title(names[0])
        axs[0].set_xlabel(names[1])
        axs[0].set_ylabel(names[2])
        axs[1].plot(np.linspace(0, self.max_time, num=len(target)), errors, color='black', linestyle="-",
                    label="Measured")
        axs[1].legend()
        axs[1].set_title(names[3])
        axs[1].set_xlabel(names[4])
        axs[1].set_ylabel(names[5])
        plt.show()

    def get_error_table(self):


        data = [["period", self.error_period],
                ["joint_angles", self.error_ja],
                ["com_vel",self.error_vel],
                ["base_pose", self.error_bp]]

        # define header names
        col_names = ["Parameter", "v = 0.7 m/s"]

        # display table
        print(tabulate(data, headers=col_names))




data_process = Data_Processing(max_time=20,velocity=0.7)
data_dict = data_process.load_data()
data_process.process_velocity()
data_process.process_period()
data_process.process_joint_angles()
data_process.process_base_position()
data_process.get_error_table()








