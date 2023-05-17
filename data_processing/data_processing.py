import random

import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt

# python
import enum
import numpy as np

velocities = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

################# Helper Functions ###############
def get_tensor_to_array(tensor, ind):
    cn = torch.zeros(len(tensor), ind)

    for i in range(len(tensor)):
        cn[i] = tensor[i]
    right_shape = cn.cpu().numpy()

    return right_shape

def get_percent_error(reference, simulation):
    error = ((simulation - reference) / reference) * 100
    return error

################ Load Data #########################

def _load_data(file) :
    f_path = '/home/robohike/test_data/fc'
    v_path = '/home/robohike/test_data/vel'
    t_path = '/home/robohike/test_data/trq'
    ref_path = '/home/robohike/test_data/ref'
    pos_path = '/home/robohike/test_data/pos'
    period_path = '/home/robohike/test_data/period'

    forces = torch.load(f_path+file)
    torques = torch.load(t_path+file)
    vel = torch.load(v_path+file)
    ref = torch.load(ref_path+file)
    pos = torch.load(pos_path+file)
    period = torch.load(period_path+file)
    return forces, torques, vel, ref, pos, period

def load_sim(which_file):

    contacts01, torques01, vel01, ref01, pos01, period01 = _load_data(which_file+'01.pt')
    contacts02, torques02, vel02, ref02, pos02, period02 = _load_data(which_file+'02.pt')
    contacts03, torques03, vel03, ref03, pos03, period03 = _load_data(which_file+'03.pt')
    contacts04, torques04, vel04, ref04, pos04, period04 = _load_data(which_file+'04.pt')
    contacts06, torques06, vel06, ref06, pos06, period06 = _load_data(which_file+'06.pt')
    contacts05, torques05, vel05, ref05, pos05, period05 = _load_data(which_file+'05.pt')
    contacts07, torques07, vel07, ref07, pos07, period07 = _load_data(which_file+'07.pt')
    contacts08, torques08, vel08, ref08, pos08, period08 = _load_data(which_file+'08.pt')
    contacts09, torques09, vel09, ref09, pos09, period09 = _load_data(which_file+'09.pt')

    contacts = [contacts01, contacts02, contacts03, contacts04, contacts05, contacts06,contacts07, contacts08]
    torques = [torques01, torques02, torques03, torques04,torques05, torques06, torques07, torques08]
    references = [ref01, ref02, ref03, ref04, ref05, ref06, ref07, ref08, ]
    periods = [period01, period02, period03, period04, period05, period06, period07, period08]
    vels = [vel01,vel02,vel03,vel04, vel05, vel06, vel07,vel08]

    #return contacts, torques, references, periods, vels
    return torques, contacts

def load_mpc():

    home_dir = os.path.expanduser('~')
    path = home_dir+'/motion_imitation/2500/'

    # load MPC gaits
    ########### tripod

    with np.load(path + "foot_contacts1.npz") as ct1:
        ct1 = ct1["feet_contacts"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot_contacts2.npz") as ct2:
        ct2 = ct2["feet_contacts"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot_contacts3.npz") as ct3:
        ct3 = ct3["feet_contacts"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot_contacts4.npz") as ct4:
        ct4 = ct4["feet_contacts"]  # [n_time_steps, feet indices, xyz
    with np.load(path + "foot_contacts5.npz") as ct5:
        ct5 = ct5["feet_contacts"]
    with np.load(path + "foot_contacts6.npz") as ct6:
        ct6 = ct6["feet_contacts"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot_contacts7.npz") as ct7:
        ct7 = ct7["feet_contacts"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot_contacts8.npz") as ct8:
        ct8 = ct8["feet_contacts"]  # [n_time_steps, feet indices, xyz
    with np.load(path + "foot_contacts9.npz") as ct9:
        ct9 = ct9["feet_contacts"]


    with np.load(path + "foot1.npz") as f1:
        f1 = f1["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot2.npz") as f2:
        f2 = f2["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot3.npz") as f3:
        f3 = f3["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot4.npz") as f4:
        f4 = f4["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot5.npz") as f5:
        f5 = f5["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot6.npz") as f6:
        f6 = f6["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot7.npz") as f7:
        f7 = f7["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot8.npz") as f8:
        f8 = f8["foot_pos"]  # [n_time_steps, feet indices, xyz]
    with np.load(path + "foot9.npz") as f9:
        f9 = f9["foot_pos"]  # [n_time_steps, feet indices, xyz]

    with np.load(path + "period1.npz") as period1:
        period1 = period1["period"]
    with np.load(path + "period2.npz") as period2:
        period2 = period2["period"]
    with np.load(path + "period3.npz") as period3:
        period3 = period3["period"]
    with np.load(path + "period4.npz") as period4:
        period4 = period4["period"]
    with np.load(path + "period5.npz") as period5:
        period5 = period5["period"]
    with np.load(path + "period6.npz") as period6:
        period6 = period6["period"]
    with np.load(path + "period7.npz") as period7:
        period7 = period7["period"]
    with np.load(path + "period8.npz") as period8:
        period8 = period8["period"]
    with np.load(path + "period9.npz") as period9:
        period9 = period9["period"]

    contacts = [ct1,ct2,ct3,ct4,ct5,ct6, ct7, ct8, ct9]
    positions = [f1, f2,f3, f4,f5, f6, f7, f8]
    periods = [period1,period2,period3,period4,period5,period6,period7,period8]

    return contacts, positions, periods

################### Gait Data ###########################


def get_sim_gaits(which_gait,which_file):

    contacts = load_sim(which_file)[0]

    contacts =contacts[which_gait]
    np_contact = get_tensor_to_array(contacts,12)

    fl_foot = np_contact[:, 0]
    fr_foot = np_contact[:, 1]
    rl_foot = np_contact[:, 2]
    rr_foot = np_contact[:, 3]

    # fl_foot = np.array(np.where(np_contact[:, 1] == (1)))
    # fr_foot = np.array(np.where(np_contact[:, 0] == (1)))
    # rl_foot = np.array(np.where(np_contact[:, 3] == (1)))
    # rr_foot = np.array(np.where(np_contact[:, 2] == (1)))

    return fl_foot,fr_foot,rl_foot,rr_foot

def get_ref_gait(which_gait):
    contacts, positions, periods = load_mpc()

    offset = 8
    offset_index = np.zeros((offset,4))
    contacts_mpc= np.zeros((2500,4))
    footsteps_mpc = np.zeros((2500, 4))


    contacts = contacts[which_gait]
    positions = positions[which_gait]


    contacts_mpc[:offset,:] = offset_index
    contacts_mpc[offset:,:] = contacts[:2500 -offset,:]

    footsteps_mpc[:offset, :] = offset_index
    footsteps_mpc = positions[:2500 -offset,:]

    offset = np.zeros(7)

    fl = contacts_mpc[:, 0]
    fr = contacts_mpc[:, 1]
    rl = contacts_mpc[:, 2]
    rr = contacts_mpc[:, 3]


    # fr = np.array(np.where(contacts_mpc[:, 1] == (1)))
    # fl = np.array(np.where(contacts_mpc[:, 0] == (1)))
    # rr = np.array(np.where(contacts_mpc[:, 3] == (1)))
    # rl = np.array(np.where(contacts_mpc[:, 2] == (1)))

    return fr, fl, rr, rl

def get_gaits_plot(bottom,top,which_gait,which_file,velocities):
    offset =7

    fl_foot,fr_foot,rr_foot,rl_foot = get_sim_gaits(which_gait,which_file)
    fl_ref,fr_ref,rr_ref,rl_ref = get_ref_gait(which_gait)


    fl_foot = fl_foot[bottom:top]
    fr_foot = fr_foot[bottom:top]
    rl_foot = rl_foot[bottom:top]
    rr_foot = rr_foot[bottom:top]

    fl_ref = fl_ref[bottom:top]
    fr_ref = fr_ref[bottom:top]
    rl_ref = rl_ref[bottom:top]
    rr_ref = rr_ref[bottom:top]



    ################### Gait PLot ######################

    which_vel = velocities[which_gait]

    episode = np.arange(0,top-bottom)

    plt.scatter(episode,fl_foot, marker="_",color='red')
    plt.scatter(episode,fr_foot-0.05,marker="_",color='blue')
    plt.scatter(episode,rl_foot-0.1,marker="_", color='magenta')
    plt.scatter(episode,rr_foot-0.15,marker="_",color='green')

    # plt.scatter(episode,fl_ref, marker="x", color='red')
    # plt.scatter(episode,fr_ref-0.05,marker="x",color='blue')
    # plt.scatter(episode,rl_ref-0.1,marker="x",color='magenta')
    # plt.scatter(episode,rr_ref-0.15,marker="x",color='green')

    plt.scatter(episode,fl_ref-0.3, marker="x", color='red')
    plt.scatter(episode,fr_ref-0.35,marker="x",color='blue')
    plt.scatter(episode,rl_ref-0.4,marker="x",color='magenta')
    plt.scatter(episode,rr_ref-0.45,marker="x",color='green')

    plt.legend(('Front Left','Front Right','Rare Left','Rare Right'))
    plt.title('Stance Graph of Velocity %0.1f m/s Against Optimal' %which_vel)
    plt.ylim((0.5,1.1))
    plt.axis('off')
    plt.show()

def get_gait_to_ref_stat(which_gait,which_file):


    fl_foot, fr_foot, rr_foot, rl_foot = get_sim_gaits(which_gait,which_file)

    fl_ref, fr_ref, rr_ref, rl_ref = get_ref_gait(which_gait)

    reference = np.mean(np.sum([fl_foot, fr_foot, rr_foot, rl_foot ], axis=0))
    simulation = np.mean(np.sum([fl_ref, fr_ref, rr_ref, rl_ref ], axis =0))

    error = get_percent_error(reference,simulation)

    return error


################## Period Data #############

def get_period_error(which_gait,which_file):

    #load sim data


    periods_sim = load_sim(which_file)[3]
    period_sim = periods_sim[which_gait]

    period_sim_mean = np.mean(get_tensor_to_array(period_sim, 1))

    #load ref data
    periods_ref = load_mpc()[2]
    period_ref = periods_ref[which_gait]

    error = get_percent_error(period_ref,period_sim_mean)
    error = np.mean(error)
    # x = np.arrange(1,len(periods_sim))
    # plt.scatter(x,period_sim)
    # plt.show()

    return error


#
# error_gait = get_gait_to_ref_stat(which_gait,which_file)
# error_gait_curriculum = get_gait_to_ref_stat(which_gait,'c')
# print('Gait Error: ', error_gait)
# print('Gait Error Curriculum: ', error_gait_curriculum)
#
# error_period = get_period_error(which_gait,which_file)
# error_period_curriculum = get_period_error(which_gait,'c')
# print('Period Error: ', error_period)
# print('Period Error Curriculum: ', error_period_curriculum)
# #

################ Velocity Blending ################


def velocity_profile(vel_chosen,velocities):


    ind1 = np.random.randint(0,len(velocities),)
    ind2 = np.random.randint(0,len(velocities),)

    # velocities corresponding to these indeces
    vel1 = 0.2 #velocities[ind1]
    vel2 = 0.5 #velocities[ind2]

    w = 0.05
    D = np.linspace(0, 2, len(vel_chosen))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels1 = vel1 + (vel2 - vel1) * (1 - sigmaD)

    w2 = 0.08
    D2 = np.linspace(0, 2, len(vel_chosen))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D2) / w2))
    vels2 = vel1 + (vel2 - vel1) * (1 - sigmaD)

    w3 = 0.15
    D3 = np.linspace(0, 2, len(vel_chosen))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D3) / w3))
    vels3 = vel1 + (vel2 - vel1) * (1 - sigmaD)

    return vels1, vels2, vels3, [ind1,ind2]


def velocity_blending(which_gait,which_file):

    vels_sim = load_sim(which_file)[-1]
    vel_sim_ = vels_sim[which_gait]
    vel_sim= get_tensor_to_array(vel_sim_,3)


    vel_simulation =vel_sim[:,0]
    vel_profile1, vel_profile2, vel_profile3,vels = velocity_profile(vel_simulation,velocities)



    error = get_percent_error(vel_profile1,vel_simulation)
    error_velocity_blending = np.mean(error)

    return vel_simulation,vel_profile1,vel_profile2,vel_profile3, error, error_velocity_blending, vels


def plot_velocity_blending(which_gait,which_file,velocities):
    which_vel = velocities[which_gait]

    vel_sim,vel_profile1,vel_profile2,vel_profile3, error, error_velocity_blending,vels = velocity_blending(which_gait,which_file)

    vels_sim_plot =vel_sim[0::int((len(vel_profile1)/len(vel_sim)))]
    ideal_plt = which_vel* np.ones(len(vel_sim))

    which_vel = [velocities[vels[0]], velocities[vels[1]]]
    x_axis = np.arange(0,len(vels_sim_plot))

    ind1 = np.random.randint(0, len(velocities), )
    ind2 = np.random.randint(0, len(velocities), )

    # velocities corresponding to these indeces
    vel1 = 0.2 #velocities[ind1]
    vel2 = 0.5 #velocities[ind2]

    w = 0.1
    D = np.linspace(0, 2, len(vel_sim))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels1 = vel1 + (vel2 - vel1) * (1 - sigmaD)

    w = 0.25
    D = np.linspace(0, 2, len(vel_sim))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels2 = vel1 + (vel2 - vel1) * (1 - sigmaD)

    w = 0.01
    D = np.linspace(0, 2, len(vel_sim))
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels3 = vel1 + (vel2 - vel1) * (1 - sigmaD)



    plt.figure()
    plt.plot(x_axis, vels3)  # vel_profile2,vel_profile3)
    plt.plot(x_axis,vels1) #vel_profile2,vel_profile3)
    plt.title('Velocity Profile for Velocity Transition between '+str(which_vel[0])+'m/s and '+str(which_vel[1])+'m/s')
    plt.title('Velocity Profile for Velocity Transition between ' + str(vel1) + 'm/s and ' + str(vel2) + 'm/s')

    plt.ylabel('Commanded Velocity x (m/s)')
    plt.xlabel('Episode')
    #plt.legend(('w=0.01','w=0.1','w=0.25'))
    #plt.legend(('Simulation','Commanded'))
    plt.show()

def main():
    which_gait = 3
    which_file = 'p'
    load_sim(which_file)
    load_mpc()
    get_gaits_plot(50,250,which_gait,which_file,velocities)


if __name__ == "__main__":
    main()

