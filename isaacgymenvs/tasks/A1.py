# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
import time

class A1(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.user_command_scale = self.cfg["env"]["learn"]["userCommandScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["gait"] = self.cfg["env"]["learn"]["gaitRewardScale"]
        self.rew_scales["gait_period"] = self.cfg["env"]["learn"]["gaitPeriodRewardScale"]
        self.rew_scales["gait_trajectory"] = self.cfg["env"]["learn"]["gaitTrajectoryRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        
        
        # self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        # self.rew_scales["knee_collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        # self.rew_scales["foot_contact"] = self.cfg["env"]["learn"]["footcontactRewardScale"]
        # self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.actions_stored = []
        


        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        # M: push interval
       
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.allow_knee_contatcs = self.cfg["env"]["learn"]["allowKneeContacts"]
        #self.Kp = self.cfg["env"]["control"]["stiffness"]
        #self.Kd = self.cfg["env"]["control"]["damping"]

        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        #torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.rb_obj = self.rb_state.view(self.num_envs, self.num_bodies, 13)
        self.rb_pos = self.rb_obj[:, :, :3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        #self.last_contacts = self.contact_forces[:,self.feet_indices,2].clone()
        #self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # Structure for reward logging
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(),
                             "ang_vel_xy": torch_zeros(), "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(), "air_time": torch_zeros(), "knee_collision": torch_zeros(),
                             "action_rate": torch_zeros(), "hip": torch_zeros(), "gait": torch_zeros(), "foot_contact": torch_zeros(),
                             "gait_period": torch_zeros(),"gait_trajectory": torch_zeros()}

        #iteration counter
        self.iteration_index = torch.zeros(self.num_envs, device=self.device)

        #push robot 
        #M: Push Robot
        self.common_step_counter = 0



        
        ## load data from MPC
        self.load_mpc_390()
        self.num_legs = 4
        ################################ Period Reward #############################
        #period reward tensors initialise
        self.init_done = True
        self.time = 0
        self.cont_time = []
        self.contact_when = None
        self.gait_cycle = torch.zeros(10000)
        self.stance_contact_counter = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.stance_updated_counter = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.swing_contact_counter = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.swing_updated_counter = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.diff = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.diff_change = torch.zeros(self.num_envs,self.num_legs, device=self.device)
        self.stance_step_counter = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.check_if_end = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.indx_update = torch.zeros(self.num_envs, self.num_legs, device=self.device)
        self.steping = True


        ####### Storing MPC Values

        ##Period
        self.load_periods = torch.tensor(np.array(
            [self.period1, self.period2, self.period3, self.period4, self.period5, self.period6, self.period7,
             self.period8, self.period9])).cuda(0)

        ## Gait Contacts
        self.load_gaits = torch.tensor(np.array(
            [self.ct_tp_01, self.ct_tp_02, self.ct_tp_03, self.ct_tp_04, self.ct_tp_05, self.ct_tt_06, self.ct_tt_07,
             self.ct_tt_08, self.ct_tt_09])).cuda(0)

        ## Positions
        self.load_calf = torch.tensor(np.array(
            [self.calf1[:, :, :], self.calf2[:, :, :], self.calf3[:, :, :], self.calf4[:, :, :], self.calf5[:, :, :],
             self.calf6[:, :, :], self.calf7[:, :, :], self.calf8[:, :, :], self.calf9[:, :, :]])).cuda(0)

        self.load_hip = torch.tensor(np.array(
            [self.hip1[:, :, :], self.hip2[:, :, :], self.hip3[:, :, :], self.hip4[:, :, :], self.hip5[:, :, :],
             self.hip6[:, :, :], self.hip7[:, :, :], self.hip8[:, :, :], self.hip9[:, :, :]])).cuda(0)

        self.load_foot = torch.tensor(np.array(
            [self.f_tp_01[:, :, :], self.f_tp_02[:, :, :], self.f_tp_03[:, :, :], self.f_tp_04[:, :, :],
             self.f_tp_05[:, :, :], self.f_tt_06[:, :, :], self.f_tt_07[:, :, :], self.f_tt_08[:, :, :],
             self.f_tt_09[:, :, :]])).cuda(0)
        
        check = len(self.load_foot)

        # velociy blending
        self.velocities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).cuda(0)


        self.num_vel = len(self.velocities)
        self.num_ts = len(self.ct_tp_02) +2
        self.num_xyz = 3

        self.gaits= torch.cat((self.load_gaits, torch.ones(len(self.load_gaits),2,self.num_legs, device=self.device)), 1).to(torch.bool)
        self.calf = torch.cat((self.load_calf, torch.ones(len(self.load_hip),2,self.num_legs,self.num_xyz, device=self.device)), 1)
        self.hip = torch.cat((self.load_hip, torch.ones(len(self.load_hip), 2, self.num_legs,self.num_xyz, device=self.device)), 1)
        self.foot = torch.cat((self.load_foot, torch.ones(len(self.load_hip), 2, self.num_legs,self.num_xyz, device=self.device)), 1)

        self.gait = torch.zeros(self.num_envs, self.num_ts, self.num_legs, dtype=torch.bool).cuda(0)
        self.period = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device, requires_grad=False)
        self.hips = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64, device=self.device)
        self.calves = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz,dtype=torch.float64, device=self.device)
        self.feet = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64,device=self.device)

        self.enable_gait = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)
        #self.enable_trajectory = torch.zeros(3, self.num_envs, self.num_legs,self.num_xyz,  dtype=torch.bool, device=self.device )

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/unitree_a1/urdf/a1_unitree_modified.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        A1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(A1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(A1_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(A1_asset)
        self.dof_names = self.gym.get_asset_dof_names(A1_asset)
        extremity_name = "calf" if asset_options.collapse_fixed_joints else "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if s.endswith("thigh")]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        hip_names = [s for s in body_names if s.endswith("hip")]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(A1_asset)
        #for i in range(self.num_dof):
            #dof_props['driveMode'][i] = self.cfg["env"]["control"]["driveMode"]
            #dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            #dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.A1_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            A1_handle = self.gym.create_actor(env_ptr, A1_asset, start_pose, "A1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, A1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, A1_handle)
            self.envs.append(env_ptr)
            self.A1_handles.append(A1_handle)

        # L structure for gait rewards
        self.L = [[self.dof_names.index('FL_thigh_joint'),self.dof_names.index('RR_thigh_joint')],
                    [self.dof_names.index('FL_calf_joint'),self.dof_names.index('RR_calf_joint')],
                    [self.dof_names.index('FR_thigh_joint'),self.dof_names.index('RL_thigh_joint')],
                    [self.dof_names.index('FR_calf_joint'),self.dof_names.index('RL_calf_joint')]]

        # H structure for hip rewards
        self.H = [self.dof_names.index('FL_hip_joint'),
                    self.dof_names.index('RR_hip_joint'),
                    self.dof_names.index('FL_hip_joint'),
                    self.dof_names.index('RR_hip_joint')]


        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], knee_names[i])
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0],
                                                                        hip_names[i])

        if asset_options.collapse_fixed_joints:
            self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], "base")
        else:
            self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], "trunk")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        for _ in range(self.decimation):
            torques = self.action_scale * self.actions
            torques = torch.clip(torques, -self.clip_actions, self.clip_actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)

        self.progress_buf += 1

        #M: Push Robot
        self.common_step_counter += 1
        self.randomize_buf += 1
        self.iteration_index += 1


        #M: Push Robot
        if self.common_step_counter % self.push_interval == 0:
            self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Update the reset buffer
        self._check_termination()

        # Calculate the rewards
        self.compute_reward()

        # Reset the agents that need termination
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Get observations
        self.compute_observations()
        
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:] 


    def compute_reward(self):

        # prepare quantities (TODO: return from obs ?)
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        projected_gravity = quat_rotate(base_quat, self.gravity_vec)

        #TODO check the height
        target_height = 0.35

        #print(self.root_states[0, 2])

        ### Reward: XY Linear Velocity (difference between base and input command)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]

        ### Reward: Z Linear Velocity 
        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]

        ### Reward: Around XY Angular Velocity 
        rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        ### Reward: Around Z Angular Velocity (difference between base and input command)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        ### Reward: Orientation
        rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        ### Reward: Torque Penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        ### Reward: Joint Acceleration Penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        ### Reward: Base height Penalty
        rew_base_height = torch.square(self.root_states[:, 2] -  target_height) * self.rew_scales["base_height"]

        ### Reward: Action rate
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # ##Reward: Gait
        rew_gait= self._get_gait_reward() * self.rew_scales["gait"]

        ##Reward: Gait Period
        rew_gait_period= self._get_gait_period_reward() * self.rew_scales["gait_period"]

        ##Reward: Gait Trajectory:
        rew_gait_trajectory = self._get_gait_trajectory_reward() * self.rew_scales['gait_trajectory']

        # ### Reward: Foot Air Time
        # rew_foot_air_time = self._get_reward_foot_air_time() * self.rew_scales["air_time"]
        # #print(rew_foot_air_time)
        #
        # ### Reward: Knee collision
        # rew_knee_collision = self._get_knee_collision_reward() * self.rew_scales["knee_collision"]
        #
        #
        # ### Reward: Foot contact
        # rew_foot_contact = self._get_foot_contact_reward() * self.rew_scales["foot_contact"]
        #
        # ### Reward: Gait
        #rew_gait = self._get_gait_reward() * self.rew_scales["gait"]
        #
        # ### Reward: Hip
        # rew_hip = self._get_reward_hip() * self.rew_scales["hip"]
        # #print(rew_hip[0])


        # total reward buffer
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
            rew_torque + rew_joint_acc + rew_action_rate + rew_gait_period + rew_gait +rew_gait_trajectory #+ rew_knee_collision +rew_foot_contact +rew_gait + rew_hip  +rew_foot_air_time

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["base_height"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["gait_period"] += rew_gait_period
        self.episode_sums["gait"] += rew_gait
        self.episode_sums["gait_trajectory"] += rew_gait_trajectory


        #self.episode_sums["air_time"] += rew_foot_air_time
        #self.episode_sums["knee_collision"] += rew_knee_collision
        # self.episode_sums["foot_contact"] += rew_foot_contact
        #self.episode_sums["gait"] += rew_period
        # self.episode_sums["hip"] += rew_hip

    def compute_observations(self):

        self.obs_buf[:] = compute_A1_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.last_actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        self.user_command_scale
        )

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # Reset agents
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Generate new commands
        self._resample_commands(env_ids)

        # Reset data buffers
        self.progress_buf[env_ids] = 0.
        self.reset_buf[env_ids] = 1.
        self.feet_air_time[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.iteration_index[env_ids] = 0.

        # Register individual reward data for logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

    def _check_termination(self):

        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=-1) > 1.
        time_out = self.progress_buf > self.max_episode_length
        reset_buf = reset | time_out

        if not self.allow_knee_contatcs:
            reset = torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
            reset_buf = reset | reset_buf

        self.reset_buf = reset_buf

    #M: Push Robot
    def push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        self.root_states[:, 7:9] = torch_rand_float(-0.3, 0.8, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))




    ############################# Reward Functions Definition
    def _get_reward_foot_air_time(self):
  
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        return rew_airTime

    def _get_knee_collision_reward(self):
        knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        knee_reward = torch.sum(knee_contact,dim=1)
        
        return knee_reward

    def _get_foot_contact_reward(self):
        feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        #feet_contact_filter = torch.logical_or(feet_contact,self.last_contacts)
        feet_contact_flipped = ~feet_contact
        feet_contact_reward = torch.sum(feet_contact_flipped,dim=1)
        return feet_contact_reward
    # 
    # def _get_gait_reward(self):
    #
    #     gait_reward = torch.zeros((self.num_envs),dtype=torch.float, device=self.device, requires_grad=False)
    #
    #     for l in self.L:
    #         gait_reward[:] += torch.abs(self.dof_pos[:,l[0]] - self.dof_pos[:,l[1]])
    #
    #     return gait_reward

    def _get_reward_hip(self):
        
        hip_reward = torch.sum(torch.abs(self.default_dof_pos[:,self.H] - self.dof_pos[:,self.H]),dim=1)

        return hip_reward

    def _get_gait_period_reward(self):

        ''' Difference between simulation step period and desired MPC step period'''

        mpc_period = self._gait_period()
        rew_gait_period = torch.sum(torch.abs(mpc_period - self.period) , dim=-1)

        return rew_gait_period
    #
    def _get_gait_reward(self):

        ''' Difference between simulation feet contacts and desired MPC foot contacts'''

        # gait forces from  MPC
        mpc_contacts = self._get_mpc_to_sim()

        # forces from sim
        self.feet_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.

        # self.footprint_draw()

        #enabling reward after spawning
        self.enabling_rewards(mpc_contacts)

        desired_mpc_enabled = torch.zeros_like(mpc_contacts, device=self.device)
        desired_mpc_enabled = torch.where(self.enable_gait, mpc_contacts.long(), 0)
        
        # make bool to int
        sim_contacts = self.feet_contacts.long()

        # difference between desired and sim
        diff = torch.abs(desired_mpc_enabled - sim_contacts)
        rew_gait = torch.sum(diff, dim=-1)
        return rew_gait

    def _get_gait_trajectory_reward(self):

        ''' Compute trajectory reard by applying Eucledian difference between mpc and sim positions '''

        mpc_trajectory, sim_trajectory = self._get_gait_trajectory()

        # enabling reward after spawning
        self.enabling_rewards(mpc_trajectory)

        mpc_trajectory_enabled = torch.zeros_like(mpc_trajectory, dtype=torch.float32, device=self.device)
        mpc_trajectory_enabled = torch.where(self.enable_trajectory, mpc_trajectory, mpc_trajectory_enabled)

        trajectory_reward  = torch.sum(torch.sum(torch.sqrt(torch.sum(torch.square(mpc_trajectory_enabled - sim_trajectory),
                                                           dim=-1)), dim=0), dim=-1)
        return trajectory_reward




    #################### Reset Functions
    def _reset_dofs(self,env_ids):

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        self.ind = torch.randint(0, self.num_vel, (len(env_ids),)).cuda(0)
        self.commands[env_ids, 0] = torch.index_select(self.velocities, 0, self.ind)

        # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1)


        # choose period according to velocity
        self.period_select = torch.index_select(self.load_periods, 0, self.ind)  # corresponding period
        self.period[env_ids] = torch.unsqueeze(self.period_select, dim=1).expand(-1, 4).to(torch.float32)

        # corresponding gait
        self.gait[env_ids]= torch.index_select(self.gaits, 0, self.ind)

        # corresponding trajectory positions
        self.calves[env_ids] = torch.index_select(self.calf,0,self.ind).cuda(0)
        self.hips[env_ids] = torch.index_select(self.hip, 0, self.ind).cuda(0)
        self.feet[env_ids] = torch.index_select(self.foot, 0, self.ind).cuda(0)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    #################### MPC Functions
    def load_mpc_390(self):
        # load data from MPC
        path = "/home/robohike/motion_imitation/2500/"

        # load MPC gaits
        ########### tripod


        with np.load(path + "foot_contacts1.npz") as ct_tp_01:
            self.ct_tp_01 = ct_tp_01["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts1.npz") as ct_tp_01:
            self.ct_tp_01 = ct_tp_01["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts2.npz") as ct_tp_02:
            self.ct_tp_02 = ct_tp_02["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts3.npz") as ct_tp_03:
            self.ct_tp_03 = ct_tp_03["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts4.npz") as ct_tp_04:
            self.ct_tp_04 = ct_tp_04["feet_contacts"]
        with np.load(path + "foot_contacts5.npz") as ct_tp_05:
            self.ct_tp_05 = ct_tp_05["feet_contacts"]
        with np.load(path + "foot_contacts6.npz") as ct_tt_06:
            self.ct_tt_06 = ct_tt_06["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts7.npz") as ct_tt_07:
            self.ct_tt_07 = ct_tt_07["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts8.npz") as ct_tt_08:
            self.ct_tt_08 = ct_tt_08["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts9.npz") as ct_tt_09:
            self.ct_tt_09 = ct_tt_09["feet_contacts"]
        # with np.load(path + "foot_contacts10tt.npz") as ct_tt_10:
        #     self.ct_tt_10 = ct_tt_10["feet_contacts"]

        # load MPC footsteps
        ########### trotting

        with np.load(path + "hip1.npz") as hip1:
            self.hip1 = hip1["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf1.npz") as calf1:
            self.calf1 = calf1["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip2.npz") as hip2:
            self.hip2 = hip2["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf2.npz") as calf2:
            self.calf2 = calf2["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip3.npz") as hip3:
            self.hip3 = hip3["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf3.npz") as calf3:
            self.calf3 = calf3["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip4.npz") as hip4:
            self.hip4 = hip4["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf4.npz") as calf4:
            self.calf4 = calf4["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip5.npz") as hip5:
            self.hip5 = hip5["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf5.npz") as calf5:
            self.calf5 = calf5["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip6.npz") as hip6:
            self.hip6 = hip6["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf6.npz") as calf6:
            self.calf6 = calf6["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip7.npz") as hip7:
            self.hip7 = hip7["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf7.npz") as calf7:
            self.calf7 = calf7["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip8.npz") as hip8:
            self.hip8 = hip8["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf8.npz") as calf8:
            self.calf8 = calf8["calf_pos"]  # [n_time_steps, feet indices, xyz]

        with np.load(path + "hip9.npz") as hip9:
            self.hip9 = hip9["hip_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "calf9.npz") as calf9:
            self.calf9 = calf9["calf_pos"]  # [n_time_steps, feet indices, xyz]

        ################ Period

        with np.load(path + "period1.npz") as period1:
            self.period1 = period1["period"]
        with np.load(path + "period2.npz") as period2:
            self.period2 = period2["period"]
        with np.load(path + "period3.npz") as period3:
            self.period3 = period3["period"]
        with np.load(path + "period4.npz") as period4:
            self.period4 = period4["period"]
        with np.load(path + "period5.npz") as period5:
            self.period5 = period5["period"]
        with np.load(path + "period6.npz") as period6:
            self.period6 = period6["period"]
        with np.load(path + "period7.npz") as period7:
            self.period7 = period7["period"]
        with np.load(path + "period8.npz") as period8:
            self.period8 = period8["period"]
        with np.load(path + "period9.npz") as period9:
            self.period9 = period9["period"]

        # load MPC footsteps
        ########### trotting
        with np.load(path + "foot1.npz") as f_tp_01:
            self.f_tp_01 = f_tp_01["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot2.npz") as f_tp_02:
            self.f_tp_02 = f_tp_02["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot3.npz") as f_tp_03:
            self.f_tp_03 = f_tp_03["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot4.npz") as f_tp_04:
            self.f_tp_04 = f_tp_04["foot_pos"]
        with np.load(path + "foot5.npz") as f_tp_05:
            self.f_tp_05 = f_tp_05["foot_pos"]

        with np.load(path + "foot6.npz") as f_tt_06:
            self.f_tt_06 = f_tt_06["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot7.npz") as f_tt_07:
            self.f_tt_07 = f_tt_07["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot8.npz") as f_tt_08:
            self.f_tt_08 = f_tt_08["foot_pos"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot9.npz") as f_tt_09:
            self.f_tt_09 = f_tt_09["foot_pos"]
        with np.load(path + "footsteps_mpc10tt.npz") as f_tt_10:
            self.f_tt_10 = f_tt_10["footsteps"]

    def _gait_period(self):
        '''Check how long it took for the reference foot to complete a full gait cycle/period.'''

        # contact of the reference foot
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.


        # contact counter for swing
        self.swing_updated_counter = torch.where(contact, self.swing_updated_counter, self.swing_updated_counter + 1)

        # contact counter for stance
        self.stance_updated_counter = torch.where(contact, self.stance_updated_counter + 1, self.stance_updated_counter)

        # see if phase changed - from swing to stance
        diff_stance = self.stance_updated_counter - self.stance_contact_counter

        # STORE OVERALL VALUES 
        self.stance_contact_counter = self.stance_updated_counter
        self.swing_contact_counter = self.swing_updated_counter

        # step: diff goes from 0 to 1
        self.diff_change = torch.where(diff_stance > 0., self.diff_change + 1,
                                       torch.zeros_like(self.diff_change).cuda(0))  # check if there is change in phase

        # check when  the step happened

        steping_condition1 = self.steping != contact
        steping_condition2 = self.diff < self.diff_change

        # times of change of variable from 0 to 1: step
        self.stance_step_counter = torch.where(torch.logical_and(steping_condition1, steping_condition2),
                                               self.stance_step_counter + 1,
                                               self.stance_step_counter)

        self.stance_step_counter = self.stance_step_counter
        self.diff = self.diff_change
        self.steping = contact

        ## cycle ends after 2 steps
        end_cycle = (self.stance_step_counter % 2) == 0  # 1,3,5,... take only every 2 steps
        # get index of second step

        condition1 = self.check_if_end < self.stance_step_counter
        condition2 = end_cycle == 0
        condition = torch.logical_and(condition2, condition1)

        period_so_far = self.swing_updated_counter + self.stance_updated_counter

        # self.cycle_index = torch.where(condition == 2*torch.ones(self.num_envs).cuda(0),self.swing_updated_counter,torch.zeros(self.num_envs).cuda(0)) #index cycle ends
        self.cycle_index = torch.where(condition1, period_so_far, self.indx_update)  # index cycle ends

        self.indx_update = self.cycle_index

        self.check_if_end = self.stance_step_counter

        # only compute reward when there is a full cycle

        self.stance_updated_counter = torch.where(condition1, torch.zeros(self.num_envs, 4).cuda(0),
                                                  self.stance_updated_counter)

        self.swing_updated_counter = torch.where(condition1, torch.zeros(self.num_envs, 4).cuda(0),
                                                 self.swing_updated_counter)

        self.reward_scaling = torch.where(condition1, torch.ones(self.num_envs, 4).cuda(0),
                                          torch.zeros(self.num_envs, 4).cuda(0))

        ### index reset

        # self.iteration_index = torch.where(condition1,
        #                                    torch.zeros(self.num_envs,4).cuda(0) + self.till_contact,
        #                                    self.iteration_index)
        #
        # self.footstep_update = torch.where(condition1, self.cycle_index,
        #                                    self.footstep_update)

        ## Checking Period
        # print('contact', contact)
        # print('step', self.stance_step_counter)
        # print('cycle', self.cycle_index)
        # print('period', period_so_far)
        # print()

        return self.cycle_index


    def enabling_rewards(self,to_enable):

        '''Only enable MPC reward calculations when robot has first contact with the ground. Robot falls at the start
        with Issac and not with Pybullet, hence here will be a missmatch'''

        # contact counter for swing
        start_gait = torch.where(self.feet_contacts,1,0).to(torch.bool)
        self.enable_gait = torch.logical_or(self.enable_gait,start_gait)

        self.enable_traj= torch.unsqueeze(self.enable_gait,-1)
        self.enable_traje = torch.unsqueeze(self.enable_traj,0)
        self.enable_trajectory = self.enable_traje.expand(3,self.num_envs,self.num_legs,self.num_xyz)
        check =1 




    def _get_mpc_to_sim(self):
        ''' Only take one idx of the mpc gaits at a time for ech environemnt, corresponding to the simulation index'''
        
        test = self.iteration_index.expand(4, -1)
        test = test.t()
        test = test.to(torch.int64)
        ttest1 = torch.unsqueeze(test, 1)


        #mintigate the fact that the robot falls at the start
        # self.gait_downsampled[:, :self.till_contact, :] = torch.zeros(self.num_envs, self.till_contact,
        #                                                               len(self.feet_indices))
        # self.gait_downsampled[:, self.till_contact:, :] = self.gait[:, self.till_contact:, :]
        check = self.gait
        desired_mpc = torch.gather(self.gait, 1, ttest1.to(torch.int64))
        #desired_mpc = torch.gather(self.gait, 1, ttest1.to(torch.int64))
        desired_mpc = torch.squeeze(desired_mpc, 1)
        return desired_mpc

    def get_mpc_footstep(self,tensor):

        # get footstep for each env and account for time index

        len_idx = self.num_legs * 3 #x,y,z coordinates
        dof_sample = tensor.view(self.num_envs, self.num_ts, 12)

        test1 = self.iteration_index.expand(len_idx, -1)
        test1 = test1.t()
        test1 = test1.to(torch.int64)
        ttest2 = torch.unsqueeze(test1, 1)

        # desired_footstep = self.footsteps.reshape(self.num_envs, 10000, 12)
        desired_footstep = torch.gather(dof_sample, 1, ttest2.to(torch.int64))
        desired_footstep = torch.squeeze(desired_footstep, 1)
        # desired_footstep = desired_footstep.reshape(self.num_envs,4,3)
        return desired_footstep

    def _get_gait_trajectory(self):
        ''' Get Simulation and Traget hip,knee and feet values in the right format'''

        ## Reshaping tesnors into [num_envs,3,4] format to match the same of IsaacGym gnerated values
        hip_tr = self.get_mpc_footstep(self.hips)  # time step mpc position
        hip_target = hip_tr.view(self.num_envs, len(self.hip_indices), 3)  # match shape of sim

        knee_tr = self.get_mpc_footstep(self.calves)  # time step mpc position
        calf_target = knee_tr.view(self.num_envs, len(self.knee_indices), 3)  # match shape of sim

        foot_tr = self.get_mpc_footstep(self.feet)  # time step mpc position
        feet_target = foot_tr.view(self.num_envs, len(self.feet_indices), 3)  # match shape of sim

        # Setup hip,knee and feet target values into one tensor
        mpc_pos = torch.zeros(3, self.num_envs, len(self.knee_indices), 3, device=self.device, requires_grad=False)
        mpc_pos[0, :, :] = hip_target
        mpc_pos[1, :, :] = calf_target
        mpc_pos[2, :, :] = feet_target

        #Extracting hip, claf, feet position from simulation
        hip_sim = self.rb_pos[:, self.hip_indices, :]
        calf_sim = self.rb_pos[:, self.knee_indices, :]
        feet_sim = self.rb_pos[:, self.feet_indices, :]

        # Setup hip,knee and feet sim values into one tensor
        sim_pos = torch.zeros(3, self.num_envs, len(self.knee_indices), 3, device=self.device, requires_grad=False)
        sim_pos[0, :, :] = hip_sim
        sim_pos[1, :, :] = calf_sim
        sim_pos[2, :, :] = feet_sim

        return mpc_pos , sim_pos



    def distance_footstep(self, sim, target):
        diff = sim - target
        sqr = torch.square(diff)
        error = torch.sum(torch.sum(sqr, dim=1))
        return error

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_A1_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                last_actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale,
                                user_command_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, List[float]) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    #dof_pos_scaled = dof_pos * dof_pos_scale
    dof_vel_scaled = dof_vel * dof_vel_scale

    commands_scaled = commands*torch.tensor(user_command_scale, requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     dof_pos_scaled,
                     dof_vel_scaled,
                     commands_scaled,
                     last_actions
                     ), dim=-1)

    return obs

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles
