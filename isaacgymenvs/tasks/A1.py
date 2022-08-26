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
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["knee_collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["foot_contact"] = self.cfg["env"]["learn"]["footcontactRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["gait"] = self.cfg["env"]["learn"]["gaitRewardScale"]

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

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # M: push interval
        # self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.allow_knee_contatcs = self.cfg["env"]["learn"]["allowKneeContacts"]


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

        #M: Push Robot
        self.common_step_counter = 0

        #M: Noise Vector
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

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
                             "action_rate": torch_zeros(), "hip": torch_zeros(), "gait": torch_zeros(), "foot_contact": torch_zeros(), "rew_buf": torch_zeros()}


        ############################### Mania Gaits #######################################

        # #Load MPC DATA
        # self.load_mpc_390()
        #
        # #Velocity selection
        # self.velocities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).cuda(0)
        # num_vel = len(self.velocities)
        #
        # #Gait MPC Tensor :
        # # self.gaits = torch.tensor(np.array([self.ct_tp_01, self.ct_tp_02, self.ct_tp_03, self.ct_tp_04,self.ct_tp_05, self.ct_tp_05, self.ct_tt_07, self.ct_tt_07, self.ct_tt_09, self.ct_tt_10])).cuda(0)
        # self.gaits = torch.tensor(np.array(
        #     [self.ct_tt_06, self.ct_tt_06, self.ct_tt_06, self.ct_tt_06, self.ct_tt_06, self.ct_tt_06, self.ct_tt_06,
        #      self.ct_tt_06, self.ct_tt_06, self.ct_tt_06])).cuda(0)
        #
        # # MPC TIME STEPS
        # self.num_ts = len(self.ct_tt_06)
        #
        # # Gait in right form
        # self.gait = torch.zeros(self.num_envs, self.num_ts, 4).cuda(0)
        #
        # # select random velocity for each environemnt
        # self.ind = torch.randint(0, num_vel - 1, (self.num_envs,)).cuda(0)  # num_vel-1
        # #random vel
        # self.commands[:, 0] = torch.index_select(self.velocities, 0, self.ind)
        # # corresponding gait
        # self.gait = torch.index_select(self.gaits, 0, self.ind)
        #
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    #M: Added noise function
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
             [NOTE]: Must be adapted when changing the observations structure

         Args:
             cfg (Dict): Environment config file

         Returns:
             [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
         """

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        # noise_vec[36:176] = self.cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[36:48] = 0.  # previous actions
        return noise_vec


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

        if asset_options.collapse_fixed_joints:
            self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], "base")
        else:
            self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.A1_handles[0], "trunk")

    #M: Push Robot
    def push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

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

        # #M: Push Robot
        # if self.common_step_counter % self.push_interval == 0:
        #     self.push_robots()

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

        #M: Add noise
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

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

        ### Reward: Foot Air Time
        rew_foot_air_time = self._get_reward_foot_air_time() * self.rew_scales["air_time"]
        #print(rew_foot_air_time)

        # ### Reward: Knee collision
        # rew_knee_collision = self._get_knee_collision_reward() * self.rew_scales["knee_collision"]
        #
        # ### Reward: Action rate
        # rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        #
        # ### Reward: Foot contact
        # rew_foot_contact = self._get_foot_contact_reward() * self.rew_scales["foot_contact"]
        #
        # ### Reward: Gait
        # rew_gait = self._get_gait_reward() * self.rew_scales["gait"]
        #
        # ### Reward: Hip
        # rew_hip = self._get_reward_hip() * self.rew_scales["hip"]
        # #print(rew_hip[0])

        # total reward buffer
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
              rew_torque + rew_joint_acc #+ rew_action_rate  + rew_foot_contact +rew_gait + rew_hip +rew_foot_air_time +rew_knee_collision

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["base_height"] += rew_joint_acc
        self.episode_sums["air_time"] += rew_foot_air_time
        # self.episode_sums["knee_collision"] += rew_joint_acc
        # self.episode_sums["action_rate"] += rew_action_rate
        # self.episode_sums["foot_contact"] += rew_foot_contact
        # self.episode_sums["gait"] += rew_gait
        # self.episode_sums["hip"] += rew_hip
        self.episode_sums["rew_buf"] += self.rew_buf

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
        # self.stance_step_counter[env_ids] = 0.

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

    def _get_gait_reward(self):

        gait_reward = torch.zeros((self.num_envs),dtype=torch.float, device=self.device, requires_grad=False)

        for l in self.L:
            gait_reward[:] += torch.abs(self.dof_pos[:,l[0]] - self.dof_pos[:,l[1]])
            
        return gait_reward

    def _get_reward_hip(self):
        
        hip_reward = torch.sum(torch.abs(self.default_dof_pos[:,self.H] - self.dof_pos[:,self.H]),dim=1)

        return hip_reward

    def _reset_dofs(self,env_ids):

        #positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] #* positions_offset
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
        # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        #
        # # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    #M Load MPC Data
    def load_mpc(self):
            # load data from MPC
        path = "/home/maria/motion_imitation/"
        with np.load(path + "footsteps_mpc.npz") as target_mpc:
            self.target_mpc = target_mpc["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts.npz") as conatct_mpc:
            self.contacts = conatct_mpc["feet_contacts"]

        # load MPC gaits
        ########### tripod
        with np.load(path + "foot_contacts01tp.npz") as ct_tp_01:
            self.ct_tp_01 = ct_tp_01["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts02tp.npz") as ct_tp_02:
            self.ct_tp_02 = ct_tp_02["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts03tp.npz") as ct_tp_03:
            self.ct_tp_03 = ct_tp_03["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts04tp.npz") as ct_tp_04:
            self.ct_tp_04 = ct_tp_04["feet_contacts"]
        with np.load(path + "foot_contacts05tp.npz") as ct_tp_05:
            self.ct_tp_05 = ct_tp_05["feet_contacts"]

        ########### trotting
        with np.load(path + "foot_contacts06tt.npz") as ct_tt_06:
            self.ct_tt_06 = ct_tt_06["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts07tt.npz") as ct_tt_07:
            self.ct_tt_07 = ct_tt_07["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts08tt.npz") as ct_tt_08:
            self.ct_tt_08 = ct_tt_08["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts09tt.npz") as ct_tt_09:
            self.ct_tt_09 = ct_tt_09["feet_contacts"]
        with np.load(path + "foot_contacts10tt.npz") as ct_tt_10:
            self.ct_tt_10 = ct_tt_10["feet_contacts"]

        # load MPC footsteps
        ########### trotting
        with np.load(path + "footsteps_mpc01tp.npz") as f_tp_01:
            self.f_tp_01 = f_tp_01["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc02tp.npz") as f_tp_02:
            self.f_tp_02 = f_tp_02["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc03tp.npz") as f_tp_03:
            self.f_tp_03 = f_tp_03["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc04tp.npz") as f_tp_04:
            self.f_tp_04 = f_tp_04["footsteps"]
        with np.load(path + "footsteps_mpc05tp.npz") as f_tp_05:
            self.f_tp_05 = f_tp_05["footsteps"]

        with np.load(path + "footsteps_mpc06tt.npz") as f_tt_06:
            self.f_tt_06 = f_tt_06["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc07tt.npz") as f_tt_07:
            self.f_tt_07 = f_tt_07["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc08tt.npz") as f_tt_08:
            self.f_tt_08 = f_tt_08["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc09tt.npz") as f_tt_09:
            self.f_tt_09 = f_tt_09["footsteps"]
        with np.load(path + "footsteps_mpc10tt.npz") as f_tt_10:
            self.f_tt_10 = f_tt_10["footsteps"]

    def load_mpc_390(self):
        # load data from MPC
        path = "/home/robohike/motion_imitation/"
        with np.load(path + "footsteps_mpc.npz") as target_mpc:
            self.target_mpc = target_mpc["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts.npz") as conatct_mpc:
            self.contacts = conatct_mpc["feet_contacts"]

        # load MPC gaits
        ########### tripod
        with np.load(path + "foot_contacts01tp.npz") as ct_tp_01:
            self.ct_tp_01 = ct_tp_01["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts02tp.npz") as ct_tp_02:
            self.ct_tp_02 = ct_tp_02["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts03tp.npz") as ct_tp_03:
            self.ct_tp_03 = ct_tp_03["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts04tp.npz") as ct_tp_04:
            self.ct_tp_04 = ct_tp_04["feet_contacts"]
        with np.load(path + "foot_contacts05tp.npz") as ct_tp_05:
            self.ct_tp_05 = ct_tp_05["feet_contacts"]

        ########### trotting
        with np.load(path + "foot_contacts06tt.npz") as ct_tt_06:
            self.ct_tt_06 = ct_tt_06["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts07tt.npz") as ct_tt_07:
            self.ct_tt_07 = ct_tt_07["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts08tt.npz") as ct_tt_08:
            self.ct_tt_08 = ct_tt_08["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts09tt.npz") as ct_tt_09:
            self.ct_tt_09 = ct_tt_09["feet_contacts"]
        with np.load(path + "foot_contacts10tt.npz") as ct_tt_10:
            self.ct_tt_10 = ct_tt_10["feet_contacts"]

        # load MPC footsteps
        ########### trotting
        with np.load(path + "footsteps_mpc01.npz") as f_tp_01:
            self.f_tp_01 = f_tp_01["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc02tp.npz") as f_tp_02:
            self.f_tp_02 = f_tp_02["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc03tp.npz") as f_tp_03:
            self.f_tp_03 = f_tp_03["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc04tp.npz") as f_tp_04:
            self.f_tp_04 = f_tp_04["footsteps"]
        with np.load(path + "footsteps_mpc05tp.npz") as f_tp_05:
            self.f_tp_05 = f_tp_05["footsteps"]

        with np.load(path + "footsteps_mpc06tt.npz") as f_tt_06:
            self.f_tt_06 = f_tt_06["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc07tt.npz") as f_tt_07:
            self.f_tt_07 = f_tt_07["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc08tt.npz") as f_tt_08:
            self.f_tt_08 = f_tt_08["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "footsteps_mpc09tt.npz") as f_tt_09:
            self.f_tt_09 = f_tt_09["footsteps"]
        with np.load(path + "footsteps_mpc10tt.npz") as f_tt_10:
            self.f_tt_10 = f_tt_10["footsteps"]

        
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
