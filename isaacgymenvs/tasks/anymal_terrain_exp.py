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
import os, time
import csv

from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from tasks.base.vec_task import VecTask

import torch
from typing import Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logit


class AnymalTerrain(VecTask):

    ##################### Initalising variables from YAML files #################################

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # graphics_device_id
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.testing = False
        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False
        self.save_footstep = []
        self.save_torques = []
        self.save_com_vel = []
        self.save_it_id = []
        # normalization scaling for observation space
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
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

        # target height
        self.target_height = self.cfg["env"]["learn"]["target_height"]

        # torque clip range
        self.torque_clip = self.cfg["env"]["learn"]["torque_limit"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # time step and decimation
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]

        # episode length
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # push interval
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        # alloe knee contact
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]

        # pd control params
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        # Multiply rewards with time step
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.rb_obj = self.rb_state.view(self.num_envs, self.num_bodies, 13)
        self.rb_pos = self.rb_obj[:, :, :3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.iteration_index = torch.zeros(self.num_envs, device=self.device)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device, requires_grad=False, )
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # Initialise height points
        self.height_points = self.init_height_points()
        self.measured_heights = None

        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device,
                                                requires_grad=False)

        # set default joint positions
        for i in range(self.num_actions):
            name = self.dof_names[i]
            # Only set the provided joints (Rokas edit)
            if name in self.named_default_joint_angles:
                angle = self.named_default_joint_angles[name]
                self.default_dof_pos[:, i] = angle

        # reward episode sums for logging
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(),
                             "ang_vel_xy": torch_zeros(), "orient": torch_zeros(), "torques": torch_zeros(),
                             "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(), "air_time": torch_zeros(), "knee_collision": torch_zeros(),
                             "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "hip": torch_zeros(), "gait": torch_zeros(),
                             "foot_contact": torch_zeros()}



        self.init_done = True

        #load the mpc data
        self.load_mpc_data()

        gaits = np.array(
            [self.ct_tp_01, self.ct_tp_02, self.ct_tp_03, self.ct_tp_04, self.ct_tp_05, self.ct_tt_06, self.ct_tt_07,
             self.ct_tt_07, self.ct_tt_09, self.ct_tt_10])

        self.gaits = torch.tensor(gaits).cuda(0)

        foot = np.array(
            [self.f_tp_01[:, :, 1], self.f_tp_02[:, :, 1], self.f_tp_03[:, :, 1], self.f_tp_04[:, :, 1],
             self.f_tp_05[:, :, 1], self.f_tt_06[:, :, 1], self.f_tt_07[:, :, 1], self.f_tt_08[:, :, 1],
             self.f_tt_09[:, :, 1], self.f_tt_10[:, :, 1]])

        self.foot = torch.tensor(foot).cuda(0)


        self.velocities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


        self.ind = torch.randint(0, 8, (self.num_envs,)).cuda(0)
        self.ind2 = torch.randint(0, 8, (self.num_envs,)).cuda(0)

        ###### Training ###################
        self.velocitie = torch.tensor(self.velocities).cuda(0)

        self.gait = torch.zeros(self.num_envs, 10000, 4).cuda(0)
        self.footsteps = torch.zeros(self.num_envs, 10000, 4, 3).cuda(0)

        self.commands[:, 0] = torch.index_select(self.velocitie, 0, self.ind)
        self.gait = torch.index_select(self.gaits, 0, self.ind)
        self.footsteps = torch.index_select(self.foot, 0, self.ind)

        # self.gait = self.gaits[0]
        # desired_mpc = torch.unsqueeze(self.gait, 0)
        self.desired_mpc = self.gait.expand(self.num_envs, -1, -1)
        # self.footsteps = self.foot[0]

        ###### Testiing ###################

        if self.testing:
            vel1,vel2 = self.test_velocity_profile()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        """Create simulation parameters """

        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)


        terrain_type = self.cfg["env"]["terrain"]["terrainType"]  # plane - flat terrain/no curiculum

        if terrain_type == 'plane':
            self._create_ground_plane()
        elif terrain_type == 'trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

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
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
             """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
             Rokas-no """
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):

        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """

        # get robot urdf file
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Set asset options
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

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)  # num of dof - 12 from yaml file
        self.num_bodies = self.gym.get_asset_rigid_body_count(
            anymal_asset)  # num of bodies - 18 (12+ 4 feet + base + base)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(anymal_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')

        # initialise actor in sim with starting position
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # Get joint indices for collision
        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)

        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        hip_name = self.cfg["env"]["urdfAsset"]["hipName"]

        # Not necessary code (Rokas edit)
        # hip_name = self.cfg["env"]["urdfAsset"]["hipName"]

        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        hip_names = [s for s in body_names if hip_name in s]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)

        # env origins - add anymal handles ( = num_envs (4096) in simulation)
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"][
                                                                                    "numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,),
                                            device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,),
                                           device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(anymal_asset, rigid_shape_prop)
            anymal_handle = self.gym.create_actor(env_handle, anymal_asset, start_pose, "anymal", i, 0, 0)
            dof_props = self.gym.get_actor_dof_properties(env_handle, anymal_handle)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            self.envs.append(env_handle)
            self.anymal_handles.append(anymal_handle)

        # get joint indeces from the actual robot handle
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0],
                                                                         feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0],
                                                                         knee_names[i])
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0],
                                                                        hip_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base")

    def check_termination(self):
        """ Check if environments need to be reset """


        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :],
                                    dim=-1) > 1.  # reset if base index touches the ground
        # if not self.allow_knee_contacts:
        #     knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        #     self.reset_buf |= torch.any(knee_contact, dim=1)

        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        """ Computes observations """
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.height_meas_scale
        self.obs_buf = torch.cat((self.base_lin_vel * self.lin_vel_scale,
                                  self.base_ang_vel * self.ang_vel_scale,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  self.dof_pos * self.dof_pos_scale,
                                  self.dof_vel * self.dof_vel_scale,
                                  self.actions
                                  ), dim=-1)

    def get_mpc_gait(self):
        test = self.iteration_index.expand(4, -1)
        test = test.t()
        test = test.to(torch.int64)
        ttest1 = torch.unsqueeze(test, 1)

        desired_mpc = torch.gather(self.desired_mpc, 1, ttest1.to(torch.int64))
        desired_mpc = torch.squeeze(desired_mpc, 1)
        return desired_mpc

    def get_mpc_footstep(self):
        test1 = self.iteration_index.expand(4, -1)
        test1 = test1.t()
        test1 = test1.to(torch.int64)
        ttest2 = torch.unsqueeze(test1, 1)

        # For Stance
        # d_footsteps = self.desired_footsteps[:, 0, :, :]
        # d_footsteps = torch.unsqueeze(d_footsteps, 1)
        # d_footsteps = d_footsteps.expand(-1, 10000, -1, -1)
        # self.desired_footsteps = d_footsteps

        # desired_footstep = self.footsteps.reshape(self.num_envs, 10000, 4)
        desired_footstep = torch.gather(self.footsteps, 1, ttest2.to(torch.int64))
        desired_footstep = torch.squeeze(desired_footstep, 1)
        # desired_footstep = desired_footstep.reshape(self.num_envs,4,3)
        return desired_footstep

    def distance_footstep(self, mpc, sim):
        # we do not want the z values
        # mpcy = mpc[:,:,1]
        # mpcy_check = mpcy <1.3
        # mpcy = torch.where[mpcy_check,mpcy,1.30]
        # mpc_sum = mpc[:,:,1] + mpc[:,:,0] + mpc[:,:,2]
        # mpc_error= torch.sqrt(torch.square(mpc_sum))
        # sim_sum =sim[:,:,1] + sim[:,:,0] + sim[:,:,2]
        # sim_error = torch.sqrt(torch.square(sim_sum))
        # diff = mpc_error - sim_error

        sim = sim[:, :, 1]
        diff = mpc - sim

        sqr = torch.square(diff)
        d = torch.sqrt(sqr)
        return d

    def compute_reward(self):
        """  Calls each reward term which had a non-zero scale and adds each terms to the episode sums and to the total reward
        reward [num_envs,1] - [4096,1]
          """

        ### Reward: XY Linear Velocity (difference between base and input command)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]

        ### Reward: Z Linear Velocity 
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]

        ### Reward: Around XY Angular Velocity 
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        ### Reward: Around Z Angular Velocity (difference between base and input command)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        ### Reward: Orientation
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        ### Reward: Torque Penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        ### Reward: Joint Acceleration Penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        ### Reward: Base height Penalty
        rew_base_height = torch.square(self.root_states[:, 2] - self.target_height) * self.rew_scales[
            "base_height"]  # TODO add target base height to cfg

        ## Reward: Gait
        #gait forces from  MPC
        desired_mpc = self.get_mpc_gait()
        #forces from sim
        self.feet_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.

        # make bool to into
        desired_mpc = desired_mpc.long()
        self.feet_contact = self.feet_contacts.long()

        #difference between desired and sim
        diff = torch.abs(desired_mpc - self.feet_contact)
        rew_gait = torch.sum(diff, dim=-1) * self.rew_scales["gait"]


        # ## Reward: Footsteps
        # get mpc foootstep in right format
        #self.mpc_footsteps = self.get_mpc_footstep()
        # getting only y coordinate
        # get sim footsteps
        #self.foot_pos = self.rb_pos[:, self.feet_indices, :]
        # get distance between footsteps:
        #self.distance = self.distance_footstep(self.mpc_footsteps, self.foot_pos)
        #rew_hip = torch.sum(self.distance, dim=-1) * self.rew_scales["hip"]


        # total reward buffer
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                       rew_torque #+ rew_gait

        # add termination reward
        # self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        # self.episode_sums["knee_collision"] += rew_knee_collision
        # self.episode_sums["action_rate"] += rew_action_rate
        # self.episode_sums["air_time"] += rew_airTime
        # self.episode_sums["base_height"] += rew_base_height
        #self.episode_sums["hip"] += rew_hip
        #self.episode_sums["gait"] += rew_gait
        # self.episode_sums["foot_contact"] += rew_foot_contact

        # Then, to reload:
        if self.testing:
            self.test_save_rewards()


    def reset_idx(self, env_ids):
        """ Reset some environments.
                Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
                [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
                Logs episode info
                Resets some buffers

            Args:
                env_ids (list[int]): List of environment ids which must be reset
            """


        self.iteration_index[env_ids] = 0

        ##############################################################################################

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.ind = torch.randint(0, 8, (len(env_ids),)).cuda(0)
        self.commands[env_ids, 0] = torch.index_select(self.velocitie,0,self.ind).type(torch.FloatTensor).cuda(0)#torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.gait = torch.index_select(self.gaits,0,self.ind)

        #torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = 0 #torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = 0  #torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1],(len(env_ids), 1), device=self.device).squeeze()


        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        """ Updates terrain level for curiculum learning, DONT LOOK"""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (
                    distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

             Args:
                 actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
             """

        self.actions = actions.clone().to(self.device)  # Send actions to GPU
        for i in range(self.decimation):
            torques = torch.clip(self.action_scale * self.actions,
                                 - self.torque_clip, self.torque_clip)  # Calculate and clip torques[4096,12]
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):

        """ check terminations, compute observations and rewards
                  calls self._post_physics_step_callback() for common computations
                  calls self._draw_debug_vis() if needed
              """
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        self.iteration_index += 1

        # Push the robot


        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        it_id = self.iteration_index.type(torch.LongTensor)

        if self.testing:
            self.commands[:,0] = self.vels[it_id]
            print('Forward Velocity',self.commands[:,0] )

        self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        ############# Testing: save variables\
        #self.test_save_vals()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                               self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    ################################### Not important #########################################################
    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device,
                               requires_grad=False)  # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device,
                               requires_grad=False)  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def push_robot(self):
        if self.common_step_counter % self.push_interval == 0:
            self.push_robots()

    def old_gaits_footsteps(self):
        # gait structure
        gait_mpc = torch.from_numpy(self.contacts).cuda(0)
        desired_mpc = torch.unsqueeze(gait_mpc, 0)
        self.desired_mpc = desired_mpc.expand(self.num_envs, -1, -1)

        # footstep position
        target_mpcf = torch.from_numpy(self.target_mpc).cuda(0)
        mpc_footsteps = torch.unsqueeze(target_mpcf, 0)
        self.desired_footsteps = mpc_footsteps.expand(self.num_envs, -1, -1, -1)

    def load_mpc_data(self):
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

    def load_mpc_data_390(self):

        # load data from MPC
        with np.load("/home/robohike/motion_imitation/footsteps_mpc.npz") as target_mpc:
            self.target_mpc = target_mpc["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts.npz") as conatct_mpc:
            self.contacts = conatct_mpc["feet_contacts"]

        # load MPC gaits
        ########### trotting
        with np.load("/home/robohike/motion_imitation/foot_contacts01tp.npz") as ct_tp_01:
            self.ct_tp_01 = ct_tp_01["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts02tp.npz") as ct_tp_02:
            self.ct_tp_02 = ct_tp_02["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts03tp.npz") as ct_tp_03:
            self.ct_tp_03 = ct_tp_03["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts04tp.npz") as ct_tp_04:
            self.ct_tp_04 = ct_tp_04["feet_contacts"]
        with np.load("/home/robohike/motion_imitation/foot_contacts05tp.npz") as ct_tp_05:
            self.ct_tp_05 = ct_tp_05["feet_contacts"]

            ########### trotting
        with np.load("/home/robohike/motion_imitation/foot_contacts06tt.npz") as ct_tt_06:
            self.ct_tt_06 = ct_tt_06["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts07tt.npz") as ct_tt_07:
            self.ct_tt_07 = ct_tt_07["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts08tt.npz") as ct_tt_08:
            self.ct_tt_08 = ct_tt_08["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/foot_contacts09tt.npz") as ct_tt_09:
            self.ct_tt_09 = ct_tt_09["feet_contacts"]
        with np.load("/home/robohike/motion_imitation/foot_contacts10tt.npz") as ct_tt_10:
            self.ct_tt_10 = ct_tt_10["feet_contacts"]

        # load MPC gaits
        ########### trotting
        with np.load("/home/robohike/motion_imitation/footsteps_mpc01tp.npz") as f_tp_01:
            self.f_tp_01 = f_tp_01["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc02tp.npz") as f_tp_02:
            self.f_tp_02 = f_tp_02["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc03tp.npz") as f_tp_03:
            self.f_tp_03 = f_tp_03["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc04tp.npz") as f_tp_04:
            self.f_tp_04 = f_tp_04["footsteps"]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc05tp.npz") as f_tp_05:
            self.f_tp_05 = f_tp_05["footsteps"]

        with np.load("/home/robohike/motion_imitation/footsteps_mpc06tt.npz") as f_tt_06:
            self.f_tt_06 = f_tt_06["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc07tt.npz") as f_tt_07:
            self.f_tt_07 = f_tt_07["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc08tt.npz") as f_tt_08:
            self.f_tt_08 = f_tt_08["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc09tt.npz") as f_tt_09:
            self.f_tt_09 = f_tt_09["footsteps"]
        with np.load("/home/robohike/motion_imitation/footsteps_mpc10tt.npz") as f_tt_10:
            self.f_tt_10 = f_tt_10["footsteps"]

    def test_velocity_profile(self):
        #two random indeces
        ind1 = self.ind.cpu().numpy()
        ind2 = self.ind2.cpu().numpy()

        #velocities corresponding to these indeces
        vel1 = self.velocities[ind1]
        vel2 = self.velocities[ind2]

        w = 0.05
        D = np.linspace(0, 2, 2500)
        sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
        vels = vel1 + (vel2 - vel1) * (1 - sigmaD)

        self.vels = torch.tensor(vels).cuda(0)


        #Plot velocity profile
        # plt.plot(D, vels)
        # plt.show()
        return vel1,vel2

    def test_save_rewards(self):
        # foot_contacts = self.feet_contact.cpu()
        # foot_contacts = foot_contacts.numpy()
        self.save_footstep.append(self.feet_contact)
        self.save_com_vel.append(self.base_lin_vel)
        self.save_torques.append(self.torques)
        self.save_it_id.append(self.iteration_index)
        # print(self.save_torques)

        # f = open('/home/robohike/save_data/fc.csv', 'w')
        # writer = csv.writer(f)
        # df = pd.DataFrame(self.save_footstep)  # convert to a dataframe
        # df.to_csv(f)  # save to file
        # # df = pd.read_csv("testfile")
        #
        # com_vels = self.base_lin_vel[:, 0].cpu()
        # com_vels = com_vels.numpy()
        # self.save_com_vel.append(com_vels)
        # v = open('/home/robohike/save_data/vel.csv', 'w')
        # # writer = csv.writer(v)
        # dfv = pd.DataFrame(self.save_com_vel)  # convert to a dataframe
        # dfv.to_csv(v)  # save to file
        #
        # trq = self.torques.cpu()
        # trq = trq.numpy()
        # self.save_torques.append(trq)
        # t= open('/home/robohike/save_data/torq.csv', 'w')
        # # writer = csv.writer(v)
        # dfv = pd.DataFrame(self.save_torques)  # convert to a dataframe
        # dfv.to_csv(t)  # save to file

    def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None, ):
        """
        Generate a uniform noise terrain

        Parameters
            terrain (SubTerrain): the terrain
            min_height (float): the minimum height of the terrain [meters]
            max_height (float): the maximum height of the terrain [meters]
            step (float): minimum height change between two points [meters]
            downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

        """
        if downsampled_scale is None:
            downsampled_scale = terrain.horizontal_scale

        # switch parameters to discrete units
        min_height = int(min_height / terrain.vertical_scale)
        max_height = int(max_height / terrain.vertical_scale)
        step = int(step / terrain.vertical_scale)

        heights_range = np.arange(min_height, max_height + step, step)
        height_field_downsampled = np.random.choice(heights_range, (
        int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
            terrain.length * terrain.horizontal_scale / downsampled_scale)))

        x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
        y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

        f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

        x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
        y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
        z_upsampled = np.rint(f(y_upsampled, x_upsampled))

        terrain.height_field_raw += z_upsampled.astype(np.int16)
        return terrain

    def create_trimesh(self):
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = self.cfg["env"]["terrain"]["mapLength"]
        self.env_width = self.cfg["env"]["terrain"]["mapWidth"]
        self.proportions = [np.sum(self.cfg["env"]["terrain"]["terrainProportions"][:i + 1]) for i in
                            range(len(self.cfg["env"]["terrain"]["terrainProportions"]))]

        self.env_rows = self.cfg["env"]["terrain"]["numLevels"]
        self.env_cols = self.cfg["env"]["terrain"]["numTerrains"]
        # self.env_origin_y = (j + 0.5) * self.env_width
        # x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
        # x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
        # y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
        # y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
        # self.num_per_env = int(self.num_envs / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.heightfield = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.heightsamples = self.heightfield

        def new_sub_terrain(): return SubTerrain(width=self.tot_rows, length=self.tot_cols,
                                                 vertical_scale=self.vertical_scale,
                                                 horizontal_scale=self.horizontal_scale)

        '''
        subterrain = SubTerrain("terrain",
                            width=self.tot_rows,
                            length=self.tot_cols,
                            vertical_scale=self.vertical_scale,
                            horizontal_scale=self.horizontal_scale)
        '''
        self.heightfield[0:self.tot_rows, :] = random_uniform_terrain(terrain=new_sub_terrain(),
                                                                      min_height=self.cfg["env"]["terrain"][
                                                                          "min_height"],
                                                                      max_height=self.cfg["env"]["terrain"][
                                                                          "max_height"],
                                                                      step=self.cfg["env"]["terrain"]["step_size"],
                                                                      downsampled_scale=self.cfg["env"]["terrain"][
                                                                          "downsampled_scale"]).height_field_raw

        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.heightfield, self.horizontal_scale,
                                                                       self.vertical_scale,
                                                                       self.cfg["env"]["terrain"]["slopeTreshold"])

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.vertices.shape[0]
        tm_params.nb_triangles = self.triangles.shape[0]
        tm_params.transform.p.x = -self.border_size
        tm_params.transform.p.y = -self.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.vertices.flatten(order='C'), self.triangles.flatten(order='C'),
                                   tm_params)



    # terrain generator
from isaacgym.terrain_utils import *


class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i + 1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale,
                                                                       self.vertical_scale, cfg["slopeTreshold"])

    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                                            platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles
