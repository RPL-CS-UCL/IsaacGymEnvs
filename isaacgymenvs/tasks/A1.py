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
        #self.user_command_scale = self.cfg["env"]["learn"]["userCommandScale"]
        self.user_command_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], requires_grad=False,).cuda()

        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.action_clip = self.cfg["env"]["control"]["actionClip"]

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
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["gait_hip"] = self.cfg["env"]["learn"]["gaitTrajectoryHipRewardScale"]
        self.rew_scales["gait_knee"] = self.cfg["env"]["learn"]["gaitTrajectoryKneeRewardScale"]
        self.rew_scales["gait_foot"] = self.cfg["env"]["learn"]["gaitTrajectoryFootRewardScale"]
        self.rew_scales["joint_angles"] = self.cfg["env"]["learn"]["JointAnglesRewardScale"]

        print('REWARD SCALES')
        print('Gait Hip Trajectory Reward Scale', self.rew_scales["gait_hip"])
        print('Gait Knee Trajectory Reward Scale', self.rew_scales["gait_knee"])
        print('Gait Foot Trajectory Reward Scale', self.rew_scales["gait_foot"])
        print('Gait Contacts Reward Scale', self.rew_scales["gait"])
        print('Gait Period Reward Scale', self.rew_scales["gait_period"])
        print('Lin vel', self.rew_scales["lin_vel_xy"])
        print('Action Rate Reward Scale', self.rew_scales["action_rate"])
        
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["knee_collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["foot_contact"] = self.cfg["env"]["learn"]["footcontactRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]

        # randomization
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["terrain"]["restitution"]

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

        # Terrain
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.height_samples = None
        self.custom_origins = False


        self.save_data = self.cfg["env"]["post_process"]["save_data"]
       
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.allow_knee_contatcs = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]


        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.push_robot = self.cfg["env"]["learn"]["pushRobot"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt


        self.sim = None

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)



        if self.graphics_device_id != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        check = self.sim

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
        self.sim_contacts = None

        self.reward_sim = 0
        self.reward_hip = 0
        self.reward_knee = 0
        self.reward_foot = 0


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
                             "gait_period": torch_zeros(),"gait_hip": torch_zeros(), "total_reward": torch_zeros(),
                             "gait_knee": torch_zeros(), "gait_foot": torch_zeros(), "joint_angles": torch_zeros()}

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
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.allow_hip_contacts = self.cfg["env"]["learn"]["allowHipContacts"]


        ####### Storing MPC Values

        #Period
        self.load_periods = torch.tensor(np.array(
            [self.period1, self.period2, self.period3, self.period4, self.period5, self.period6, self.period7,
             self.period8, self.period9])).cuda(0)

        ## Gait Contacts
        self.load_gaits = torch.tensor(np.array(
            [self.ct_tp_01[:, :], self.ct_tp_02[:, :], self.ct_tp_03[:, :], self.ct_tp_04[:, :], self.ct_tp_05[:, :], self.ct_tt_06[:, :], self.ct_tt_07[:, :],
             self.ct_tt_08[:, :], self.ct_tt_09[:, :]])).cuda(0)


        # Positions
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

        ## Joint Angles

        self.load_joint_angles = torch.tensor(np.array(
            [self.ja_01[:, :, :], self.ja_02[:, :, :], self.ja_03[:, :, :], self.ja_04[:, :, :],
             self.ja_05[:, :, :], self.ja_06[:, :, :], self.ja_07[:, :, :], self.ja_08[:, :, :],
             self.ja_09[:, :, :]])).cuda(0)
        #
        # #Actions - Torques
        # self.load_actions= torch.tensor(np.array(
        #     [self.actions_01[:, :, :], self.actions_02[:, :, :], self.actions_03[:, :, :], self.actions_04[:, :, :],
        #      self.actions_05[:, :, :], self.actions_06[:, :, :], self.actions_07[:, :, :], self.actions_08[:, :, :],
        #      self.actions_09[:, :, :]])).cuda(0)


        self.velocities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).cuda(0)




        self.num_vel = len(self.velocities)
        self.num_ts = len(self.ct_tp_02) + 2 #added two as had issues with dimensions not matching
        self.num_xyz = 3

        ## load MPC data
        self.gaits= torch.cat((self.load_gaits, torch.ones(len(self.load_gaits),2,self.num_legs, device=self.device)), 1).to(torch.bool)
        self.calf = torch.cat((self.load_calf, torch.ones(len(self.load_calf),2,self.num_legs,self.num_xyz, device=self.device)), 1)
        self.hip = torch.cat((self.load_hip, torch.ones(len(self.load_hip), 2, self.num_legs,self.num_xyz, device=self.device)), 1)
        self.foot = torch.cat((self.load_foot, torch.ones(len(self.load_foot), 2, self.num_legs,self.num_xyz, device=self.device)), 1)

        self.joint_angle_MPC = torch.cat((self.load_joint_angles, torch.ones(len(self.load_hip), 2, self.num_legs, self.num_xyz, device=self.device)), 1)
        # self.action_MPC = torch.cat(
        #     (self.load_actions, torch.ones(len(self.load_hip), 2, self.num_legs, self.num_xyz, device=self.device)), 1)

        ## placeholders for post processing MPC data
        self.gait = torch.zeros(self.num_envs, self.num_ts, self.num_legs, dtype=torch.bool).cuda(0)
        self.period = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device, requires_grad=False)
        self.hips = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64, device=self.device)
        self.calves = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz,dtype=torch.float64, device=self.device)
        self.feet = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64,device=self.device)

        self.joint_angles_MPC = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64,
                                  device=self.device)
        # self.actions_MPC = torch.zeros(self.num_envs, self.num_ts, self.num_legs, self.num_xyz, dtype=torch.float64,
        #                         device=self.device)

        self.enable_gait = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device)
        self.enable_reward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #self.enable_trajectory = torch.zeros(3, self.num_envs, self.num_legs,self.num_xyz,  dtype=torch.bool, device=self.device )

        #### testing data
        if self.save_data:
            self.save_footstep = []
            self.save_ref_cont =[]
            self.save_torques = []
            self.save_com_vel = []
            self.save_pos = []
            self.save_target_pos =[]
            self.save_period = []

        self.reset_idx(torch.arange(self.num_envs, device=self.device))



    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2


        if self.sim != None:
            self.gym.destroy_sim(self.sim)


        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)


        terrain_type = self.cfg["env"]["terrain"]["terrainType"]  # plane - flat terrain/no curiculum

        if terrain_type == 'plane':
            self._create_ground_plane()
        elif terrain_type == 'trimesh':
            self._create_trimesh()
            self.custom_origins = True

        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))



    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg."""
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
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]

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

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(A1_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)

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
        # for i in range(self.num_dof):
        #     dof_props['driveMode'][i] = self.cfg["env"]["control"]["driveMode"]
        #     dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
        #     dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        # Terrain: env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"][
                                                                                    "numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,),
                                            device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,),
                                           device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0

        env_lower = gymapi.Vec3(1+spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(-2+spacing, 10+ spacing, 0.0)
        self.A1_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            ##Terrain
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                #pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)


            #Rigid Properties
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(A1_asset, rigid_shape_prop)

            A1_handle = self.gym.create_actor(env_ptr, A1_asset, start_pose, "A1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, A1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, A1_handle)

            self.envs.append(env_ptr)
            self.A1_handles.append(A1_handle)

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

        ## Load and test the weights of pre-trained NN
        testNN = self.cfg["env"]["post_process"]["test_NN"]

        if testNN == True:
            actionsNN = self.test_extracted_NN('traced_new_position.jit')
            self.actions= actionsNN.clone().to(self.device)

        else:
            self.actions = actions.clone().to(self.device)

        for _ in range(self.decimation):

            if self.cfg["env"]["post_process"]["actionFilter"]:
                torques = self.action_filter.filter(torques.clone())

            if self.cfg["env"]["control"]["control_type"] == 'P':
                torques = torch.clip(self.Kp * (
                        self.action_scale * self.actions + self.default_dof_pos - self.dof_pos) - self.Kd * self.dof_vel,
                                     -self.action_clip, self.action_clip)

            if self.cfg["env"]["control"]["control_type"] == 'T':
                torques = self.actions * self.action_scale

            torques = torch.clip(torques, -self.action_clip, self.action_clip)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.progress_buf += 1

        #Update the counters
        self.common_step_counter += 1
        self.randomize_buf += 1
        self.iteration_index += 1


        #Push Robot
        if self.cfg["env"]["learn"]["pushRobots"] and self.common_step_counter % self.push_interval == 0:
            self.push_robots()


        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]

        it = self.iteration_index

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.save_data:
            self.ind2 = torch.randint(0,self.num_vel, (self.num_envs,)).cuda(0)
            self.ind2 = torch.where(self.ind2 != self.ind, self.ind2,
                                    torch.randint(0,self.num_vel, (self.num_envs,)).cuda(0)).cuda(0)

            it_id = self.iteration_index.type(torch.LongTensor)
            self.commands[:, 0] = self.testing_velocity_profile()[it_id]

            print('Starting Velocity', self.vel1)
            print('End Velocity', self.vel2)
            print('command', self.commands[:, 0])
            print('base lin vel', self.base_lin_vel)
            print()


        # Chcek of termination conditions are met
        self._check_termination()

        # Calculate the rewards
        self.compute_reward()

        # Save Data when testing
        if self.save_data:
            self.testing_save_data('robohike')

        # Reset the agents that need termination
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)


        # Get observations
        self.compute_observations()

        # Store last actions and last joint velocity observations
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]


    def compute_reward(self):


        #TODO check the height
        target_height = 0.30

        #print(self.root_states[0, 2])

        ### Reward: XY Linear Velocity (difference between base and input command)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]

        ### Reward: Z Linear Velocity 
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]

        ### Reward: Around XY Angular Velocity 
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        ### Reward: Around Z Angular Velocity (difference between base and input command)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        ### Reward: Orientation
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        ### Reward: Torque Penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        ### Reward: Joint Acceleration Penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        ### Reward: Base height Penalty
        rew_base_height = torch.square(self.root_states[:, 2] -  target_height) * self.rew_scales["base_height"]

        ### Reward: Action rate
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        ''' Mania '''

        ### Reward: Gait
        rew_gait = self._get_gait_reward() * self.rew_scales["gait"]


        ##Reward: Gait Period
        rew_gait_period= self._get_gait_period_reward() * self.rew_scales["gait_period"]

        ##Reward: Gait Trajectory:
        rew_gait_trajectory= self._get_gait_trajectory_reward()
        rew_gait_hip = rew_gait_trajectory[0,:] * self.rew_scales["gait_hip"]
        rew_gait_knee = rew_gait_trajectory[1, :] * self.rew_scales["gait_knee"]
        rew_gait_foot = rew_gait_trajectory[2, :] * self.rew_scales["gait_foot"]

        ###Reward: Joint Position
        rew_joint_angles = self.get_joint_angles_reward()  * self.rew_scales["joint_angles"]


        ### Reward: Foot Air Time
        #rew_feet_air_time = self._get_reward_foot_air_time() * self.rew_scales["air_time"]


        ### Reward: Knee collision
        rew_knee_collision = self._get_knee_collision_reward() * self.rew_scales["knee_collision"]


        ### Reward: Foot contact
        rew_foot_contact = self._get_foot_contact_reward() * self.rew_scales["foot_contact"]


        ### Reward: Hip
        #rew_hip = self._get_reward_hip() * self.rew_scales["hip"]
        #print(rew_hip[0])


        # Total reward buffer

        # self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
        #    rew_torque + rew_joint_acc + rew_action_rate

        self.rew_buf = rew_gait_period

        ## Enable reward only when there is first contact for accurate contact matching
        # self.rew_buf = torch.where(self.enable_reward, self.rew_buf, torch.zeros_like(self.rew_buf))


        #  +rew_knee_collision +rew_foot_contact +rew_gait +rew_foot_air_time +
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

        #Logging gait rewards
        self.episode_sums["gait_period"] += rew_gait_period
        self.episode_sums["gait"] += rew_gait


        self.episode_sums["total_reward"] += self.rew_buf


        # self.episode_sums["air_time"] += rew_foot_air_time
        self.episode_sums["knee_collision"] += rew_knee_collision
        self.episode_sums["foot_contact"] += rew_foot_contact
        self.episode_sums["gait"] += rew_gait
        self.episode_sums["gait_hip"] += rew_gait_hip
        self.episode_sums["gait_knee"] += rew_gait_knee
        self.episode_sums["gait_foot"] += rew_gait_foot
        self.episode_sums["joint_angles"] += rew_joint_angles


        # Save data
        if self.save_data:
            vel_error = (self.base_lin_vel[:, 0] - self.commands[:, 0]) / self.commands[:, 0]

            self.save_footstep.append(self.sim_contacts)
            self.save_ref_cont.append(self.sim_contacts)
            self.save_com_vel.append(self.base_lin_vel)
            self.save_torques.append(self.torques)
            self.save_target_pos.append(self._get_gait_trajectory()[0])
            self.save_pos.append(self._get_gait_trajectory()[1])
            self.save_period.append(torch.mean(self._gait_period(),dim=-1))

        self.reward_sim = torch.sum(self.rew_buf)
        self.reward_hip= torch.sum(rew_gait_hip)
        self.reward_knee= torch.sum(rew_gait_knee)
        self.reward_foot = torch.sum(rew_gait_foot)



    def compute_observations(self):

        self.obs_buf= torch.cat((self.base_lin_vel * self.lin_vel_scale,
                                  self.base_ang_vel * self.ang_vel_scale,
                                  self.projected_gravity,
                                  self.commands * self.user_command_scale,
                                  self.dof_pos * self.dof_pos_scale,
                                  self.dof_vel * self.dof_vel_scale,
                                  self.actions
                                  ), dim=-1)


    def reset_idx(self, env_ids):


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
        """Terminate learning process if the following conditions are met """


        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.

        #if knee touches the ground
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(knee_contact, dim=1)

        #if hip touches the ground
        if not self.allow_hip_contacts:
            hip_contact = torch.norm(self.contact_forces[:, self.hip_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(hip_contact, dim=1)

        # if base height is less than a minimum value
        if self.cfg['env']['learn']['terminatedBasedHeight']:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1), dim=1) \
                                       < self.cfg['env']['learn']['terminalHeight']
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)


        #if base has an extreme orientation
        if self.cfg['env']['learn']['terminatedBasedOrient']:
            self.body_orient_buf = torch.mean(self.root_states[:, 6].unsqueeze(1), dim=1) \
                                   < self.cfg['env']['learn']['terminalOrient']
            self.reset_buf = torch.logical_or(self.body_orient_buf, self.reset_buf)

        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

    #Push Robot
    def push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        self.root_states[:, 7:9] = torch_rand_float(-0.3, 0.8, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def update_terrain_level(self, env_ids):
        """ Updates terrain level for curiculum learning, DONT LOOK"""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]



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



    def _get_gait_period_reward(self):

        ''' Difference between simulation step period and desired MPC step period'''

        sim_period = self.period/self.decimation
        mpc_period = self._gait_period()
        rew_gait_period = torch.sqrt(torch.sum(torch.square(mpc_period - sim_period) , dim=-1))

        return rew_gait_period
    #
    def _get_gait_reward(self):

        ''' Difference between simulation feet contacts and desired MPC foot contacts'''

        # gait forces from  MPC
        mpc_contacts = self._get_mpc_to_sim()

        # forces from sim
        self.feet_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.

        #enabling reward after spawning
        self.enabling_rewards(mpc_contacts)

        desired_mpc_enabled = torch.zeros_like(mpc_contacts, device=self.device)
        desired_mpc_enabled = torch.where(self.enable_gait, mpc_contacts.long(), 0)
        
        # make bool to int
        sim_contacts = self.feet_contacts.long()

        # difference between desired and sim
        # diff = torch.square(desired_mpc_enabled - sim_contacts)  ## for enabling after first contact
        diff = torch.square(mpc_contacts.long() - sim_contacts)
        rew_gait = torch.sum(diff, dim=-1)
        return rew_gait

    def _get_gait_trajectory_reward(self):

        ''' Compute trajectory reard by applying Eucledian difference between mpc and sim positions '''

        mpc_trajectory, sim_trajectory = self._get_gait_trajectory()



        # enabling reward after spawning
        self.enabling_rewards(mpc_trajectory)

        mpc_trajectory_enabled = torch.zeros_like(mpc_trajectory, dtype=torch.float32, device=self.device)
        mpc_trajectory_enabled = torch.where(self.enable_trajectory, mpc_trajectory, sim_trajectory)


        #### enable trajectory or not?

        sqr_diff = torch.square(mpc_trajectory - sim_trajectory)  #squered diff between desired and simulated trajectories
        sum_sqr_diff = torch.sum(sqr_diff, dim=-1) # sum over xyz positions


        sqrt_sum_sqr_diff= torch.sqrt(sum_sqr_diff) #sqrt of the summed value
        sum_sqrt_sum_sqr_diff = torch.sum(sqrt_sum_sqr_diff,dim=-1) #sum over 4 feet
        trajectory_reward = torch.sum(sum_sqrt_sum_sqr_diff, dim=0) # sum over hips, calves nd feet


        return sum_sqrt_sum_sqr_diff




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
        # self.ind = torch.ones((len(env_ids)), dtype=torch.int8).cuda(0) * 5
        self.commands[env_ids, 0] = torch.index_select(self.velocities, 0, self.ind)

        # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1)



        # to test
        # self.commands[env_ids, 0] = 0.5
        # self.commands[env_ids, 1] = 0
        # self.commands[env_ids, 2] = 0


        # choose period according to velocity
        self.period_select = torch.index_select(self.load_periods, 0, self.ind)  # corresponding period
        self.period[env_ids] = torch.unsqueeze(self.period_select, dim=1).expand(-1, 4).to(torch.float32)

        # corresponding gait
        self.gait[env_ids]= torch.index_select(self.gaits, 0, self.ind)

        # corresponding trajectory positions
        self.calves[env_ids] = torch.index_select(self.calf,0,self.ind).cuda(0)
        self.hips[env_ids] = torch.index_select(self.hip, 0, self.ind).cuda(0)
        self.feet[env_ids] = torch.index_select(self.foot, 0, self.ind).cuda(0)
        #
        self.joint_angles_MPC[env_ids] = torch.index_select(self.joint_angle_MPC, 0, self.ind).cuda(0)
        # self.actions_MPC[env_ids] = torch.index_select(self.action_MPC, 0, self.ind).cuda(0)
        #
        #
        #
        #


        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    #################### MPC Functions
    def load_mpc_390(self):
        # load data from MPC
        path = "/home/robohike/motion_imitation/1000/"

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
        # ########### trotting
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
        # with np.load(path + "foot10.npz") as f_tt_10:
        #     self.f_tt_10 = f_tt_10["footsteps"]

        #
        #
        # load MPC joint angles
        # ########### trotting
        with np.load(path + "joint_angles1.npz") as ja_01:
            self.ja_01 = ja_01["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles2.npz") as ja_02:
            self.ja_02 = ja_02["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles3.npz") as ja_03:
            self.ja_03 = ja_03["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles4.npz") as ja_04:
            self.ja_04 = ja_04["joint_angles"]
        with np.load(path + "joint_angles5.npz") as ja_05:
            self.ja_05 = ja_05["joint_angles"]
        with np.load(path + "joint_angles6.npz") as ja_06:
            self.ja_06 = ja_06["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles7.npz") as ja_07:
            self.ja_07 = ja_07["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles8.npz") as ja_08:
            self.ja_08 = ja_08["joint_angles"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "joint_angles9.npz") as ja_09:
            self.ja_09 = ja_09["joint_angles"]
        # with np.load(path + "foot10.npz") as f_tt_10:
        #     self.f_tt_10 = f_tt_10["footsteps"

        # # load MPC joint angles
        # # ########### trotting
        # with np.load(path + "actions1.npz") as actions_01:
        #     self.action_01 = actions_01["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions2.npz") as actions_02:
        #     self.action_02 = actions_02["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions3.npz") as actions_03:
        #     self.action_03 = actions_03["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions4.npz") as actions_04:
        #     self.action_04 = actions_04["actions"]
        # with np.load(path + "actions5.npz") as actions_05:
        #     self.action_05 = actions_05["actions"]
        # with np.load(path + "actions6.npz") as actions_06:
        #     self.action_06 = actions_06["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions7.npz") as actions_07:
        #     self.action_07 = actions_07["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions8.npz") as actions_08:
        #     self.action_08 = actions_08["actions"]  # [n_time_steps, feet indices, xyz]
        # with np.load(path + "actions9.npz") as actions_09:
        #     self.action_09 = actions_09["actions"]


    def _gait_period(self):
        '''Check how long it took for the reference foot to complete a full gait cycle/period.'''

        # contact of the reference foot
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.


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

        # Checking Period
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

        start_reward =  start_gait.any(dim =-1)


        self.enable_reward = torch.logical_or(self.enable_reward,start_reward)

        self.enable_gait = torch.logical_or(self.enable_gait,start_gait)

        self.enable_traj= torch.unsqueeze(self.enable_gait,-1)
        self.enable_traje = torch.unsqueeze(self.enable_traj,0)
        self.enable_trajectory = self.enable_traje.expand(3,self.num_envs,self.num_legs,self.num_xyz)





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

        desired_mpc = torch.gather(self.gait, 1, ttest1.to(torch.int64))
        #desired_mpc = torch.gather(self.gait, 1, ttest1.to(torch.int64))
        desired_mpc = torch.squeeze(desired_mpc, 1)
        self.sim_contacts = desired_mpc
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

    def get_joint_angles_reward(self):

        joint_ang_MPC = self.get_mpc_footstep(self.joint_angles_MPC)  # jot angles mpc
        joint_ang_sim = self.dof_pos

        sqr_diff = torch.square(
            joint_ang_MPC - joint_ang_sim)  # squered diff between desired and simulated trajectories
        rew_joint_angles = torch.sum(sqr_diff, dim=-1)

        return rew_joint_angles





    def distance_footstep(self, sim, target):
        diff = sim - target
        sqr = torch.square(diff)
        error = torch.sum(torch.sum(sqr, dim=1))
        return error

    ######################### Testing Functions

    def testing_save_data(self,pc):
        save_vel = 'p09'
        torch.save(self.save_footstep, '/home/'+pc+'/test_data/fc'+save_vel+'.pt')
        torch.save(self.save_com_vel, '/home/'+pc+'/test_data/vel'+save_vel+'.pt')
        torch.save(self.save_torques, '/home/'+pc+'/test_data/trq'+save_vel+'.pt')
        torch.save(self.save_ref_cont, '/home/'+pc+'/test_data/ref'+save_vel+'.pt')
        torch.save(self.save_pos, '/home/'+pc+'/test_data/pos'+save_vel+'.pt')
        torch.save(self.save_target_pos, '/home/' + pc + '/test_data/target_pos' + save_vel + '.pt')
        torch.save(self.save_period, '/home/'+pc+'/test_data/period'+save_vel+'.pt')

    def test_extracted_NN(self, name_of_file):
        # take to ocnstructor

        NN_file = os.path.join(self.NN_path, name_of_file)
        load_NN = torch.jit.load(NN_file)
        load_NN.eval()
        actions = load_NN.forward(self.obs_buf)

        actions = torch.unsqueeze(actions, 0)

        return actions

    def testing_velocity_profile(self):
        #two random indeces
        ind1 = self.ind.cpu().numpy()
        ind2 = self.ind2.cpu().numpy()

        #velocities corresponding to these indeces
        self.vel1 = self.velocities[ind1].cpu().numpy()
        self.vel2 = self.velocities[ind2].cpu().numpy()

        w = 0.05
        D = np.linspace(0, 2, 2500)
        sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
        vels = self.vel1 + (self.vel2 - self.vel1) * (1 - sigmaD)

        velocity_profile = torch.tensor(vels).cuda(0)

        return velocity_profile




from isaacgym.terrain_utils import *


class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        # self.horizontal_scale = 0.1
        # self.vertical_scale = 0.005
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 10
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
        self.unstructured= False
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

            # choice = np.random.uniform(0, 1)
            # if choice < 0.1:
            # if np.random.choice([0, 1]):
            # sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            #sloped_terrain(terrain, 0)
            if self.unstructured == True:
                random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
            # else:
            #     pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            # elif choice < 0.6:
            #     # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
            #     step_height = np.random.choice([-0.15, 0.15])
            #     pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            # elif choice < 1.:
            #     discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            self.env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, self.env_origin_z]
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

                slope = difficulty * 0.1
                # step_height = 0.05 + 0.175 * difficulty
                # discrete_obstacles_height = 0.025 + difficulty * 0.15
                # stepping_stones_size = 2 - 1.8 * difficulty

                if choice < self.proportions[0]:
                   random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.025, downsampled_scale=0.2)

                # if choice < self.proportions[0]:
                #     if choice < 0.05:
                #         slope *= -1
                #     sloped_terrain(terrain, slope=slope)
                #
                # elif choice < self.proportions[1]:
                #     if choice < 0.15:
                #         slope *= -1
                #     sloped_terrain(terrain, slope=slope)
                #     random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                #
                # elif choice < self.proportions[3]:
                #     if choice<self.proportions[2]:
                #         step_height *= -1
                #     pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                # elif choice < self.proportions[4]:
                #     discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                # else:
                #     stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]






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

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, Tensor) -> Tensor
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


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

