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
        self.rew_scales["gait_period"] = self.cfg["env"]["learn"]["gaitPeriodRewardScale"]
        self.rew_scales["footstep"] = self.cfg["env"]["learn"]["footstepReward"]
        self.rew_scales["gait"] = self.cfg["env"]["learn"]["gaitRewardScale"]
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminationReward"]


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

        # Terrain
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.height_samples = None
        self.custom_origins = False

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.testing = False


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
        self.common_step_counter = 0
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
        self.period = torch.zeros(self.num_envs,dtype=torch.float, device=self.device, requires_grad=False)

        ## Initiliaze Gait Tensor
        self.init_done = True
        self.time = 0
        self.cont_time = []
        self.contact_when = None
        self.gait_cycle = torch.zeros(10000)
        self.stance_contact_counter = torch.zeros(self.num_envs, device=self.device)
        self.stance_updated_counter = torch.zeros(self.num_envs, device=self.device)
        self.swing_contact_counter = torch.zeros(self.num_envs, device=self.device)
        self.swing_updated_counter = torch.zeros(self.num_envs, device=self.device)
        self.diff = torch.zeros(self.num_envs, device=self.device)
        self.diff_change = torch.zeros(self.num_envs, device=self.device)
        self.stance_step_counter = torch.zeros(self.num_envs, device=self.device)
        self.check_if_end = torch.zeros(self.num_envs, device=self.device)
        self.indx_update  = torch.zeros(self.num_envs, device=self.device)
        self.steping = True
        self.till_contact = 8
        self.pos_till_contact =  torch.Tensor([[[[ 0.1669,  0.1594,  0.0666], [ 0.1641, -0.1592,  0.0663], [-0.2535,  0.1588,  0.0755],[-0.2546, -0.1586,  0.0751]],
                                        [[ 0.1627,  0.1563,  0.0621], [ 0.1604, -0.1588,  0.0627],[-0.2553,  0.1595,  0.0710],[-0.2574, -0.1580,  0.0710]],
                                        [[ 0.1565,  0.1527,  0.0560],[ 0.1568, -0.1581,  0.0579],[-0.2568,  0.1605,  0.0651],[-0.2605, -0.1579,  0.0656]],
                                        [[ 0.1512,  0.1481,  0.0491],[ 0.1525, -0.1574,  0.0522],  [-0.2584,  0.1616,  0.0579],[-0.2623, -0.1569,  0.0583]],
                                        [[ 0.1465,  0.1430,  0.0413], [ 0.1483, -0.1564,  0.0455],[-0.2599,  0.1629,  0.0496],[-0.2660, -0.1541,  0.0405]],
                                        [[ 0.1372,  0.1319,  0.0227],[ 0.1401, -0.1540,  0.0288],[-0.2650,  0.1653,  0.0310],[-0.2677, -0.1525,  0.0298]],
                                        [[ 0.1365,  0.1318,  0.0199],[ 0.1372, -0.1534,  0.0204],[-0.2668,  0.1662,  0.0203],[-0.2683, -0.1518,  0.0199]],
                                        [[ 0.1364,  0.1318,  0.0200],[ 0.1371, -0.1534,  0.0200],[-0.2668,  0.1662,  0.0200],[-0.2683, -0.1518,  0.0200]]]]).cuda(0)
       

        self.iteration_index = torch.zeros(self.num_envs, device=self.device)
        self.footstep_update = torch.zeros(self.num_envs, device=self.device)


        self.idx = []


        #Initialise height points
        self.height_points = self.init_height_points()
        self.measured_heights = None

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
                             "action_rate": torch_zeros(), "hip": torch_zeros(), "gait": torch_zeros(), "foot_contact": torch_zeros(),
                             "gait_period": torch_zeros() ,"rew_buf": torch_zeros(), 'footstep': torch_zeros(), 'termination': torch_zeros()}

        #### testing data
        if self.testing:
            self.save_footstep = []
            self.save_ref_cont =[]
            self.save_torques = []
            self.save_com_vel = []
            self.save_pos = []
            self.save_period = []
        ############################### Mania Gaits #######################################

        #Load MPC DATA
        self.load_mpc_390()



        #Velocity selection
        self.velocities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).cuda(0)
        #self.velocities = torch.tensor([0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]).cuda(0)
      
        self.num_vel = len(self.velocities)
        

        #Gait MPC Tensor :
        ##Gaits Contacts
        #self.gaits = torch.tensor(np.array([self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03, self.ct_tp_03])).cuda(0)
        self.gaits = torch.tensor(np.array([self.ct_tp_01 ,self.ct_tp_02, self.ct_tp_03, self.ct_tp_04,self.ct_tp_05,self.ct_tt_06,self.ct_tt_07, self.ct_tt_08, self.ct_tt_09])).cuda(0)
        # ## Positions
        self.calf = torch.tensor(np.array([self.calf1[:,:,0],self.calf2[:,:,0],self.calf3[:,:,0],self.calf4[:,:,0],self.calf5[:,:,0],self.calf6[:,:,0],self.calf7[:,:,0],self.calf8[:,:,0],self.calf9[:,:,0]])).cuda(0)
        self.hip = torch.tensor(np.array([self.hip1[:,:,0],self.hip2[:,:,0], self.hip3[:,:,0],self.hip4[:,:,0],self.hip5[:,:,0],self.hip6[:,:,0],self.hip7[:,:,0],self.hip8[:,:,0],self.hip9[:,:,0]])).cuda(0)
        self.foot = torch.tensor(np.array([self.f_tp_01[:,:,:], self.f_tp_02[:,:,:], self.f_tp_03[:,:,:], self.f_tp_04[:,:,:], self.f_tp_05[:,:,:], self.f_tt_06[:,:,:], self.f_tt_07[:,:,:], self.f_tt_08[:,:,:], self.f_tt_09[:,:,:]])).cuda(0)
        # 
        # ## Gait Period
        self.periods = torch.tensor(np.array([self.period1, self.period2, self.period3, self.period4,self.period5, self.period6, self.period7, self.period8, self.period9])).cuda(0)
        
        # #MPC TIME STEPS
        self.num_ts = len(self.ct_tp_02)
        self.gait = torch.zeros(self.num_envs, self.num_ts, 4).cuda(0)
        # 
        self.gait_downsampled = torch.zeros(self.num_envs, self.num_ts + 2, 4).cuda(0)
        self.pos_downsampled = torch.zeros(self.num_envs, self.num_ts + 2, 4).cuda(0)
        self.feet_downsampled = torch.zeros(self.num_envs, self.num_ts + 2, 4,3).cuda(0)
        # 
        # self.ind = torch.randint(0, self.num_vel, (self.num_envs,)).cuda(0)
        # self.velocity_blending()

        self.commands[:, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[:, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (self.num_envs, 1), device=self.device).squeeze(1)
        self.commands[:, 2] = 0 # torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (self.num_envs, 1), device=self.device).squeeze(1)

        if self.testing:
            self.ind2 = torch.randint(0, 8, (self.num_envs,)).cuda(0)
            self.testing_velocity_profile()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        terrain_type = self.cfg["env"]["terrain"]["terrainType"]  # plane - flat terrain/no curiculum

        if terrain_type == 'plane':
            self._create_ground_plane()
        elif terrain_type == 'trimesh':
            self._create_trimesh()
            self.custom_origins = True
        
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # 
        # # If randomizing, apply once immediately on startup before the fist sim step
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

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

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

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

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(A1_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')

        #self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
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
        
        # Terrain: env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.
        

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.A1_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            ##Terrain
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)


            #Rigid Properties
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(A1_asset, rigid_shape_prop)

            A1_handle = self.gym.create_actor(env_ptr, A1_asset, start_pose, "A1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, A1_handle, dof_props)
            #self.gym.enable_actor_dof_force_sensors(env_ptr, A1_handle)
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

    def _check_termination(self):

        # reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=-1) > 1.
        # time_out = self.progress_buf > self.max_episode_length
        # reset_buf = reset | time_out
        #
        # if not self.allow_knee_contatcs:
        #     reset = torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
        #     reset_buf = reset | reset_buf
        #self.reset_buf = reset_buf

        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.

        if not self.allow_knee_contatcs:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(knee_contact, dim=1)

        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)


    def compute_observations(self):
        self.measured_heights = self.get_heights()
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

    def compute_reward(self):

        # prepare quantities (TODO: return from obs ?)
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        projected_gravity = quat_rotate(base_quat, self.gravity_vec)

        # TODO check the height
        target_height = 0.35

        ### Reward: XY Linear Velocity (difference between base and input command)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]

        ### Reward: Z Linear Velocity
        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]

        ### Reward: Around XY Angular Velocity
        rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        ### Reward: Around Z Angular Velocity (difference between base and input command)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25)  * self.rew_scales["ang_vel_z"]

        ### Reward: Orientation
        rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        ### Reward: Torque Penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        ### Reward: Joint Acceleration Penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        ### Reward: Base height Penalty
        rew_base_height = torch.square(self.root_states[:, 2] - target_height) * self.rew_scales["base_height"]

        ### Reward: Action rate
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        ### Reward: Gait
        rew_gait = self._get_gait_reward() * self.rew_scales["gait"]

        ### Reward: Gait Period

        # print('desired', self.period)
        # print('reward',torch.abs(self._gait_period() - self.period))

        rew_gait_period = torch.abs(self._gait_period() - self.period)  *self.rew_scales['gait_period']

        ## Reward: Footsteps
        #rew_footstep = self._get_foot_position_reward()  * self.rew_scales['footstep']

        #get mpc foootstep in right format

        # total reward buffer
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                       rew_torque + rew_joint_acc + rew_gait + rew_gait_period
        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        ##COM REWARDS
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["base_height"] += rew_joint_acc
        # GAIT REWARDS
        self.episode_sums["rew_buf"] += self.rew_buf
        self.episode_sums["gait"] += rew_gait
        self.episode_sums["gait_period"] += rew_gait_period
        #self.episode_sums["footstep"] += rew_footstep
        #self.episode_sums["termination"] += rew_footstep

        # if self.testing:
        #     error = (self.base_lin_vel[:, 0] - self.commands[:, 0]) / self.commands[:, 0]
        #     self.save_footstep.append(self.sim_contacts)
        #     self.save_ref_cont.append(self.desired_mpc)
        #     self.save_com_vel.append(self.base_lin_vel)
        #     self.save_torques.append(self.torques)
        #     self.save_pos.append(self.feet_sim)
        #     self.save_period.append(self._gait_period())

    ######## Rest ##########

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
        # self.ind = torch.randint(0, self.num_vel, (len(env_ids),)).cuda(0) #4 * torch.ones(self.num_envs,dtype=torch.int32).cuda(0)  #self.num_vel
        # self.velocity_blending()
        #
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1],(len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = 0 #torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
   
   
        # # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # Reset agents
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Generate new commands
        #self._resample_commands(env_ids)

        # Reset data buffers
        self.progress_buf[env_ids] = 0.
        self.reset_buf[env_ids] = 1.
        self.feet_air_time[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.iteration_index[env_ids] = 0.
        self.footstep_update[env_ids] = 0.
        self.stance_step_counter[env_ids] = 0.

        # Register individual reward data for logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

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
        self.randomize_buf += 1
        self.iteration_index += 1
        self.footstep_update +=1

        # #M: Push Robot
        # if self.common_step_counter % self.push_interval == 0:
        #     self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # self.ind = torch.randint(0, self.num_vel, (self.num_envs,)).cuda(0)  #5 * torch.ones(self.num_envs, dtype=torch.int32).cuda(0) #torch.randint(0, self.num_vel, (self.num_envs,)).cuda(0)
        # self.velocity_blending()

        # print('command',self.commands[:,0])
        # print('base lin vel', self.base_lin_vel)


        if self.testing:
            it_id = self.iteration_index.type(torch.LongTensor)
            self.commands[:, 0] = self.vels[it_id]

            print('Starting Velocity', self.vel1)
            print('End Velocity', self.vel2)
            print('command',self.commands[:,0])
            print('base lin vel', self.base_lin_vel)
            print()

        # Update the reset buffer
        self._check_termination()

        # Calculate the rewards
        self.compute_reward()

        # if self.testing:
        #     self.testing_save_data('robohike')

        # Reset the agents that need termination
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Get observations
        self.compute_observations()

        #M: Add noise
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]


    ########################## GAIT ##############################
    def load_mpc_390(self):

        # load data from MPC
        path = "/home/robohike/motion_imitation/"
        with np.load(path + "footsteps_mpc.npz") as target_mpc:
            self.target_mpc = target_mpc["footsteps"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts.npz") as conatct_mpc:
            self.contacts = conatct_mpc["feet_contacts"]

        # load MPC gaits
        ########### tripod
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

        ########### trotting
        with np.load(path + "foot_contacts6.npz") as ct_tt_06:
            self.ct_tt_06 = ct_tt_06["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts7.npz") as ct_tt_07:
            self.ct_tt_07 = ct_tt_07["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts8.npz") as ct_tt_08:
            self.ct_tt_08 = ct_tt_08["feet_contacts"]  # [n_time_steps, feet indices, xyz]
        with np.load(path + "foot_contacts9.npz") as ct_tt_09:
            self.ct_tt_09 = ct_tt_09["feet_contacts"]
        with np.load(path + "foot_contacts10tt.npz") as ct_tt_10:
            self.ct_tt_10 = ct_tt_10["feet_contacts"]

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

    def velocity_blending(self):

        #### save_vel
     #4 * torch.ones(self.num_envs,dtype=torch.int32).cuda(0)  #torch.randint(0, num_vel, (self.num_envs,)).cuda(0)  # num_vel-1
        self.commands[:, 0] = torch.index_select(self.velocities, 0, self.ind) # corresponding gait
        self.gait = torch.index_select(self.gaits, 0, self.ind) # corresponding gait
        self.period = torch.index_select(self.periods, 0, self.ind) #corresponding period
        self.calf_mpc = torch.index_select(self.calf,0,self.ind).cuda(0)
        self.hip_mpc = torch.index_select(self.hip, 0, self.ind).cuda(0)
        self.foot_mpc= torch.index_select(self.foot, 0, self.ind).cuda(0)

    def _gait_period(self):
        'Check how long it took for the reference foot to complete a full gait cycle/period'

        # contact of the reference foot
        contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.
        #print('contact', contact)
        
        # contact counter for swing
        self.swing_updated_counter = torch.where(contact, self.swing_updated_counter, self.swing_updated_counter + 1)

        
        # contact counter for stance
        self.stance_updated_counter = torch.where(contact, self.stance_updated_counter + 1, self.stance_updated_counter)


        # see if phase changed - from swing to stance
        diff_stance = self.stance_updated_counter - self.stance_contact_counter
        
        #STORE OVERALL VALUES 
        self.stance_contact_counter = self.stance_updated_counter
        self.swing_contact_counter  =self.swing_updated_counter
       

        # step: diff goes from 0 to 1
        self.diff_change = torch.where(diff_stance > 0., self.diff_change + 1,
                                       torch.zeros(self.num_envs).cuda(0))  # check if there is change in phase

        # check when  the step happened

        steping_condition1 = self.steping != contact
        steping_condition2 = self.diff < self.diff_change


        self.stance_step_counter = torch.where(torch.logical_and(steping_condition1, steping_condition2), self.stance_step_counter + 1,
                                               self.stance_step_counter)  # times of change of variable from 0 to 1: step

        self.stance_step_counter = self.stance_step_counter
        self.diff = self.diff_change
        self.steping = contact
        #print('step', self.stance_step_counter)

        ## cycle ends after 2 steps
        end_cycle = (self.stance_step_counter % 2) == 0  # 1,3,5,... take only every 2 steps
        # get index of second step

        condition1 = self.check_if_end < self.stance_step_counter
        condition2 = end_cycle == 0

        period_so_far = self.swing_updated_counter + self.stance_updated_counter
        #print('period', period_so_far)


        # self.cycle_index = torch.where(condition == 2*torch.ones(self.num_envs).cuda(0),self.swing_updated_counter,torch.zeros(self.num_envs).cuda(0)) #index cycle ends
        self.cycle_index = torch.where(condition1, period_so_far,self.indx_update)  # index cycle ends
        self.indx_update = self.cycle_index



        self.check_if_end = self.stance_step_counter

        # self.check_if_end < self.stance_step_counter and

        # only compute reward when there is a full cycle


        self.stance_updated_counter = torch.where(condition1, torch.zeros(self.num_envs).cuda(0),
                                           self.stance_updated_counter)
        
        self.swing_updated_counter = torch.where(condition1, torch.zeros(self.num_envs).cuda(0),
                                           self.swing_updated_counter)

        self.reward_scaling = torch.where(condition1, torch.ones(self.num_envs).cuda(0),
                                           torch.zeros(self.num_envs).cuda(0))

        ### index reset
        # # # #
        # self.iteration_index = torch.where(condition1,
        #                                    torch.zeros(self.num_envs).cuda(0) + self.till_contact,
        #                                    self.iteration_index)
        #
        # self.footstep_update = torch.where(self.cycle_index != 0, self.cycle_index,
        #                                    self.footstep_update)



        # print('Iteration', self.iteration_index)
        # print('cycle', self.cycle_index)
        # print('period', period_so_far)



        return self.cycle_index

    def get_mpc_gait(self):
        #print(self.iteration_index)
        test = self.iteration_index.expand(4, -1)
        test = test.t()
        test = test.to(torch.int64)
        ttest1 = torch.unsqueeze(test, 1)

        self.gait_downsampled[:, :self.till_contact, :] = torch.zeros(self.num_envs, self.till_contact,
                                                                      len(self.feet_indices))
        self.gait_downsampled[:, self.till_contact:, :] = self.gait[:, :2500 - self.till_contact + 2, :]

        desired_mpc = torch.gather(self.gait_downsampled, 1, ttest1.to(torch.int64))
        #desired_mpc = torch.gather(self.gait, 1, ttest1.to(torch.int64))
        desired_mpc = torch.squeeze(desired_mpc, 1)



        return desired_mpc

    def distance_feet_gait(self,indx):
        foot_gait = torch.abs(self.desired_mpc[:, indx] - self.sim_contacts[:, indx])
        return foot_gait

    def get_mpc_hip_knee(self,dof):
        # get footstep for each env and account for time index

        len_idx = 4

        test1 = self.iteration_index.expand(len_idx, -1)
        test1 = test1.t()
        test1 = test1.to(torch.int64)
        ttest2 = torch.unsqueeze(test1, 1)
            
            
        ## mpc gait in the same format as Isaac
        self.pos_downsampled[:, :self.till_contact, :,:] = torch.zeros(self.num_envs, self.till_contact,
                                                                      len_idx)
        self.pos_downsampled[:, self.till_contact:, :,:] = dof[:, :2500 - self.till_contact + 2, :]


        # desired_footstep = self.footsteps.reshape(self.num_envs, 10000, 12)
        desired_footstep = torch.gather(self.pos_downsampled, 1, ttest2.to(torch.int64))
        desired_footstep = torch.squeeze(desired_footstep, 1)
        # desired_footstep = desired_footstep.reshape(self.num_envs,4,3)
        return desired_footstep

    def get_mpc_footstep(self):
        # get footstep for each env and account for time index

        ## mpc gait in the same format as Isaac
        self.feet_downsampled[:, :self.till_contact, :,:] = self.pos_till_contact
        self.feet_downsampled[:, self.till_contact:, :, :] = self.foot_mpc[:, :2500 - self.till_contact + 2, :, :]

        dof = self.feet_downsampled.view(self.num_envs, self.num_ts+2, 12)
        len_idx = 12

        test1 = self.footstep_update.expand(len_idx, -1)
        test1 = test1.t()
        test1 = test1.to(torch.int64)
        ttest2 = torch.unsqueeze(test1, 1)

        # desired_footstep = self.footsteps.reshape(self.num_envs, 10000, 12)
        desired_footstep = torch.gather(dof, 1, ttest2.to(torch.int64))
        desired_footstep = torch.squeeze(desired_footstep, 1)
        # desired_footstep = desired_footstep.reshape(self.num_envs,4,3)
        return desired_footstep


    def distance_footstep(self):

        diff = self.feet_sim  - self.feet_target
        sqr = torch.square(diff)
        error = torch.sum(torch.sum(sqr,dim=1))
        return error

    # def footprint_draw(self):
    #
    #     #contcats from mp
    #     contacts_mpc = self.gait_downsampled.cpu().numpy()
    #
    #     #index of contacts within contact matrix
    #     fr = np.where(contacts_mpc[:,:,1] == (1))
    #     fl = np.where(contacts_mpc[:,:, 0] == (1))
    #     rr = np.where(contacts_mpc[:,:, 3] == (1))
    #     rl = np.where(contacts_mpc[:,:,2] == (1))
    #
    #     #fl foot footprints
    #     fl_footsteps = self.target_mpc[fl, 0, :]
    #     fl_footsteps = fl_footsteps[0, :]
    #     fr_footsteps = self.target_mpc[fr, 1, :]
    #     fr_footsteps =fr_footsteps[0,:]
    #     rl_footsteps = self.target_mpc[rl, 2, :]
    #     rl_footsteps = rl_footsteps[0, :]
    #     rr_footsteps = self.target_mpc[rr, 3, :]
    #     rr_footsteps = rr_footsteps[0, :]
    #
    #     footstep_collection = [fl_footsteps,fr_footsteps,rl_footsteps,rr_footsteps]
    #
    #
    #     axes_geom = gymutil.AxesGeometry(0.1)
    #     for feet in range(len(footstep_collection)):
    #         footsteps = footstep_collection[feet]
    #         for i in range(len(footsteps[:, 0])):
    #
    #             x_vals = footsteps[i, 0]
    #             y_vals = footsteps[i, 1]
    #             z_vals= footsteps[i, 2]
    #
    #             s_pose = gymapi.Transform()
    #             s_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #             s_pose.p.x = float(x_vals)
    #             s_pose.p.y = float(y_vals)
    #             s_pose.p.z = float(0)
    #             gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0],s_pose)
    #
    #
    #
    #     self.fr = self.np_right_dim(fr_footsteps)
    #     self.fl = self.np_right_dim(fl_footsteps)
    #     self.rr = self.np_right_dim(rr_footsteps)
    #     self.rl = self.np_right_dim(rl_footsteps)

    # def np_right_dim(self, tensor):
    #    tens_r = torch.from_numpy(tensor).cuda(0)
    #    tens_r = torch.unsqueeze(tens_r, 0)
    #    tens_r = tens_r.expand(self.num_envs, -1,-1)
    #    return tens_r

    # def footstep_real_time(self):
    #     feet = {
    #         "FL" : 4,
    #         "FR" : 8,
    #         "RL" : 12,
    #         "RR" : 16 }
    #
    #     self.foot_pos = self.rb_pos[:, :, :]
    #     foot_pos = np.array(self.foot_pos.cpu())
    #
    #     # CREATE SWING/STANCE PAIRS FOR TROTING
    #     pair_1 = [feet["RR"], feet["RR"]]
    #     pair_2 = [feet["FR"], feet["RL"]]
    #
    #     FL_contact = self.feet_contacts[:, 0]
    #     FL_contact = sum(FL_contact) > 0
    #     RR_contact = self.feet_contacts[:, 4]
    #     RR_contact = sum(RR_contact) > 0
    #     FR_contact = self.feet_contacts[:, 1]
    #     FR_contact = sum(FR_contact) > 0
    #     RL_contact = self.feet_contacts[:, 2]
    #     RL_contact = sum(RL_contact) > 0
    #
    #     self.f = 16
    #         # self.f2 = pair_1[1]
    #     #if FR_contact and RL_contact:
    #     #     self.f = pair_2[0]
    #     #     self.f2 = pair_2[1]
    #     #
    #     axes_geom = gymutil.AxesGeometry(0.1)
    #
    #     # generate a footprint
    #     s_pose = gymapi.Transform()
    #     s_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #     s_pose.p.x = float(foot_pos[0, self.f, 0])
    #     s_pose.p.y = float(foot_pos[0, self.f, 1])
    #     s_pose.p.z = float(foot_pos[0, self.f, 2])
    #
    #     # s_pose2 = gymapi.Transform()
    #     # s_pose2.p = gymapi.Vec3(0.0, 0.0, 0.0)
    #     # s_pose2.p.x = float(foot_pos[0, self.f2, 0])
    #     # s_pose2.p.y = float(foot_pos[0, self.f2, 1])
    #     # s_pose2.p.z = float(foot_pos[0, self.f2, 2])
    #
    #     self.gym.clear_lines(self.viewer)
    #
    #     # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], s_pose)
    #     gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], s_pose)
    ########################## Terrain ##############################

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

    ############## Rewards #####################

    
    def _get_hip_knee_position_reward(self):

        #get mpc foootstep in right format

        self.hip_target = self.get_mpc_hip_knee(self.hip_mpc)
        self.calf_target = self.get_mpc_hip_knee(self.calf_mpc)

        self.mpc_pos =  torch.zeros(2,self.num_envs, 4, device=self.device, requires_grad=False)
        self.mpc_pos[0,:,:] = self.calf_target
        self.mpc_pos[1,:,:] = self.hip_target

        self.hip_sim = self.rb_pos[:, self.hip_indices, 1]
        self.calf_sim = self.rb_pos[:, self.knee_indices, 1]

        self.sim_pos = torch.zeros(2, self.num_envs, 4, device=self.device, requires_grad=False)
        self.sim_pos[0, :, :] = self.hip_sim
        self.sim_pos[1, :, :] = self.calf_sim

        hip_knee_position = torch.sum(torch.sum(torch.square(self.mpc_pos-self.sim_pos),dim = -1),dim=0) * self.rew_scales["footstep"]
            
        return hip_knee_position

    def _get_foot_position_reward(self):

        self.foot_target = self.get_mpc_footstep()  # time step mpc position
        self.feet_target = self.foot_target.view(self.num_envs, len(self.feet_indices), 3)  # match shape of sim 

        self.feet_sim = self.rb_pos[:, self.feet_indices, :]  # sim step position

        sse = self.distance_footstep()

        return sse

    def _get_gait_reward(self):
        # gait forces from  MPC
        desired_mpc = self.get_mpc_gait()

        # forces from sim
        self.feet_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 0.2

        # self.footprint_draw()

        # make bool to int
        self.desired_mpc = desired_mpc.long()

        self.sim_contacts = self.feet_contacts.long()

        # difference between desired and sim
        diff = torch.abs(self.desired_mpc - self.sim_contacts)

        # rew_fl_foot = self.distance_feet_gait(0)
        # rew_fr_foot = self.distance_feet_gait(1)
        # rew_rl_foot = self.distance_feet_gait(2)
        # rew_rr_foot = self.distance_feet_gait(3)
        # 
        # rew_gait = torch.sum(1+rew_fl_foot + rew_fr_foot + rew_rl_foot + rew_rr_foot) * self.rew_scales["gait"]

        rew_gait = torch.sum(diff, dim=-1) * self.rew_scales["gait"]
        
        return rew_gait 


    ########## Saving Data ######################
    
    # def _get_gait_period_reward(self):
    #
    #     gait_error = torch.abs(self._gait_period() - self.period)
    #     return rew_gait_period
    #

    def testing_save_data(self,pc):
        save_vel = 'b01'
        torch.save(self.save_footstep, '/home/'+pc+'/test_data/fc'+save_vel+'.pt')
        torch.save(self.save_com_vel, '/home/'+pc+'/test_data/vel'+save_vel+'.pt')
        torch.save(self.save_torques, '/home/'+pc+'/test_data/trq'+save_vel+'.pt')
        torch.save(self.save_ref_cont, '/home/'+pc+'/test_data/ref'+save_vel+'.pt')
        torch.save(self.save_pos, '/home/'+pc+'/test_data/pos'+save_vel+'.pt')
        torch.save(self.save_period, '/home/'+pc+'/test_data/period'+save_vel+'.pt')

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

        self.vels = torch.tensor(vels).cuda(0)


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
                    if choice < 0.05:
                        slope *= -1
                    sloped_terrain(terrain, slope=slope)

                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    sloped_terrain(terrain, slope=slope)
                    # random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
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
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

