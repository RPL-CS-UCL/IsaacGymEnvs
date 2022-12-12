"""A model based controller framework."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np
import time
from typing import Any, Callable
import torch as tc 
from gym import spaces

class NN_controller(object):
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      robot: Any,
      pybullet_client,
      state_estimator,
      clock,
      lin_speed,
      ang_speed,
      last_actions
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      swing_leg_controller: Generates motor actions for swing legs.
      stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._robot = robot
    self._p = pybullet_client
    self._clock = clock
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._state_estimator = state_estimator
    self.lin_speed =lin_speed
    self.ang_speed =ang_speed
    self.last_actions = last_actions
    self.p = pybullet_client

    self.path_to_file = "/home/maria/A1_controllers/quadrupedal_controllers/NN/traced_A1_NN_NEW.pt"
    self.device = tc.device('cpu')

    #Normalisation Parameters
    self.linearVelocityScale= 2.0
    self.angularVelocityScale= 0.25
    self.projectedGravityScale= 1.0
    self.dofPositionScale= 1.0
    self.dofVelocityScale= 0.05
    self.userCommandScale= [2.0, 2.0, 0.25]

    
  @property
  def state_estimator(self):
    return self._state_estimator



  def reset(self, current_time: float)-> None:
    del current_time
    self._reset_time = self._clock()
    self._time_since_reset = 0
    #self._state_estimator.reset(self._time_since_reset)
    if self._time_since_reset == 0:
        self._state_estimator = self._robot.state

    
  def update(self):
    self._time_since_reset = self._clock() - self._reset_time
    #self._state_estimator.update(self._time_since_reset)
    if self._time_since_reset != 0:

        _state_estimator = np.zeros(48)


        #base velocity 
        self._robot.base_vel= np.array(self._robot.GetBaseVelocity()).copy()    
        base_orientation = self.p.getBasePositionAndOrientation(1)[1]
        base_euler = self.p.getEulerFromQuaternion(base_orientation)
        self._robot.base_or = np.array(self._robot.GetBaseRollPitchYawRate()).copy()
        
        #projected gravity
        rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        world_gravity_direction = [0,0,-1]
        self._robot.grav_vec = (rot_mat.transpose()).dot(world_gravity_direction)
        
        #Commands 
        commands=np.zeros(3)

        # self._robot.commands[:2]= self.lin_speed[:2]
        # self._robot.commands[2:]= self.ang_speed
        commands[:2]= self.lin_speed[:2]
        commands[2:]= self.ang_speed


        #joint position 
        q_mes = self._robot.GetTrueMotorAngles()
        self._robot.q_in  = self.reorderJoints(q_mes)
        #self._robot.q_in  = q_mes

        # joint velocity
        dq_mes = self._robot.GetMotorVelocities()
        self._robot.dq_in  = self.reorderJoints(dq_mes)
        #self._robot.dq_in  = dq_mes
        
        #last action 
        #robot.last_command = torque_action[np.nonzero(torque_action.copy())]
        self.last_actions= self.last_actions

        _state_estimator[:3] = self._robot.base_vel[:] * self.linearVelocityScale
        _state_estimator[3:6] = self._robot.base_or[:] * self.angularVelocityScale
        _state_estimator[6:9] = self._robot.grav_vec[:] * self.projectedGravityScale
        _state_estimator[9:21] = self._robot.q_in[:] * self.dofPositionScale
        _state_estimator[21:33] = self._robot.dq_in[:] * self.dofVelocityScale
        _state_estimator[33:36] = commands[:] *  self.userCommandScale
        _state_estimator[36:48] = self.last_actions[:]

        self._state_estimator = _state_estimator
    
    return self._state_estimator
    

   
  def get_action(self,state):
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    # swing_action, joint_pos = self._swing_leg_controller.get_action()
    # # start_time = time.time()
    # stance_action, qp_sol = self._stance_leg_controller.get_action()
    # # print(time.time() - start_time)
   
    joint_ids =[]
    action = []
    # num = self._robot.num_motors
    # for name in self._robot.pbint.joint_name_to_id:
    #     joint_id = self._robot.joint_name_to_id[name] 
    #     joint_ids.append(joint_id)

    # for joint_id in range(self._robot.num_motors):
    #   if joint_id in swing_action:
    #     action.extend(swing_action[joint_id])
    #   else:
    #     assert joint_id in stance_action
    #     action.extend(stance_action[joint_id])
    # action = np.array(action, dtype=np.float32)

    input_tensor = tc.tensor(self._state_estimator).cpu()

    load_NN = tc.jit.load('NN/traced_A1_NN.pt')
    # load_NN = tc.jit.load('NN/traced_A1_NN_not_normalised.pt')
    load_NN.eval()

    action_keep = np.zeros(60)

    action = load_NN.forward(tc.unsqueeze(input_tensor, 0)).detach().numpy()
    action = np.squeeze(action)

    #if hybrid control is on 
    action_keep[4::5] = action

    #re-order action due Isaac missalignment 
    actions_reordered = self.reorderActions(action)
    
    return actions_reordered/3


  def reorderJoints(self,in_vec):
    q_out = np.zeros(12)
    q_out[0] = in_vec[3]
    q_out[1] = in_vec[4]
    q_out[2] = in_vec[5]
    q_out[3] = in_vec[0]
    q_out[4] = in_vec[1]
    q_out[5] = in_vec[2]
    q_out[6] = in_vec[9]
    q_out[7] = in_vec[10]
    q_out[8] = in_vec[11]
    q_out[9] = in_vec[6]
    q_out[10] = in_vec[7]
    q_out[11] = in_vec[8]
    q_out = q_out

    return q_out



  def reorderActions(self,in_vec):
    q_out = np.zeros(12)
    q_out[3] = in_vec[0]
    q_out[4] = in_vec[1]
    q_out[5] = in_vec[2]
    q_out[0] = in_vec[3]
    q_out[1] = in_vec[4]
    q_out[2] = in_vec[5]
    q_out[9] = in_vec[6]
    q_out[10] = in_vec[7]
    q_out[11] = in_vec[8]
    q_out[6] = in_vec[9]
    q_out[7] = in_vec[10]
    q_out[8] = in_vec[11]
    q_out = q_out

    return q_out
