import os
import time
import numpy as np
import pybullet as p
from typing import List, Tuple
import pathlib

class Rizon4:
    eps: float = 1.0e-5
    
    def __init__(self, headless: bool=True):
        self.headless = headless
        p.connect(p.DIRECT if self.headless else p.GUI)

        script_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('flexiv_api/assets/flexiv_rizon4.urdf')
        self.urdf_file = str(script_dir)
        print("Load from URDF:", self.urdf_file)
        self.robot: int = p.loadURDF(self.urdf_file)    # body index
        
        self.num_joints = 7        
        self.flange_link = 7
        self.tcp_link = 10                              # closed_fingers_tcp: 10
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    
    @staticmethod
    def clear_debug():
        p.removeAllUserDebugItems()
        
    def visualize(self, link_id: int=-1):
        if link_id == -1:
            link_id = self.tcp_link
        state = p.getLinkState(self.robot, link_id)
        position = state[0]
        orientation = state[1]
        self.visualize_coords(position, orientation)
    
    def visualize_point(self, position: np.ndarray, color: Tuple[int]=(0, 0, 0), size: float=5):
        p.addUserDebugPoints([position], [color], size)
        
    def visualize_coords(self, position: np.ndarray, orientation: np.ndarray):
        mat = np.eye(4)
        mat[:3, 3] = position
        mat[:3, :3] = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,0]*0.3, (1,0,0), 3.0)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,1]*0.3, (0,1,0), 3.0)
        p.addUserDebugLine(mat[:3,3], mat[:3,3] + mat[:3,2]*0.3, (0,0,1), 3.0)        
    
    def get_dof_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """dof limits (8-dofs, including gripper)

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        p.getNumJoints(self.robot)

        dof_lower_limits = []
        dof_upper_limits = []
            
        for i in self.joint_indices:
            ret = p.getJointInfo(self.robot, i)    
            dof_lower_limits.append(ret[8])
            dof_upper_limits.append(ret[9])
        
        # gripper width limits
        # dof_lower_limits.append(0)
        # dof_upper_limits.append(0.10)
        return np.array(dof_lower_limits, dtype=np.float32), np.array(dof_upper_limits, dtype=np.float32)
    
    @staticmethod
    def get_tcp_limits() -> Tuple[np.ndarray, np.ndarray]:
        """tcp limits

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        tcp_lower_bound = np.array([0.50, -0.24, 0.20, -1, -1, -1, -1], dtype=np.float32) # xyz, wxyz, griv
        tcp_upper_bound = np.array([0.82,  0.32, 0.55,  1,  1,  1,  1], dtype=np.float32)
        
        return tcp_lower_bound, tcp_upper_bound
    
    @staticmethod
    def get_flange_limits() -> Tuple[np.ndarray, np.ndarray]:
        """tcp limits

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        flange_lower_bound = np.array([0.45, -0.24, 0.23, -np.pi, -np.pi, -np.pi], dtype=np.float32) # xyz, wxyz, griv
        flange_upper_bound = np.array([0.82,  0.32, 0.55,  np.pi,  np.pi,  np.pi], dtype=np.float32)
        
        return flange_lower_bound, flange_upper_bound

    def get_catersian(self, link_id: int):
        """Get cartesian states

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        p.stepSimulation()
        state = p.getLinkState(self.robot, link_id)
        position = state[0]
        orientation = state[1]
        return np.array(position + orientation)
    
    def get_flange_catersian(self):
        """get flange cartesian states"""
        return self.get_catersian(link_id=self.flange_link)
    
    def get_tcp_catersian(self):
        """get tcp cartesian states"""
        return self.get_catersian(link_id=self.tcp_link)
    
    def get_joints(self):
        p.stepSimulation()
        return [p.getJointState(self.robot, i)[0] for i in self.joint_indices]
    
    def set_joints(self, joint_states: np.ndarray):
        """Set joint states

        Args:
            joint_states (np.ndarray): _description_
        """
        for i, joint_state in enumerate(joint_states):
            idx = self.joint_indices[i]
            assert p.getJointInfo(self.robot, idx)[2] == 0
            p.resetJointState(self.robot, idx, joint_state)
        # print("@@@@@@@@")
        # print(joint_states)
        # # get joint state
        # for i in self.joint_indices:
        #     print(p.getJointState(self.robot, i)[0])
        # print('########')
        p.stepSimulation()
    
    def calc_joints(self, link_id: int, position: np.ndarray, orientation: np.ndarray) -> List[float]:
        """Get joint states

        Args:
            position (np.ndarray): _description_
            orientation (np.ndarray): w, x, y, z

        Returns:
            np.ndarray: _description_
        """
        joint_states = p.calculateInverseKinematics(
            bodyIndex=self.robot,
            endEffectorLinkIndex=link_id,
            targetPosition=position,
            targetOrientation=orientation
        )
        # print(self.joint_indices, self.joint_indices.dtype, len(joint_states))
        joint_states = [joint_states[i] for i in self.joint_indices]
        return joint_states
    
    def calc_ik(self, tq: np.ndarray, is_tcp: bool=False):
        """tq represents postion, orientation (w, x, y, z)

        Args:
            position (np.ndarray): _description_
            orientation (np.ndarray): _description_
        """
        position = tq[:3]
        orientation = tq[3:]

        p.stepSimulation()
        link_id = self.tcp_link if is_tcp else self.flange_link
        for t in range(1000):
            joints = self.calc_joints(link_id=link_id, position=position, orientation=orientation)
            self.set_joints(joints)
            p.stepSimulation()
            delta = np.linalg.norm(tq - self.get_catersian(link_id))
            if delta < Rizon4.eps:
                break
        else:
            raise ValueError('calc_ik failed!')

        return self.get_joints()
            

if __name__ == '__main__':
    robot = Rizon4(headless=False)
    low, up = robot.get_dof_limits()

    robot.visualize()
    target = np.array([0.4, 0.4, 0.4, 1, 0, 0, 0])
    robot.visualize_point(target[:3], (1, 0, 0), 10)
    
    robot.calc_ik(target)
    robot.visualize()
    
    
    import itertools
    tcp_lower_bound = np.array([0.45, -0.24, 0.03], dtype=np.float32) # xyz
    tcp_upper_bound = np.array([0.82,  0.32, 0.35], dtype=np.float32)
    
    bounding_box_corners = list(itertools.product(*zip(tcp_lower_bound, tcp_upper_bound)))
    for corner in bounding_box_corners:
        robot.visualize_point(corner, (1, 0, 0), 5)    
        target[:3] = corner
        robot.calc_ik(target, is_tcp=True)
        print(robot.get_flange_catersian()[:3])


    while True:
        p.stepSimulation()