import sys
sys.path.insert(0, '/home/flexiv/Desktop/flexiv_rdk/lib_py')
import flexivrdk
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import spdlog  

class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """

    def __init__(self, gripper_robot_addr=["192.168.3.100"]):
        """
        Initialize.

        Args:
            gripper_robot_addr: ["192.168.3.100"], robot_ip

        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.mode = flexivrdk.Mode
        self.robot_states = {"gripper_robot":flexivrdk.RobotStates()}

        self.robot_addr = {"gripper_robot":gripper_robot_addr}
        self.log = spdlog.ConsoleLogger("FlexivRobot")
        self.robot = {"gripper_robot":None}

        self.init_robot('gripper_robot')
        self.robot['gripper_robot'].setMode(self.mode.NRT_JOINT_POSITION)
        self.init_gripper()

        self.joint_limits_low = np.array([-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671]) + 0.1
        self.joint_limits_high = np.array([2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 2.9671]) - 0.1

    def go_init_joint(self):
        # init_gripper_joints = np.array([-0.57789973, -0.70800994,  0.53129942,  1.58402704,  0.31958843, 1.0613998 , -0.79267218])
        # init_ft_joints = np.array([1.32362411, -1.25974886, -0.91948404,  1.96467796,  0.31982947,1.62020271,  1.17286422])


        # init_gripper_joints = np.array([-0.48694, -0.74843, 0.451392, 1.56708, 0.28096, 1.096458, -0.6841])
        # init_ft_joints = np.array([1.381740, -1.236994, -0.93122, 2.100697, 0.37159, 1.663677, 1.17685])

        init_gripper_joints = np.array([-0.586136, -0.745720, 0.52749, 1.55923, 0.3157, 1.063024, -0.81544])

        target_vel = [0.0]*7 + [0.05]
        target_acc = [0.0]*7 + [10]
        qinit = init_gripper_joints.tolist() + [0.06]

        max_vel=[0.1] * 7
        max_acc=[0.3] * 7
        print("You must be aware of the movements that will occur during the initialization of the robotic arm")
        import pdb;pdb.set_trace()
        self.update_joint_state(qinit, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
        import pdb;pdb.set_trace()
    
    def set_zero_ft(self,name='gripper_robot'):
        self.robot[name].setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot[name].executePrimitive("ZeroFTSensor()")
        while self.robot[name].isBusy():
            time.sleep(1)
        self.log.info("Sensor zeroing complete")
        self.robot[name].setMode(self.mode.NRT_JOINT_POSITION)

    
    def update_joint_state(self, target_pos, target_vel=[0.0]*7 + [0.05], target_acc=[0.0]*7 + [0.05], max_vel=[0.2] * 7, max_acc=[0.3] * 7):
        '''
        8-dof: target [:7] gripper robot cmd, target[7] gripper cmd
        7-dof: max_vel, max_acc
        '''
        gripper_pos, gripper_vel, gripper_acc, gripper_max_vel, gripper_max_acc = \
            target_pos[:7], target_vel[:7], target_acc[:7], max_vel[:7], max_acc[:7]

        gripper_width = target_pos[7]
        gripper_velocity = target_vel[7]
        gripper_force = target_acc[7]

        gripper_pos = np.clip(np.array(gripper_pos), self.joint_limits_low, self.joint_limits_high).tolist()
        self.robot['gripper_robot'].sendJointPosition(gripper_pos, gripper_vel, gripper_acc, gripper_max_vel, gripper_max_acc)
        self.gripper.move(gripper_width, gripper_velocity, gripper_force)
    
    def update_trajectory_state(self, target_pos_list, target_vel, target_acc, max_vel=[0.2] * 7, max_acc=[0.3] * 7):
        '''
        [8-dof]: target [:7] gripper robot cmd, target[7] gripper cmd
        7-dof: max_vel, max_acc
        '''
        for target_pos in target_pos_list:
            gripper_dis, width_dis = self.get_delta_q(target_pos)
            self.update_joint_state(target_pos, target_vel, target_acc, max_vel, max_acc)
            while gripper_dis > 0.02 or width_dis > 0.01:
                time.sleep(0.01)
                gripper_dis, width_dis = self.get_delta_q(target_pos)
            
    
    def get_delta_q(self, target_pos):
        gripper_pos, gripper_width = target_pos[:7], target_pos[7]
        current_q = self.get_q()
        gripper_dis = np.max(np.abs(np.array(current_q[:7]) - np.array(gripper_pos)))
        width_dis = np.abs(current_q[7]*2 - gripper_width)
        return gripper_dis,width_dis

    
    def init_gripper(self):
        set_mode = False
        if self.robot['gripper_robot'] == self.mode.IDLE:
            self.robot['gripper_robot'].setMode(self.mode.NRT_JOINT_POSITION)
            set_mode = True
        self.gripper = flexivrdk.Gripper(self.robot['gripper_robot'])
        self.gripper_states = flexivrdk.GripperStates()
        if set_mode:
            self.robot['gripper_robot'].setMode(self.mode.IDLE)
    
    def _get_gripper_state(self):
        if self.robot['gripper_robot'] == self.mode.IDLE:
            self.log.info('Gripper control is not available if the robot is in IDLE mode')
        else:
            self.gripper.getGripperStates(self.gripper_states)
            # print("width: ", round(self.gripper_states.width, 2))
            # print("force: ", round(self.gripper_states.force, 2))
            # print("max_width: ", round(self.gripper_states.maxWidth, 2))
            # print("is_moving: ", self.gripper_states.isMoving)
        return self.gripper_states

    def get_gripper_width(self):
        return round(self._get_gripper_state().width,4)
    
    def get_q(self):
        gripper_joint = self.get_joint_pos('gripper_robot')
        gripper_width = self.get_gripper_width()
        return gripper_joint.tolist() + [gripper_width/2,gripper_width/2]
    
    def get_ext_wrench(self,name='gripper_robot',base=True):
        if base:
            return np.array(self._get_robot_status(name).extWrenchInBase)
        else:
            return np.array(self._get_robot_status(name).extWrenchInTcp)
    
    def init_robot(self, name="gripper_robot"):
        robot_ip = self.robot_addr[name]
        robot = flexivrdk.Robot(robot_ip)
        # Clear fault on robot server if any
        if robot.isFault():
            self.log.info("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                self.log.info("Fault cannot be cleared, exiting ...")
                return
            self.log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        self.log.info(f"Enabling {name} ...")
        robot.enable()

        # Wait for the robot to become operational
        seconds_waited = 0
        while not robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                self.log.info("Still waiting for robot to become operational, please check that the robot 1) has no fault, 2) is in [Auto (remote)] mode.")

        self.log.info(f"{name} is now operational")
        self.robot[name] = robot
    
    def enable(self, name, max_time=10):
        """Enable robot after emergency button is released."""
        robot = self.robot[name]
        robot.enable()
        tic = time.time()
        while not self.is_operational(name):
            if time.time() - tic > max_time:
                return f"{name} enable failed"
            time.sleep(0.01)
        return
    
    def clear_fault(self, name):
        self.robot[name].clearFault()
    
    def is_fault(self,name):
        """Check if robot is in FAULT state."""
        return self.robot[name].isFault()
    
    def is_stopped(self, name):
        """Check if robot is stopped."""
        return self.robot[name].isStopped()
    
    def is_connected(self,name):
        """return if connected.

        Returns: True/False
        """
        return self.robot[name].isConnected()
    
    def is_operational(self,name):
        """Check if robot is operational."""
        return self.robot[name].isOperational()
    
    def _get_robot_status(self,name):
        self.robot[name].getRobotStates(self.robot_states[name])
        return self.robot_states[name]
    
    def get_tcp_pose(self,name, matrix=False):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if matrix:
            tcppose = np.array(self._get_robot_status(name).tcpPose)
            pose = np.identity(4)
            # pose[:3,:3] = Rot.from_quat(np.array(tcppose[4:].tolist()+[tcppose[3]])).as_matrix()
            # pose[:3,:3] = Rot.from_quat(tcppose[3:],scalar_first=True).as_matrix()
            pose[:3,:3] = Rot.from_quat(np.array([tcppose[4],tcppose[5],tcppose[6],tcppose[3]])).as_matrix()
            pose[:3,3] = np.array(tcppose[:3])
            return pose
        return np.array(self._get_robot_status(name).tcpPose)
    
    def get_tcp_vel(self,name):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status(name).tcpVel)
    
    def get_joint_pos(self,name):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status(name).q)
    
    def get_joint_vel(self,name):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status(name).dq)
    

if __name__ == "__main__":
    flexiv_robot = FlexivRobot()
    
    gripper_joints = np.array([-0.00507624, -0.56437659, 0.003704, 1.504975, 0.0027480, 0.4870387, 0.011373])
    gripper_rest_joints = np.array([0.2181339 , -0.68337023, -0.06261784,  1.70697415, -0.04558531, 1.08509004,  0.18694383])
    init_gripper_joints = np.array([-0.57789973, -0.70800994,  0.53129942,  1.58402704,  0.31958843, 1.0613998 , -0.79267218])
    
    q1 = gripper_rest_joints.tolist() + [0.09] 
    q2 = gripper_joints.tolist() + [0.01] 
    target_vel = [0.0]*7 + [0.05] 
    target_acc = [0.0]*7 + [10]

    qinit = init_gripper_joints.tolist() + [0.09]

    max_vel=[0.2] * 7
    max_acc=[0.3] * 7

    import pdb;pdb.set_trace()
    flexiv_robot.go_init_joint()
    flexiv_robot.set_zero_ft()
    flexiv_robot.get_ext_wrench()

    flexiv_robot.update_joint_state(qinit, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
    flexiv_robot.update_joint_state(q1, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
    flexiv_robot.update_joint_state(q2, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
