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

    def __init__(self, robot_sn="Rizon4s-062534"):
        """
        Initialize.

        Args:
            gripper_robot_sn

        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.mode = flexivrdk.Mode
        self.robot_states = {"gripper_robot":flexivrdk.RobotStates()}

        self.robot_sn = robot_sn
        self.log = spdlog.ConsoleLogger("FlexivRobot")
        self.robot = None
        self.gripper = None

        self.init_robot()
        self.init_gripper()

        self.joint_limits_low = np.array([-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671]) + 0.1
        self.joint_limits_high = np.array([2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 2.9671]) - 0.1

    def init_robot(self):
        robot_sn = self.robot_sn
        self.robot = flexivrdk.Robot(robot_sn)

        # Clear fault on the connected robot if any
        if self.robot.fault():
            self.log.warn("Fault occurred on the connected robot, trying to clear ...")
            # Try to clear the fault
            if not self.robot.ClearFault():
                self.log.error("Fault cannot be cleared, exiting ...")
                return 1
            self.log.info("Fault on the connected robot is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        self.log.info("Enabling robot ...")
        self.robot.Enable()

        # Enable the robot, make sure the E-stop is released before enabling
        self.log.info(f"Enabling ...")
        self.robot.Enable()

        # Wait for the robot to become operational
        while not self.robot.operational():
            time.sleep(1)

        self.log.info("Robot is now operational")    

    def init_gripper(self):
        set_mode = False
        if self.robot == self.mode.IDLE:
            self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
            set_mode = True
        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper_states = flexivrdk.GripperStates()
        if set_mode:
            self.robot.SwitchMode(self.mode.IDLE)

    def move_to_home(self):
        # Move robot to home pose
        self.log.info("Moving to home pose")
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("Home()")

    def update_joint_state(self, target_pos, target_vel=[0.0]*7, target_acc=[0.0]*7, max_vel=[0.2] * 7, max_acc=[0.3] * 7):
        '''
        7-dof: target [:7] robot cmd
        7-dof: max_vel, max_acc
        '''
        gripper_pos, gripper_vel, gripper_acc, gripper_max_vel, gripper_max_acc = \
            target_pos[:7], target_vel[:7], target_acc[:7], max_vel[:7], max_acc[:7]

        gripper_pos = np.clip(np.array(gripper_pos), self.joint_limits_low, self.joint_limits_high).tolist()
        self.robot.SendJointPosition(gripper_pos, gripper_vel, gripper_acc, gripper_max_vel, gripper_max_acc)
    
    def update_gripper_states(self, gripper_width, gripper_velocity, gripper_force):
        
        self.gripper.Move(gripper_width, gripper_velocity, gripper_force)
    

    def go_init_joint(self):

        init_gripper_joints = np.array([-0.586136, -0.745720, 0.52749, 1.55923, 0.3157, 1.063024, -0.81544])

        target_vel = [0.0]*7 + [0.05]
        target_acc = [0.0]*7 + [10]
        qinit = init_gripper_joints.tolist() + [0.06]

        max_vel=[0.1] * 7
        max_acc=[0.3] * 7
        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
        # import pdb;pdb.set_trace()
        self.update_joint_state(qinit, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
        # import pdb;pdb.set_trace()
    
    def joint_control(self, target_pos_list, target_vel, target_acc, max_vel=[0.2] * 7, max_acc=[0.3] * 7):
        '''
        [8-dof]: target [:7] gripper robot cmd, target[7] gripper cmd
        7-dof: max_vel, max_acc
        '''
        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
        for target_pos in target_pos_list:
            gripper_dis, width_dis = self.get_delta_q(target_pos)
            self.update_joint_state(target_pos, target_vel, target_acc, max_vel, max_acc)
            while gripper_dis > 0.02 or width_dis > 0.01:
                time.sleep(0.01)
                gripper_dis, width_dis = self.get_delta_q(target_pos)
            
    def cartesian_motion_force_control(self, target_pose):
            
        self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SendCartesianMotionForce(target_pose)

    def get_delta_q(self, target_pos):
        gripper_pos, gripper_width = target_pos[:7], target_pos[7]
        current_q = self.get_q()
        gripper_dis = np.max(np.abs(np.array(current_q[:7]) - np.array(gripper_pos)))
        width_dis = np.abs(current_q[7]*2 - gripper_width)
        return gripper_dis,width_dis

    def set_zero_ft(self):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor()")
        while self.robot.busy():
            time.sleep(1)
        self.log.info("Sensor zeroing complete")
        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
    
    def _get_gripper_state(self):
        if self.robot == self.mode.IDLE:
            self.log.info('Gripper control is not available if the robot is in IDLE mode')
        else:
            self.gripper.getGripperStates(self.gripper_states)
            print("width: ", round(self.gripper_states.width, 2))
            print("force: ", round(self.gripper_states.force, 2))
            print("max_width: ", round(self.gripper_states.maxWidth, 2))
            print("is_moving: ", self.gripper_states.isMoving)
        return self.gripper_states

    def get_gripper_width(self):
        return round(self._get_gripper_state().width,4)
    
    def get_q(self):
        gripper_joint = self.get_joint_pos()
        gripper_width = self.get_gripper_width()
        return gripper_joint.tolist() + [gripper_width/2,gripper_width/2]
    
    def get_ext_wrench(self,base=True):
        if base:
            return np.array(self.robot.states().ext_wrench_in_world)
        else:
            return np.array(self.robot.states().ext_wrench_in_tcp)
        
    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.isStopped()
    
    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.connected()    
    
    def get_tcp_pose(self, matrix=False):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if matrix:
            tcppose = np.array(self.robot().tcpPose)
            pose = np.identity(4)
            # pose[:3,:3] = Rot.from_quat(np.array(tcppose[4:].tolist()+[tcppose[3]])).as_matrix()
            # pose[:3,:3] = Rot.from_quat(tcppose[3:],scalar_first=True).as_matrix()
            pose[:3,:3] = Rot.from_quat(np.array([tcppose[4],tcppose[5],tcppose[6],tcppose[3]])).as_matrix()
            pose[:3,3] = np.array(tcppose[:3])
            return pose
        return np.array(self.robot().tcpPose)
    
    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot.states().tcpVel)
    
    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot.states().q)
    
    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot().dq)
    

if __name__ == "__main__":
    flexiv_robot = FlexivRobot()
    
    init_gripper_joints = np.array([-0.57789973, -0.70800994,  0.53129942,  1.58402704,  0.31958843, 1.0613998 , -0.79267218])
    
    target_vel = [0.0]*7 + [0.05] 
    target_acc = [0.0]*7 + [10]

    qinit = init_gripper_joints.tolist() + [0.5]

    max_vel=[0.2] * 7
    max_acc=[0.3] * 7

    flexiv_robot.move_to_home()
    flexiv_robot.go_init_joint()
    flexiv_robot.set_zero_ft()
    flexiv_robot.get_ext_wrench()

    flexiv_robot.update_joint_state(qinit, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
