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

        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)

        self.log.info("Initializing gripper, this process takes about 10 seconds ...")
        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper.Init()
        self.log.info("Initialization complete")

    def move_to_home(self):
        # Move robot to home pose
        self.log.info("Moving to home pose")
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("Home()")

        # Wait for the primitive to finish
        while self.robot.busy():
            time.sleep(1)
        
    def joint_control(self, target_pose, target_vel, target_acc, max_vel=[0.2] * 7, max_acc=[0.3] * 7):
        '''
        [7-dof]: target [:7] gripper robot cmd
        7-dof: max_vel, max_acc
        '''
        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
        joint_pos, joint_vel, joint_acc, joint_max_vel, joint_max_acc = \
            target_pose[:7], target_vel[:7], target_acc[:7], max_vel[:7], max_acc[:7]

        joint_pos = np.clip(np.array(joint_pos), self.joint_limits_low, self.joint_limits_high).tolist()
        self.robot.SendJointPosition(joint_pos, joint_vel, joint_acc, joint_max_vel, joint_max_acc)

        actual_pose = self.get_joint_pos()
        q_diff = np.max(np.abs(np.array(target_pose) - actual_pose))

        while q_diff > 0.02:
            time.sleep(0.01)
            actual_pose = self.get_q()
            q_diff = np.max(np.abs(np.array(target_pose) - np.array(actual_pose)))

            
    def cartesian_motion_force_control(self, target_pose, vel = 0.1, angleVel=15):
        """
        Perform Cartesian motion force control.

        Args:
            target_pose: 6D (x, y, z, roll, pitch, yaw), or 7D (x, y, z, rw, rx, ry, rz)
        """
        
        target_pose_euler = []
        if len(target_pose)==7:
            translation = np.array(target_pose[:3])  # [x, y, z]
            quaternion = target_pose[3:]  # [qw, qx, qy, qz]
            euler_angles = self.quat2eulerZYX(quaternion, degree=True)
            target_pose_euler = np.concatenate([translation, euler_angles])
        else:
            target_pose_euler = target_pose

        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("MoveL(target=" + 
                    self.list2str(target_pose_euler) + "WORLD WORLD_ORIGN," + 
                    "vel=" + str(vel) + "," + "angleVel=" + str(angleVel) + ")")

        # Wait for reached target
        while self.parse_pt_states(self.robot.primitive_states(), "reachedTarget") != "1":
            time.sleep(1)

    def search_contact(self, contact_dir = [0, 0, -1], max_contact_force = 10):

        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("Contact(contactDir=" + self.list2str(contact_dir) + ","
                                    + "maxContactForce=" + str(max_contact_force) + ","
                                    + "enableFineContact=" + str(1) + ")")

        # Wait for reached target
        while self.parse_pt_states(self.robot.primitive_states(), "curContactForce") != "":
            time.sleep(1)
            

    def gripper_control(self, gripper_width, gripper_velocity, gripper_force):
        self.gripper.Move(gripper_width, gripper_velocity, gripper_force)

        actual_width, actual_force = self.get_gripper_states()
        width_diff, force_diff = abs(gripper_width-actual_width), abs(gripper_force-actual_force)

        while width_diff > 0.01 or force_diff > 0.2:
            time.sleep(0.01)
            actual_width, actual_force = self.get_gripper_states()
            width_diff, force_diff = abs(gripper_width-actual_width), abs(gripper_force-actual_force)

    def set_zero_ft(self):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor()")

        while self.robot.busy():
            time.sleep(1)
        self.log.info("Sensor zeroing complete")

    
    def get_gripper_states(self):

        return self.gripper.states().width, self.gripper.states().force

    
    def get_ext_wrench(self,base=True):
        if base:
            return np.array(self.robot.states().ext_wrench_in_world)
        else:
            return np.array(self.robot.states().ext_wrench_in_tcp)
            
    def is_connected(self):
        """return if connected.
        Returns: True/False
        """
        return self.robot.connected()    
    
    def get_tcp_pose(self, matrix=False, euler=False, degree=False):
        """Get current robot's tool pose in world frame.

        Args:
            matrix (bool): If True, return the pose as a 4x4 transformation matrix.
            euler (bool): If True, return the pose with Euler angles (x, y, z, roll, pitch, yaw).
        
        Returns:
            If matrix is True: 4x4 transformation matrix.
            If euler is True: 6D pose (x, y, z, roll, pitch, yaw) using Euler angles.
            Else: 7D list consisting of (x, y, z, rw, rx, ry, rz) using quaternions.

        Raises:
            RuntimeError: Error occurred when mode is None.
        """
        tcppose = np.array(self.robot.states().tcp_pose)
        
        # If matrix option is selected, return the pose as a 4x4 transformation matrix
        if matrix:
            pose = np.identity(4)
            pose[:3, :3] = Rot.from_quat([tcppose[4], tcppose[5], tcppose[6], tcppose[3]]).as_matrix()
            pose[:3, 3] = tcppose[:3]
            return pose

        # If euler option is selected, convert quaternion to Euler angles
        if euler:
            translation = np.array(tcppose[:3])  # [x, y, z]
            quaternion = tcppose[3:]  # [qw, qx, qy, qz]
            euler_angles = self.quat2eulerZYX(quaternion, degree)

            return np.concatenate([translation, euler_angles])

        # Default: return pose as 7D list (x, y, z, rw, rx, ry, rz)
        return tcppose
    
    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot.states().tcp_vel)
    
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
        return np.array(self.robot.states().dq)

    def quat2eulerZYX(self, quat, degree=False):
        """
        Convert quaternion to Euler angles with ZYX axis rotations.

        Parameters
        ----------
        quat : float list
            Quaternion input in [w,x,y,z] order.
        degree : bool
            Return values in degrees, otherwise in radians.

        Returns
        ----------
        float list
            Euler angles in [x,y,z] order, radian by default unless specified otherwise.
        """

        # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
        # NOTE: scipy uses [x,y,z,w] order to represent quaternion
        eulerZYX = (
            Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
            .as_euler("xyz", degrees=degree)
            .tolist()
        )

        return eulerZYX


    def list2str(self, ls):
        """
        Convert a list to a string.

        Parameters
        ----------
        ls : list
            Source list of any size.

        Returns
        ----------
        str
            A string with format "ls[0] ls[1] ... ls[n] ", i.e. each value
            followed by a space, including the last one.
        """

        ret_str = ""
        for i in ls:
            ret_str += str(i) + " "
        return ret_str


    def parse_pt_states(self, pt_states, parse_target):
        """
        Parse the value of a specified primitive state from the pt_states string list.

        Parameters
        ----------
        pt_states : str list
            Primitive states string list returned from Robot::primitive_states().
        parse_target : str
            Name of the primitive state to parse for.

        Returns
        ----------
        str
            Value of the specified primitive state in string format. Empty string is
            returned if parse_target does not exist.
        """
        for state in pt_states:
            # Split the state sentence into words
            words = state.split()

            if words[0] == parse_target:
                return words[-1]

        return ""


if __name__ == "__main__":
    flexiv_robot = FlexivRobot()

    flexiv_robot.move_to_home()
    flexiv_robot.set_zero_ft()
    flexiv_robot.search_contact()
    
    # ext_wrench = flexiv_robot.get_ext_wrench()
    # print("External wrench:", ext_wrench)
    # tcp_pose = flexiv_robot.get_tcp_pose(euler=True, degree=True)
    # print("TCP pose:", tcp_pose)
    # flexiv_robot.init_gripper()
    # gripper_states = flexiv_robot.get_gripper_states()
    # print("Gripper states:", gripper_states)

    # cartesian_list = [[0.65, -0.3, 0.2, 180, 0, 180],
    #                   [0.65, 0, 0.2, 180, 0, 180],
    #                   [0.65, -0.3, 0.3, 180, 0, 180]]
    
    # for cartesian in cartesian_list:
    #     flexiv_robot.cartesian_motion_force_control(cartesian)
    #     time.sleep(1)

    # joints_list = [[-0.57789973, -0.70800994,  0.53129942,  1.58402704,  0.31958843, 1.0613998 , -0.79267218], 
    #                [-0.67789973, -0.80800994,  0.53129942,  1.58402704,  0.31958843, 1.0613998 , -0.79267218]]
    
    # target_vel = [0.0]*7  
    # target_acc = [0.0]*7 
    # max_vel=[0.2] * 7
    # max_acc=[0.3] * 7

    # for joints in joints_list:
    #     flexiv_robot.joint_control(joints, target_vel, target_acc, max_vel=max_vel, max_acc=max_acc)
    #     time.sleep(1)
    
    # gripper_list = [[0.01, 0.1, 20], [0.09, 0.1, 20], [0.01, 0.1, 20]]
    # for gripper in gripper_list:
    #     flexiv_robot.gripper_control(gripper[0], gripper[1], gripper[2])
    #     time.sleep(1)

