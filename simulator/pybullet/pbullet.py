import pybullet as p
import time
import numpy as np
import pybullet_data
import gym


class Pbullet(gym.Env):
    def __init__(self, gui=True):
        """
        Initialize the simulator.

        Parameters:
        - gui: bool, if True, PyBullet will start in GUI mode; otherwise, in DIRECT mode for faster computation without visualization.
        """
        self.gui = gui
        self.physics_client = None
        self.arm = None
        self.hz = 240
        self.table_pos = [0, 1, 0]
        self.arm_pos = [0, 0.1, 0.5]
        self.arm_orientation = [0, 0, 90*np.pi/180]
        self.joints = None
        # self.homej = [2.0059815e-01, -2.2976594e-01, -1.8512011e-02, 1.6786172e+00,
        #               4.5142174e-03, 3.3744574e-01, 1.7518656e+00,0.0000000e+00,
        #               0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 9.5367432e-07,
        #               0.0000000e+00, 0.0000000e+00]
        self.homej = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
                      -0.299912, 0.000000, -0.000043]

        self.ee_index = 6
        self.gripper_index = 7
        self.finger_tip_force = 2
        self.ee_angle = 0
        self.fingerA_force = 2
        self.fingerB_force = 2.5
        self.max_force = 200.
        # Camera parameters
        # Camera position in world coordinates
        self.camera_position = [0, -1, 3]
        # Where the camera is pointing, in world coordinates
        self.camera_target = [0, 0.8, -0.1]
        self.camera_view_matrix = None
        self.camera_projection_matrix = None
        self.camera_far_val = 100  # Far clipping plane distance
        self.fov = 60
        self.aspect = 1.0
        self.nearVal = 0.1
        self.width = 640
        self.height = 480

        image_size = (480, 640)
        proprioception_dim = 6
        color_tuple = [
            gym.spaces.Box(low=0, high=255, shape=image_size +
                           (3,), dtype=np.uint8)
        ]
        depth_tuple = [gym.spaces.Box(
            low=0.0, high=20.0, shape=image_size, dtype=np.float32)]
        proprioception_tuple = [gym.spaces.Box(
            low=0.0, high=5.0, shape=(proprioception_dim,), dtype=np.float32)]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
            'proprioception': gym.spaces.Tuple(proprioception_tuple)
        })
        position_tuple = [
            gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)]
        gripper_tuple = [
            gym.spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32)]
        orientation_tuple = [
            gym.spaces.Box(-np.pi, np.pi, shape=(3,), dtype=np.float32)]
        self.action_space = gym.spaces.Dict({
            'position': gym.spaces.Tuple(position_tuple),
            'orientation': gym.spaces.Tuple(orientation_tuple),
            'gripper': gym.spaces.Tuple(gripper_tuple)
        })

    def connect(self):
        """
        Connect to the PyBullet simulation.
        """
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setRealTimeSimulation(True, self.physics_client)
        p.setTimeStep(1. / self.hz)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        p.setGravity(0, 0, -9.8)

        # View parameters
        p.resetDebugVisualizerCamera(cameraDistance=2,  # Distance from the target point
                                     cameraYaw=90, cameraPitch=-30,
                                     cameraTargetPosition=[0, 1, 0])  # XYZ position in the world where the camera looks at

    def load_object(self):
        self.arm_orientation = p.getQuaternionFromEuler(self.arm_orientation)
        p.loadURDF("table/table.urdf", self.table_pos, useFixedBase=True)
        p.loadURDF("cube_small.urdf", [0, 1, 0.8])
        # self.arm = p.loadURDF("../../flexiv_workcell_builder/components/system_description/flexiv/urdf/system1.urdf",
        #                  self.arm_pos, self.arm_orientation, useFixedBase=True)
        self.arm = p.loadSDF("kuka_iiwa/kuka_with_gripper.sdf")[0]

        p.resetBasePositionAndOrientation(
            self.arm, self.arm_pos, self.arm_orientation)

        # Initialize camera
        # Up direction for the camera, usually the world's up vector
        up_vector = [0, 0, 1]
        self.camera_view_matrix = p.computeViewMatrix(
            self.camera_position, self.camera_target, up_vector)
        self.camera_projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                                     aspect=self.aspect, nearVal=self.nearVal,
                                                                     farVal=self.camera_far_val)

    def get_robot_info(self):
        """ Joint info returns a tuple with the following structure:
            (jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction,
            jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, linkName,
            jointAxis, parentFramePos, parentFrameOrn, parentIndex) """

        num_joints = p.getNumJoints(self.arm)
        for joint_index in range(num_joints):
            # Get joint info to retrieve the link name
            joint_info = p.getJointInfo(self.arm, joint_index)
            joint_state = p.getJointState(self.arm, joint_index)
            link_name = joint_info[12].decode('utf-8')

            # Get the link state to find the world position and orientation
            link_state = p.getLinkState(self.arm, joint_index)
            joint_angle = joint_state[0]  # The position (angle) of the joint
            # World position of the URDF link frame
            world_position = link_state[4]
            # World orientation of the URDF link frame in quaternion
            world_orientation_quat = link_state[5]

            # Convert quaternion to Euler angles for easier interpretation
            world_orientation_euler = p.getEulerFromQuaternion(
                world_orientation_quat)

            # Print link information
            print(f"Link Index: {joint_index}")
            print(f"Link Name: {link_name}")
            print(f"Joint Angle: {joint_angle}")
            print(f"Link World Position: {world_position}")
            print(f"Link World Orientation (Euler): {world_orientation_euler}")
            print("-" * 40)

    def _get_proprioception(self):

        state = p.getLinkState(self.arm, self.gripper_index)
        print(state)
        # Extract position and orientation from the state
        position = state[4]  # World position of the URDF link frame
        # World orientation of the URDF link frame (quaternion)
        orientation = state[5]

        orientation = p.getEulerFromQuaternion(orientation)
        proprioception = np.concatenate((position, orientation))

        return proprioception

    def _get_vision(self):
        # Capture an image
        width, height, rgb_img, depth_img, _ = p.getCameraImage(width=self.width,
                                                                height=self.height, viewMatrix=self.camera_view_matrix,
                                                                projectionMatrix=self.camera_projection_matrix)

        # Convert depth image to depth values
        depth_image = np.array(depth_img).reshape(
            (height, width)) * self.camera_far_val / 255.0

        return rgb_img, depth_image

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = {'color': (), 'depth': (), 'proprioception': ()}
        color, depth = self._get_vision()
        obs['color'] += (color,)
        obs['depth'] += (depth,)
        obs['proprioception'] += (self._get_proprioception(),)

        return obs

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.arm)
        joints = [p.getJointInfo(self.arm, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home configuration.
        self.move_home()

        obs = self.step()

        return obs

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
        action: action to execute.

        Returns:
        (obs, reward, done, info) tuple containing MDP step data.
        """

        if action is not None:
            timeout = self.movep(action['position'], action['orientation'])
            self.gripper_control(action['gripper'][0], action['gripper'][1])

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self._get_obs()
                return obs, 0.0, True

        # Step simulator asynchronously until objects settle.
        # while not self.is_static:
        #     p.stepSimulation()

        # Get task rewards.
        # reward, info = self.task.reward() if action is not None else (0, {})
        # done = self.task.done()

        obs = self._get_obs()

        return obs

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def solve_ik(self, position, orietnation):
        """Calculate joint configuration with inverse kinematics."""
        orientation = p.getQuaternionFromEuler(orietnation)
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.arm,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=position,
            targetOrientation=orientation,
            # lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            # upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            # jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=self.homej,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def movej(self, targj, speed=0.01, timeout=5):
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.arm, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.arm,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, position, orietnation, speed=0.01):
        targj = self.solve_ik(position, orietnation)
        return self.movej(targj, speed)

    def move_home(self, speed=0.01):
        self.movej(self.homej, speed)

    def gripper_control(self, da, finger_angle):
        self.ee_angle += da
        p.setJointMotorControl2(self.arm,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=self.ee_angle,
                                force=self.max_force)
        p.setJointMotorControl2(self.arm,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=-finger_angle,
                                force=self.fingerA_force)
        p.setJointMotorControl2(self.arm,
                                11,
                                p.POSITION_CONTROL,
                                targetPosition=finger_angle,
                                force=self.fingerB_force)

        p.setJointMotorControl2(self.arm,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.finger_tip_force)
        p.setJointMotorControl2(self.arm,
                                13,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.finger_tip_force)
