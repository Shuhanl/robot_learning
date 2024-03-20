import pybullet as p
import time
import numpy as np
import pybullet_data
import gym


class HumanoidEnv(gym.Env):
    def __init__(self, gui=True):
        super(HumanoidEnv, self).__init__()
        self.gui = gui
        self.physics_client = None
        self.humanoid = None
        self.hz = 240
        
        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        self.connect()

    def connect(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / self.hz)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1], useFixedBase=False)
        return self._get_obs()

    def _get_obs(self):
        # Simplified observation getter
        obs = []
        for joint in range(p.getNumJoints(self.humanoid)):
            jointState = p.getJointState(self.humanoid, joint)
            obs.extend([jointState[0], jointState[1]]) # position, velocity
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Apply the actions to the humanoid
        for joint in range(p.getNumJoints(self.humanoid)):
            p.setJointMotorControl2(self.humanoid, joint, p.POSITION_CONTROL, targetPosition=action[joint])

        p.stepSimulation()
        obs = self._get_obs()

        # Placeholder for reward and done calculation
        reward = -1.0
        done = False

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass  # Rendering is handled through PyBullet's GUI

    def close(self):
        p.disconnect()

    def _get_vision(self):
        # Capture an image
        width, height, rgb_img, depth_img, _ = p.getCameraImage(width=self.width,
                                                                height=self.height, viewMatrix=self.camera_view_matrix,
                                                                projectionMatrix=self.camera_projection_matrix)

        # Convert depth image to depth values
        depth_image = np.array(depth_img).reshape(
            (height, width)) * self.camera_far_val / 255.0

        return rgb_img, depth_image

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed







