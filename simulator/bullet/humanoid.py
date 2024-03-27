import gym
import pybullet as p
import pybullet_data
import numpy as np

""" OpenAI Gym humanoid environment wrapper """
class HumanoidWrapper(gym.Wrapper):
    def __init__(self, env):
        super(HumanoidWrapper, self).__init__(env)

    
    def reset(self):
        """
        Reset the environment, add a box, and modify the initial observation.
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        Take a step using the given action, modify the reward, or add custom logic.
        """
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        """
        return self.env.render(mode)


""" PyBullet Humanoid Environment """
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

    def connect(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / self.hz)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # View parameters
        p.resetDebugVisualizerCamera(cameraDistance=20,  # Distance from the target point
                                     cameraYaw=90, cameraPitch=-30,
                                     cameraTargetPosition=[0, 1, 0])  # XYZ position in the world where the camera looks at

    def load_object(self):
        p.loadURDF("plane.urdf")
        self.humanoid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1], useFixedBase=False)
        
    def reset(self):
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

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed