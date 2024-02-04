import pybullet as p
import time
import pybullet_data

class RobotSimulator:
    def __init__(self, gui=True):
        """
        Initialize the simulator.

        Parameters:
        - gui: bool, if True, PyBullet will start in GUI mode; otherwise, in DIRECT mode for faster computation without visualization.
        """
        self.gui = gui
        self.physicsClient = None
        self.robot_id = None

    def connect(self):
        """
        Connect to the PyBullet simulation.
        """
        if self.gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def load_robot(self, urdf_path, initial_position=[0,0,1]):
        """
        Load the robot from a URDF file.

        Parameters:
        - urdf_path: str, the path to the URDF file of the robot.
        - initial_position: list of float, the initial position of the robot in the simulation.
        """
        self.robot_id = p.loadURDF(urdf_path, initial_position)
        return self.robot_id

    def set_gravity(self, gravity=-9.81):
        """
        Set the gravity for the simulation.

        Parameters:
        - gravity: float, the gravity value (in m/s^2).
        """
        p.setGravity(0, 0, gravity)

    def simulate(self, steps=1000, time_step=1./240.):
        """
        Run the simulation.

        Parameters:
        - steps: int, the number of simulation steps to execute.
        - time_step: float, the time step of the simulation in seconds.
        """
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(time_step)

    def disconnect(self):
        """
        Disconnect the simulation.
        """
        p.disconnect()

    # Visualization methods
    def draw_debug_frame(self, position, orientation, frame_size=0.1, life_time=0):
        """
        Draw a debug frame at a given position and orientation.

        Parameters:
        - position: tuple or list, the 3D position of the frame.
        - orientation: tuple or list, the orientation of the frame as a quaternion.
        - frame_size: float, the size of the frame axes.
        - life_time: float, the time in seconds the frame will be drawn for. 0 (default) means permanent until reset.
        """
        # Convert quaternion to rotation matrix to get frame axes
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        x_axis = rot_matrix[0:3]
        y_axis = rot_matrix[3:6]
        z_axis = rot_matrix[6:9]

        # Draw frame axes
        p.addUserDebugLine(position, [position[0] + frame_size * x_axis[0], position[1] + frame_size * x_axis[1], position[2] + frame_size * x_axis[2]], [1, 0, 0], lifeTime=life_time)
        p.addUserDebugLine(position, [position[0] + frame_size * y_axis[0], position[1] + frame_size * y_axis[1], position[2] + frame_size * y_axis[2]], [0, 1, 0], lifeTime=life_time)
        p.addUserDebugLine(position, [position[0] + frame_size * z_axis[0], position[1] + frame_size * z_axis[1], position[2] + frame_size * z_axis[2]], [0, 0, 1], lifeTime=life_time)




