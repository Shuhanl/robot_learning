from tactile import PyTac3D
from robot.flexiv import FlexivRobot
import numpy as np
import time
import subprocess

class TactileSlam(object):
    def __init__(self, tac3d_config, tac3d_port=9988):
        """
        Initialize the Tactile Data Collector, including the sensor, robot, and transformations.
        """
        self.tac3d_config = tac3d_config
        self.tac3d_port = tac3d_port
        self.tac3d_process = None
        self.tac3d = None

        # Transformation matrix from the tactile sensor frame to the robot end-effector frame
        self.T_sensor_to_ee = np.eye(4)  

        # Data storage
        self.coord_data = []
        self.friction_data = []
        self.stiffness_data = []

    def start_tac3d_sensor(self):
        """
        Start the Tac3D sensor process via a subprocess call.
        """
        try:
            self.tac3d_process = subprocess.Popen(
                ['./tactile/Tac3D', '-c', self.tac3d_config, '-d', '0', '-i', '127.0.0.1', '-p', str(self.tac3d_port)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print("Tac3D sensor process started...")
            time.sleep(5)  # Give time for the sensor to initialize
        except Exception as e:
            print(f"Failed to start Tac3D sensor process: {e}")
            exit(1)

        self.tac3d = PyTac3D.Sensor(port=self.tac3d_port, maxQSize=5)

    def calibrate_sensor(self):
        """
        Calibrate the Tac3D sensor.
        """
        self.tac3d.waitForFrame()
        time.sleep(5)  # Wait for sensor to connect
        SN = self.tac3d.getFrame()['SN']
        self.tac3d.calibrate(SN)
        time.sleep(5)  # Wait for calibration

    def compute_world_coordinates(self, tactile_positions, ee_pose):
        """
        Compute the world coordinates (x, y, z) of each point in the tactile sensor grid.
        
        :param tactile_positions: 400x3 array of tactile sensor grid positions in the local sensor frame.
        :param ee_pose: 4x4 transformation matrix representing the robot's end-effector pose in the world frame.
        :return: point_cloud: 400x3 array of points transformed into world coordinates.
        """
        # Compute the full transformation from the tactile sensor frame to the world frame
        T_sensor_to_world = ee_pose @ self.T_sensor_to_ee
        
        # Convert tactile_positions (400x3) to homogeneous coordinates (400x4)
        num_points = tactile_positions.shape[0]
        ones = np.ones((num_points, 1))  # Create a column of ones for homogeneous coordinates
        tactile_points_hom = np.hstack((tactile_positions, ones))  # Shape: (400, 4)

        # Apply the transformation matrix to each point (sensor frame to world frame)
        world_points_hom = (T_sensor_to_world @ tactile_points_hom.T).T  # Shape: (400, 4)

        # Extract the x, y, z coordinates from the resulting homogeneous coordinates
        point_cloud = world_points_hom[:, :3]

        return point_cloud


    def collect_tactile_data(self, robot_pose):
        """
        Collect tactile data from the Tac3D sensor and the Flexiv robot over multiple poses.
        
        :param num_poses: Number of poses to collect data for.
        """

        P = self.tac3d.getFrame()['3D_Positions']
        D = self.tac3d.getFrame()['3D_Displacements']
        F = self.tac3d.getFrame()['3D_Forces']

        friction = np.zeros((400, 1))  # 400x1 array for friction
        stiffness = np.zeros((400, 1))  # 400x1 array for stiffness

        for idx in range(400):
            Fx, Fy, Fz = F[idx]      # Forces in x, y, z directions
            _, _, z_disp = D[idx]    # Displacement in z direction

            if Fz != 0:
                friction[idx] = np.sqrt(Fx**2 + Fy**2) / Fz
                stiffness[idx] = Fz / z_disp if z_disp != 0 else 0
            else:
                friction[idx] = 0
                stiffness[idx] = 0

        # Compute the point cloud for this frame using the combined transformation
        coord = self.compute_world_coordinates(P, robot_pose)

        # Append friction, stiffness, and coordinates for the current frame
        self.coord_data.append(coord)
        self.friction_data.append(friction)
        self.stiffness_data.append(stiffness)

    def save_data(self, filename="tactile.npz"):
        """
        Save the collected tactile data (coordinates, friction, stiffness) to a .npz file.
        
        :param filename: Filename to save the tactile data.
        """
        # Convert lists to numpy arrays and flatten them

        coord_data = np.vstack(self.coord_data)  
        friction_data = np.vstack(self.friction_data)  
        stiffness_data = np.vstack(self.stiffness_data) 

        np.savez(filename, coords=coord_data, friction=friction_data, stiffness=stiffness_data)
        print(f"Tactile data saved to {filename}")


# Example usage
if __name__ == "__main__":

    tactile_slam = TactileSlam(tac3d_config='tactile/A1-0082R')
    flexiv_robot = FlexivRobot()

    # Start the Tac3D sensor
    tactile_slam.start_tac3d_sensor()
    # Calibrate the sensor
    tactile_slam.calibrate_sensor()

    flexiv_robot.move_to_home()
    flexiv_robot.set_zero_ft()

    home_pose = flexiv_robot.get_tcp_pose(euler=True, degree=True)
    x, y, z, roll, pitch, yaw = home_pose
    x = x - 0.15
    z = z + 0.25

    position_swing = 0.15
    rotation_swing = 12
    
    calib_poses = [
        [x, y, z, roll, pitch, yaw],  
        [x + position_swing, y, z, roll, pitch - rotation_swing, yaw + rotation_swing],  
        # [x - position_swing, y, z, roll, pitch + rotation_swing, yaw], 
        # [x, y + position_swing, z, roll + rotation_swing, pitch, yaw],  
        # [x, y - position_swing, z, roll - rotation_swing, pitch, yaw],  
    ]

    for pose in calib_poses:
        flexiv_robot.cartesian_motion_force_control(pose)
        robot_pose = flexiv_robot.get_tcp_pose(matrix = True)
        time.sleep(1)
        tactile_slam.collect_tactile_data(robot_pose)

    # Save the tactile data to a file
    tactile_slam.save_data("tactile.npz")
