from tactile import PyTac3D
from robot.flexiv import FlexivRobot
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
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
            time.sleep(3)  # Give time for the sensor to initialize
        except Exception as e:
            print(f"Failed to start Tac3D sensor process: {e}")
            exit(1)

        self.tac3d = PyTac3D.Sensor(port=self.tac3d_port, maxQSize=5)

    def calibrate_sensor(self):
        """
        Calibrate the Tac3D sensor.
        """
        self.tac3d.waitForFrame()
        time.sleep(3)  # Wait for sensor to connect
        SN = self.tac3d.getFrame()['SN']
        self.tac3d.calibrate(SN)
        time.sleep(3)  # Wait for calibration

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

        if self.tac3d.getFrame() is not None:

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

    def generate_3d_scan_trajectory(self, object_position, num_points_per_segment=3, num_zigzags=2,
                                    amplitude=0.05, total_length=0.3):
        '''
        Generates a zig-zag trajectory along the xy plane with the end-effector pointing downward.

        :param object_position: 3D list [x, y, z] representing the location of the object in world coordinates.
        :param num_points_per_segment: Number of points for the zig-zag trajectory.
        :param num_zigzags: Number of zig-zag segments.
        :param amplitude: Amplitude of the zig-zag (peak deviation from the central line).
        :param total_length: Total length of the trajectory along the x direction.
        :param height_offset: Height above the object in z-direction.
        :return: List of poses in 6D format [x, y, z, roll, pitch, yaw].
        '''

        # Create an empty list to store the trajectory poses
        trajectory_poses = []

        # === Generate zig-zag trajectory along the xy plane ===
        for i in range(num_points_per_segment):
            fraction = i / (num_points_per_segment - 1)
            x = object_position[0] + fraction * total_length - total_length / 2  # Moving along x-axis
            y = object_position[1] + amplitude * np.sin(fraction * num_zigzags * np.pi)  # Zig-zag motion in y
            z = object_position[2] 

            roll = object_position[3]    
            pitch = object_position[4]
            yaw = object_position[5]

            # Create the 6D pose [x, y, z, roll, pitch, yaw]
            pose = np.array([x, y, z, roll, pitch, yaw])
            trajectory_poses.append(pose)

        return trajectory_poses


    def plot_trajectory(self, trajectory_poses, object_position=None, show_orientation=False, scale=0.02):
        '''
        Plots the 3D trajectory generated by generate_3d_scan_trajectory.

        :param trajectory_poses: List of poses in 6D format [x, y, z, roll, pitch, yaw].
        :param object_position: Optional 3D list [x, y, z] representing the location of the object.
        :param show_orientation: Boolean indicating whether to display the end-effector orientation.
        :param scale: Scale factor for the orientation arrows.
        '''

        # Extract x, y, z coordinates from the trajectory
        x = [pose[0] for pose in trajectory_poses]
        y = [pose[1] for pose in trajectory_poses]
        z = [pose[2] for pose in trajectory_poses]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(x, y, z, label='Trajectory', color='blue')
        ax.scatter(x, y, z, c='blue', marker='o')

        # Plot the orientation at each point if requested
        if show_orientation:
            for pose in trajectory_poses:
                pos = pose[:3]
                roll, pitch, yaw = pose[3], pose[4], pose[5]
                # Convert Euler angles to rotation matrix
                rotation = Rot.from_euler('xyz', [roll, pitch, yaw])
                # Get the direction vector (e.g., the x-axis of the end-effector frame)
                direction = rotation.apply([1, 0, 0]) * scale
                # Plot the orientation as an arrow
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    direction[0], direction[1], direction[2],
                    color='green', length=scale, normalize=True
                )

        # Optionally plot the object position
        if object_position is not None:
            ax.scatter(
                [object_position[0]],
                [object_position[1]],
                [object_position[2]],
                c='red',
                marker='^',
                s=100,
                label='Object Position'
            )

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scan Trajectory')
        ax.legend()

        plt.show()


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
    flexiv_robot.search_contact()
    home_pose = flexiv_robot.get_tcp_pose(euler=True, degree=True)
    object_position = home_pose

    # Generate the trajectory
    trajectory = tactile_slam.generate_3d_scan_trajectory(
        object_position=object_position,
        num_points_per_segment=10,
        num_zigzags=2,
        amplitude=0.05,
        total_length=0.1)

    # Plot the trajectory
    tactile_slam.plot_trajectory(trajectory, object_position=object_position, show_orientation=True, scale=0.02)
    target_wrench = [0, 0, -5, 0, 0, 0]
    vel = 0.05
    
    for pose in trajectory:
        flexiv_robot.hybrid_force_control(pose, target_wrench, vel)
        robot_pose = flexiv_robot.get_tcp_pose(matrix = True)
        time.sleep(1)
        tactile_slam.collect_tactile_data(robot_pose)

    # Save the tactile data to a file
    tactile_slam.save_data("tactile.npz")
