from camera.camera import RealSenseCamera
from robot.flexiv import FlexivRobot
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import time

class VisionSLAM(object):
    def __init__(self):
        self.point_clouds = []

    def generate_3d_scan_trajectory(self, object_position, num_points_per_segment=3, xy_variation = 0.03, height_variation=0.1, height_offset=0.05):
        '''
        Generates a combined trajectory with three segments: 
        1. Move across the object in the x direction from the bottom, rising to the top, and descending back down.
        2. Move around the bottom of the object in the x-y plane while keeping z constant.
        3. Move across the object in the y direction from the bottom, rising to the top, and descending back down.

        :param home_pose: 7D list [x, y, z, rw, rx, ry, rz] representing the robot's initial pose in quaternions.
        :param object_position: 3D list [x, y, z] representing the location of the object in world coordinates.
        :param num_points_per_segment: Number of points for each segment.
        :param height_variation: The variation in height for moving below, over, and under the object in Segments 1 and 3.
        :return: List of poses in 7D format [x, y, z, rw, rx, ry, rz].
        '''

        # Create an empty list to store the trajectory poses
        trajectory_poses = []

        # === Segment 1: Move across the object in the x direction (arc motion) ===
        for i in range(num_points_per_segment):
            fraction = i / num_points_per_segment
            x = object_position[0] + (fraction - 0.5) * xy_variation  # Moving from left to right (across the object)
            y = object_position[1]  # Fixed y position
            z = object_position[2] + height_variation * np.sin(fraction * np.pi) + height_offset  # Arc motion in z, going under, over, and back under the object

            # Compute the direction vector from the current position to the object
            direction_vector = np.array(object_position) - np.array([x, y, z])
            direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector
            
            # Compute quaternion to keep pointing at the object
            z_axis = np.array([0, 0, 1])  # Assuming the robot's z-axis is the forward direction
            rotation, _ = Rot.align_vectors([direction_vector], [z_axis])
            quat = rotation.as_quat()  # Returns [rx, ry, rz, rw]
            
            # Create the 7D pose [x, y, z, rw, rx, ry, rz]
            pose = np.concatenate([[x, y, z], [quat[3], quat[0], quat[1], quat[2]]])  # [x, y, z, rw, rx, ry, rz]
            trajectory_poses.append(pose)

        # === Segment 2: Move around the object at the bottom in the x-y plane, keeping z constant ===
        for i in range(num_points_per_segment):
            fraction = i / num_points_per_segment
            angle = np.pi / 2 * fraction  # Half-circle from 12 o'clock to 3 o'clock
            x = object_position[0] + xy_variation / 2 * np.cos(angle)  # Circular motion around object in x
            y = object_position[1] + xy_variation / 2 * np.sin(angle)  # Circular motion around object in y
            z = object_position[2] + height_offset  # Keep z constant (at the bottom of the object)

            # Compute the direction vector from the current position to the object
            direction_vector = np.array(object_position) - np.array([x, y, z])
            direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector
            
            # Compute quaternion to keep pointing at the object
            z_axis = np.array([0, 0, 1])  # Assuming the robot's z-axis is the forward direction
            rotation, _ = Rot.align_vectors([direction_vector], [z_axis])
            quat = rotation.as_quat()  # Returns [rx, ry, rz, rw]

            # Create the 7D pose [x, y, z, rw, rx, ry, rz]
            pose = np.concatenate([[x, y, z], [quat[3], quat[0], quat[1], quat[2]]])  # [x, y, z, rw, rx, ry, rz]
            trajectory_poses.append(pose)

        # === Segment 3: Move across the object in the y direction (arc motion) ===
        for i in range(num_points_per_segment):
            fraction = i / num_points_per_segment
            x = object_position[0]  # Fixed x position
            y = object_position[1] - (fraction - 0.5) * xy_variation  # Moving from front to back (across the object)
            z = object_position[2] + height_variation * np.sin(fraction * np.pi) + height_offset  # Arc motion in z, going under, over, and back under the object

            # Compute the direction vector from the current position to the object
            direction_vector = np.array(object_position) - np.array([x, y, z])
            direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector
            
            # Compute quaternion to keep pointing at the object
            z_axis = np.array([0, 0, 1])  # Assuming the robot's z-axis is the forward direction
            rotation, _ = Rot.align_vectors([direction_vector], [z_axis])
            quat = rotation.as_quat()  # Returns [rx, ry, rz, rw]

            # Create the 7D pose [x, y, z, rw, rx, ry, rz]
            pose = np.concatenate([[x, y, z], [quat[3], quat[0], quat[1], quat[2]]])  # [x, y, z, rw, rx, ry, rz]
            trajectory_poses.append(pose)

        return trajectory_poses

    def plot_3d_trajectory(self, trajectory_poses, object_position):
        '''
        Plots the 3D trajectory of the end-effector around the object, showing position.

        :param trajectory_poses: List of poses in 7D format [x, y, z, rw, rx, ry, rz].
        :param object_position: 3D list [x, y, z] representing the object position.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract x, y, z positions from the trajectory poses
        x = [pose[0] for pose in trajectory_poses]
        y = [pose[1] for pose in trajectory_poses]
        z = [pose[2] for pose in trajectory_poses]

        # Plot the trajectory
        ax.plot(x, y, z, label="End-Effector Trajectory", color='b', marker='o')
        
        # Plot the object position
        ax.scatter(*object_position, color='r', label="Object Position", s=100)

        # Labels and legend
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Trajectory of the End-Effector')
        ax.legend()
        
        plt.show()


    def add_point_cloud(self, xyzrgb):
        self.point_clouds.append(xyzrgb)

if __name__ == "__main__":
    cam = RealSenseCamera()
    camIntrinsics = cam.getIntrinsics()
    camera_pose = np.load('config/camera_calib.npy')
    flexiv_robot = FlexivRobot()
    vision_slam = VisionSLAM()

    flexiv_robot.move_to_home()
    flexiv_robot.set_zero_ft()
    home_pose = flexiv_robot.get_tcp_pose(euler=False)
    object_position = home_pose[:3]

    trajectory_poses = vision_slam.generate_3d_scan_trajectory(object_position)
    vision_slam.plot_3d_trajectory(trajectory_poses, object_position)

    for pose in trajectory_poses[:3]:
        flexiv_robot.cartesian_motion_force_control(pose)
        robot_pose = flexiv_robot.get_tcp_pose(matrix = True)
        time.sleep(1)
        color, depth = cam.get_data(hole_filling=False)
        xyzrgb = cam.getXYZRGB(color, depth, camIntrinsics, robot_pose, camera_pose, max_depth=0.5, inpaint=True)
        vision_slam.add_point_cloud(xyzrgb)


    np.savez('xyzrgb.npz', *vision_slam.point_clouds)

