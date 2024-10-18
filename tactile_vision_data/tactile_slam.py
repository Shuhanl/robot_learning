from tactile import PyTac3D
from robot.flexiv import FlexivRobot
import numpy as np
import time

def compute_world_coordinates(tactile_positions, ee_pose, T_sensor_to_ee):
    """
    Compute the world coordinates (x, y, z) of each point in the tactile sensor grid.
    
    Parameters:
    - tactile_positions: 400x3 array of tactile sensor grid positions in the local sensor frame.
    - ee_pose: 4x4 transformation matrix representing the robot's end-effector pose in the world frame.
    - T_sensor_to_ee: 4x4 transformation matrix from the sensor frame to the end-effector frame.
    
    Returns:
    - point_cloud: 400x3 array of points transformed into world coordinates.
    """
    # Compute the full transformation from the tactile sensor frame to the world frame
    T_sensor_to_world = ee_pose @ T_sensor_to_ee
    
    # Convert tactile_positions (400x3) to homogeneous coordinates (400x4)
    num_points = tactile_positions.shape[0]
    ones = np.ones((num_points, 1))  # Create a column of ones for homogeneous coordinates
    tactile_points_hom = np.hstack((tactile_positions, ones))  # Shape: (400, 4)

    # Apply the transformation matrix to each point (sensor frame to world frame)
    world_points_hom = (T_sensor_to_world @ tactile_points_hom.T).T  # Shape: (400, 4)

    # Extract the x, y, z coordinates from the resulting homogeneous coordinates
    point_cloud = world_points_hom[:, :3]

    return point_cloud

if __name__ == "__main__":

    tac3d = PyTac3D.Sensor(port=9988, maxQSize=5)
    flexiv_robot = FlexivRobot()

    # 等待Tac3D-Desktop端启动传感器并建立连接
    tac3d.waitForFrame()
    time.sleep(5) 

    # 发送一次校准信号（应确保校准时传感器未与任何物体接触！否则会输出错误的数据！）
    SN = tac3d.getFrame()['SN']
    tac3d.calibrate(SN)
    time.sleep(5) 

    coord_data = []
    friction_data = []
    stiffness_data = []

    # Transformation matrix from the tactile sensor frame to the robot end-effector frame
    T_sensor_to_ee = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0],  
        [0, 0, 1, 0],  
        [0, 0, 0, 1]   
    ])

    for pose in range(10):
        P = tac3d.getFrame()['3D_Positions']
        D = tac3d.getFrame()['3D_Displacements']
        F = tac3d.getFrame()['3D_Forces']
        # Compute friction and stiffness for the 20x20 tactile sensor grid
        friction = np.zeros((20, 20))
        stiffness = np.zeros((20, 20))

        ee_pose = flexiv_robot.get_tcp_pose(matrix=True)  # 4x4 transformation matrix

        for i in range(20):
            for j in range(20):
                idx = i * 20 + j  # Convert 2D index to 1D (400 points total)

                Fx, Fy, Fz = F[idx]
                _, _, z_disp = D[idx]
                
                if Fz != 0:
                    friction[i, j] = np.sqrt(Fx**2 + Fy**2) / Fz
                    stiffness[i, j] = Fz / z_disp if z_disp != 0 else 0
                else:
                    friction[i, j] = 0
                    stiffness[i, j] = 0

        # Compute the point cloud for this frame using the combined transformation
        coord = compute_world_coordinates(P, ee_pose, T_sensor_to_ee)

        # Append friction and stiffness for the current frame
        coord_data.append(coord)
        friction_data.append(friction)
        stiffness_data.append(stiffness)

        # Convert lists to numpy arrays
        coord_data = np.array(coord_data)  
        friction_data = np.array(friction_data)
        stiffness_data = np.array(stiffness_data)

        time.sleep(2)

    np.savez("tactile.npz", coords=coord_data, friction=friction_data, stiffness=stiffness_data)