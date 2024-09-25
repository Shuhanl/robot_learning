from camera.camera import RealSenseCamera
from robot.flexiv import FlexivRobot
import numpy as np
import time

class SLAM(object):
    def __init__(self):
        self.point_clouds = []

    def add_point_cloud(self, xyzrgb):
        self.point_clouds.append(xyzrgb)

if __name__ == "__main__":
    cam = RealSenseCamera()
    camIntrinsics = cam.getIntrinsics()
    flexiv_robot = FlexivRobot()
    slam = SLAM()

    flexiv_robot.move_to_home()
    flexiv_robot.set_zero_ft()
    home_pose = flexiv_robot.get_tcp_pose(euler=True)
    x, y, z, roll, pitch, yaw = home_pose

    position_swing = 0.05
    rotation_swing = 0.2
    
    calib_poses = [
        [x, y, z, roll, pitch, yaw],  # Home pose (base)
        [x + position_swing, y, z, roll, pitch - rotation_swing, yaw],  
        [x - position_swing, y, z, roll, pitch + rotation_swing, yaw], 
        [x, y + position_swing, z, roll + rotation_swing, pitch, yaw],  
        [x, y - position_swing, z, roll - rotation_swing, pitch, yaw],  
    ]

    for pose in calib_poses:
        flexiv_robot.cartesian_motion_force_control(pose, is_euler=True)
        color, depth = cam.get_data(hole_filling=False)
        xyzrgb = cam.getXYZRGB(color, depth, camIntrinsics, np.identity(4), np.identity(4), inpaint=False)
        slam.add_point_cloud(xyzrgb)
        time.sleep(1)

    np.save('xyzrgb.npy',slam.point_clouds)

