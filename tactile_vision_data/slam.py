from camera.camera import RealSenseCamera
# from robot.flexiv import FlexivRobot
import numpy as np

class SLAM(object):
    def __init__(self):
        self.point_clouds = []

    def add_point_cloud(self, xyzrgb):
        self.point_clouds.append(xyzrgb)

if __name__ == "__main__":
    cam = RealSenseCamera()
    camIntrinsics = cam.getIntrinsics()
    # flexiv_robot = FlexivRobot()
    slam = SLAM()

    jointList = [[0.41219, -0.807, -0.922, -1.858, 0.168, 1.870, 0.0011],
                 [-0.314, -1.01202, 0.180, -2.315, 0.401, 2.122, 0.589]]


    for i, joint in enumerate(jointList):
        # robot_pose = flexiv_robot.get_tcp_pose()
        color, depth = cam.get_data(hole_filling=False)
        xyzrgb = cam.getXYZRGB(color, depth, camIntrinsics, np.identity(4), np.identity(4), inpaint=False)
        slam.add_point_cloud(xyzrgb)

    np.save('xyzrgb.npy',slam.point_clouds)

