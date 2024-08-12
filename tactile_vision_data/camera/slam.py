from camera import RealSenseCamera
import numpy as np
import cv2
import pdb
import open3d as o3d

class SLAM(object):
    def __init__(self):
        print("Init slam")

    def inpaint(self, img, missing_value=0):
        '''
        pip opencv-python == 3.4.8.29
        :param image:
        :param roi: [x0,y0,x1,y1]
        :param missing_value:
        :return:
        '''
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(img).max()
        if scale < 1e-3:
            pdb.set_trace()
        img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        img = img[1:-1, 1:-1]
        img = img * scale
        return img

    def getleft(self, obj1):
        index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
        index = np.bitwise_and(obj1[:, 1] < 0.5, index)
        index = np.bitwise_and(obj1[:, 1] > -0.5, index)
        # index = np.bitwise_and(obj1[:, 2] > -0.1, index)
        index = np.bitwise_and(obj1[:, 2] > 0.24, index)
        index = np.bitwise_and(obj1[:, 2] < 0.6, index)
        return obj1[index]
    
    def getXYZRGB(self,color, depth, robot_pose,camee_pose,camIntrinsics,inpaint=True):
        '''

        :param color:
        :param depth:
        :param robot_pose: array 4*4
        :param camee_pose: array 4*4
        :param camIntrinsics: array 3*3
        :param inpaint: bool
        :return: xyzrgb
        '''
        heightIMG, widthIMG, _ = color.shape
        # heightIMG = 720
        # widthIMG = 1280
        depthImg = depth / 1000.
        # depthImg = depth
        if inpaint:
            depthImg = self.inpaint(depthImg)
        robot_pose = np.dot(robot_pose, camee_pose)

        [pixX, pixY] = np.meshgrid(np.arange(widthIMG), np.arange(heightIMG))
        camX = (pixX - camIntrinsics[0][2]) * depthImg / camIntrinsics[0][0]
        camY = (pixY - camIntrinsics[1][2]) * depthImg / camIntrinsics[1][1]
        camZ = depthImg

        camPts = [camX.reshape(camX.shape + (1,)), camY.reshape(camY.shape + (1,)), camZ.reshape(camZ.shape + (1,))]
        camPts = np.concatenate(camPts, 2)
        camPts = camPts.reshape((camPts.shape[0] * camPts.shape[1], camPts.shape[2]))  # shape = (heightIMG*widthIMG, 3)
        worldPts = np.dot(robot_pose[:3, :3], camPts.transpose()) + robot_pose[:3, 3].reshape(3,
                                                                                              1)  # shape = (3, heightIMG*widthIMG)
        rgb = color.reshape((-1, 3)) / 255.
        xyzrgb = np.hstack((worldPts.T, rgb))
        xyzrgb = self.getleft(xyzrgb)
        return xyzrgb

    def vis_pc(self, pc):
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pc1])

    def __del__(self):
        self.pipeline.stop()


if __name__ == "__main__":
    cam = RealSenseCamera()
    slam = SLAM()

    # pdb.set_trace()
    color, depth = cam.get_data(hole_filling=False)
    xyzrgb = slam.getXYZRGB(color, depth, np.identity(4), np.identity(4), cam.getIntrinsics(), inpaint=False)
    slam.vis_pc(xyzrgb[:,:3])

    while True:
        color, depth = cam.get_data(hole_filling=False)
  
        # pdb.set_trace()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03),cv2.COLORMAP_JET)
        cv2.imshow('depth', depth_colormap)
        cv2.imshow('color', color)
        cv2.waitKey(1)
