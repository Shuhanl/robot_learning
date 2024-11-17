import pyrealsense2 as rs
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt

class RealSenseCamera(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the streams
        self.config.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30)
        self.config.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
        self.hole_filling = rs.hole_filling_filter()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()

        print('cam init ...')
        i = 60
        while i>0:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            i -= 1
        print('cam init done.')

    def getIntrinsics(self):
        """ Read camera intrinsics from the camera """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        mtx = [intrinsics.width,intrinsics.height,intrinsics.ppx,intrinsics.ppy,intrinsics.fx,intrinsics.fy]
        camIntrinsics = np.array([[mtx[4],0,mtx[2]],
                                  [0,mtx[5],mtx[3]],
                                 [0,0,1.]])
        return camIntrinsics

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
    
    def capture_frames(self):
        # This function captures frames from both color and depth streams
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply color map to depth image to visualize it
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        return color_image, depth_colormap

    def get_data(self, hole_filling=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image

    def getXYZRGB(self, color, depth, camIntrinsics, robot_pose, camera_pose, max_depth=0.5, inpaint=True):
        '''
        Converts color and depth images into a 3D point cloud (XYZRGB), filtering points based on depth.

        :param color: Color image.
        :param depth: Depth image.
        :param robot_pose: Robot's 4x4 transformation matrix.
        :param camera_pose: 4x4 transformation matrix for the camera.
        :param camIntrinsics: Intrinsic parameters of the camera.
        :param max_depth: Maximum allowable depth (in meters) for filtering.
        :param inpaint: Boolean to indicate whether to inpaint the depth image.
        :return: A numpy array containing filtered 3D points (XYZ) and color information (RGB).
        '''
        heightIMG, widthIMG, _ = color.shape
        depthImg = depth / 1000.0  # Convert depth from millimeters to meters

        if inpaint:
            depthImg = self.inpaint(depthImg)

        # Combine robot pose and camera pose to get the full transformation
        full_pose = np.dot(robot_pose, camera_pose)

        # Generate pixel grid
        [pixX, pixY] = np.meshgrid(np.arange(widthIMG), np.arange(heightIMG))

        # Convert pixel coordinates to camera coordinates
        camX = (pixX - camIntrinsics[0][2]) * depthImg / camIntrinsics[0][0]
        camY = (pixY - camIntrinsics[1][2]) * depthImg / camIntrinsics[1][1]
        camZ = depthImg

        # Stack camX, camY, and camZ into a 3D point cloud
        camPts = np.stack((camX, camY, camZ), axis=-1).reshape(-1, 3)

        # Apply depth filtering by removing points with a z-coordinate greater than max_depth
        valid_idx = camPts[:, 2] <= max_depth  # Check if the depth (z) is less than or equal to max_depth
        camPts = camPts[valid_idx]  # Keep only valid points
        rgb = color.reshape((-1, 3))[valid_idx] / 255.0  # Keep corresponding RGB values for valid points

        # Transform points from camera coordinates to world coordinates
        worldPts = (full_pose[:3, :3] @ camPts.T + full_pose[:3, 3].reshape(3, 1)).T

        # Combine XYZ and RGB
        xyzrgb = np.hstack((worldPts, rgb))

        return xyzrgb
      
    
    def release(self):
        # Stop the pipeline
        self.pipeline.stop()


if __name__ == "__main__":
    cam = RealSenseCamera()

    color, depth = cam.get_data(hole_filling=True)
    plt.imshow(depth)
    plt.show()
    plt.imshow(color)
    plt.show()


