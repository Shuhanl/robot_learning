import pyrealsense2 as rs
import numpy as np
import cv2
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

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        mtx = [intrinsics.width,intrinsics.height,intrinsics.ppx,intrinsics.ppy,intrinsics.fx,intrinsics.fy]
        camIntrinsics = np.array([[mtx[4],0,mtx[2]],
                                  [0,mtx[5],mtx[3]],
                                 [0,0,1.]])
        return camIntrinsics
    
    
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

    depth_sensor = cam.pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale is: ", depth_scale)

