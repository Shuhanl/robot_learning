import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        self.pipeline.start(self.config)
        
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
    
    def release(self):
        # Stop the pipeline
        self.pipeline.stop()

class AprilTag:
    def __init__(self):
        self.options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(self.options)

        self.camera_params = [640, 480, 320, 240]  # fx, fy, cx, cy
        self.tag_size = 0.05  # AprilTag side length in meters

    def detect_apriltags(self, color_image):
        # Convert to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        tags = self.detector.detect(gray_image)
        for tag in tags:
            # Extract the bounding box and display it
            (ptA, ptB, ptC, ptD) = tag.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)

            # Calculate the pose of the tag
            pose, e0, e1 = self.detector.detection_pose(tag, self.camera_params, self.tag_size)

            # Decompose the rotation matrix to Euler angles
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            tvec = pose[:3, 3]

            # Display the translation vector (position) and rotation vector (orientation)
            cv2.putText(color_image, f'Position: {tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(color_image, f'Orientation: {rvec[0,0]:.2f}, {rvec[1,0]:.2f}, {rvec[2,0]:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Put the tag ID on the image
            tag_id = "ID: {}".format(tag.tag_id)
            cv2.putText(color_image, tag_id, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return tags

def main():
    camera = RealSenseCamera()
    april_tag = AprilTag()
    try:
        while True:
            color_image, depth_colormap = camera.capture_frames()
            if color_image is None or depth_colormap is None:
                continue

            # Tag detection
            tags = april_tag.detect_apriltags(color_image)

            cv2.imshow('RGB with Markers', color_image)
            cv2.imshow('Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
