from camera import RealSenseCamera
import cv2
import apriltag

class Tracking:
    def __init__(self):
        self.camera_params = [640, 480, 320, 240]  # fx, fy, cx, cy

        """ April tag """
        self.family="tag36h11"
        self.options = apriltag.DetectorOptions(self.family)
        self.detector = apriltag.Detector(self.options)

    
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
    

if __name__ == '__main__':
    camera = RealSenseCamera()
    tracking = Tracking()

    while True:
        color_image, depth_colormap = camera.capture_frames()
        if color_image is None or depth_colormap is None:
            continue

        # Tag detection
        tags = tracking.detect_apriltags(color_image)

        cv2.imshow('RGB with Markers', color_image)
        cv2.imshow('Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



