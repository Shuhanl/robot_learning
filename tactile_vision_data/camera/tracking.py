from camera import RealSenseCamera
import cv2
import apriltag
import numpy as np

class Tracking:
    def __init__(self):
        self.camera_params = [640, 480, 320, 240]  # fx, fy, cx, cy
        """ April tag """
        self.family="tag36h11"
        self.tag_size = 0.017
        self.options = apriltag.DetectorOptions(self.family)
        self.detector = apriltag.Detector(self.options)


    def rotation_matrix_to_quaternion(self, R):
        """ Convert a rotation matrix to a quaternion """
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
        return np.array([q0, q1, q2, q3])

    def average_quaternions(self, quaternions):
        """ Average quaternions using a normalized weighted sum """
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)  # Normalize to unit quaternion
        return avg_quat

    def compute_average_pose(self, poses):
        translations = [p[0] for p in poses]
        quaternions = [p[1] for p in poses]
        avg_tvec = np.mean(translations, axis=0)
        avg_quat = self.average_quaternions(quaternions)
        return avg_tvec, avg_quat

    def detect_apriltags(self, color_image):
        # Convert to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        tags = self.detector.detect(gray_image)
        poses = []
        quaternions = []

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

            quat = self.rotation_matrix_to_quaternion(pose[:3, :3])
            quaternions.append(quat)
            poses.append((tvec, quat))

            # Display the translation vector (position) and rotation vector (orientation)
            tag_center = (int(tag.center[0]), int(tag.center[1]))
            cv2.putText(color_image, f'Pos: {tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}', (tag_center[0] + 20, tag_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(color_image, f'Orient: {rvec[0,0]:.2f}, {rvec[1,0]:.2f}, {rvec[2,0]:.2f}', (tag_center[0] + 20, tag_center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Put the tag ID on the image
            tag_id = "ID: {}".format(tag.tag_id)
            cv2.putText(color_image, tag_id, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return tags, poses 
    

if __name__ == '__main__':
    camera = RealSenseCamera()
    tracking = Tracking()

    while True:
        color_image, depth_colormap = camera.capture_frames()
        if color_image is None or depth_colormap is None:
            continue

        # Tag detection
        tags, poses = tracking.detect_apriltags(color_image)

        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('RGB with Markers', color_image)

        if poses:
            avg_tvec, avg_quat = tracking.compute_average_pose(poses)
            print("Average Position of Cube:", avg_tvec)
            print("Average Orientation:", avg_quat)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



