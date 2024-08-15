from camera import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Calibration(object):
    def __init__(self,pattern_size=(9,6),square_size=21,handeye='EIH'):
        '''

        :param image_list:  image array, num*720*1280*3
        :param pose_list: pose array, num*4*4
        :param pattern_size: calibration pattern size
        :param square_size: calibration pattern square size, 15mm
        :param handeye:
        '''
        self.pattern_size = pattern_size  # The number of inner corners of the chessboard in width and height.
        self.square_size = square_size
        self.handeye = handeye
        # self.mtx = np.load('config/franka_d415mtx.npy')
        self.init_calib()

    def init_calib(self):
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = self.square_size * np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        for i in range(self.pattern_size[0] * self.pattern_size[1]):
            x, y = self.objp[i, 0], self.objp[i, 1]
            self.objp[i, 0], self.objp[i, 1] = y, x
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detectFeature(self,color,show=True):
        img = color
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(img, self.pattern_size, None,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH)  # + cv2.CALIB_CB_NORMALIZE_IMAGE+ cv2.CALIB_CB_FAST_CHECK)
        if ret == True:
            self.objpoints.append(self.objp)
            # corners2 = corners
            if (cv2.__version__).split('.')[0] == '2':
                # pdb.set_trace()
                cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
                corners2 = corners
            else:
                corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
            self.imgpoints.append(corners2)
            if show:
                fig, ax = plt.subplots(figsize=(20, 20))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                plt.title('img with feature point')
                for i in range(self.pattern_size[0] * self.pattern_size[1] - 3):
                    ax.plot(corners2[i, 0, 0], corners2[i, 0, 1], 'r+')
                plt.show()

    """ Convert a rotation vector to a 4x4 transformation matrix """
    def rodrigues_trans2tr(self, rvec, tvec):
        r, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to a rotation matrix
        tvec.shape = (3,)  # Ensure that the translation vector has the correct shape (flattened to a 1D array)
        T = np.identity(4)  # Create a 4x4 identity matrix
        T[0:3, 3] = tvec  # Place the translation vector in the last column of the upper 3 rows
        T[0:3, 0:3] = r  # Place the rotation matrix in the top-left 3x3 submatrix
        return T  # Return the full 4x4 transformation matrix

    def rodrigues_trans2tr(self, rvec, tvec):
        '''
        Converts a rotation vector and a translation vector to a 4x4 transformation matrix.
        '''
        r, _ = cv2.Rodrigues(rvec)
        tvec.shape = (3,)
        T = np.identity(4)
        T[0:3, 3] = tvec
        T[0:3, 0:3] = r
        return T
    
    def perform_camera_calibration(self):
        if self.imgpoints:
            gray_shape = cv2.cvtColor(self.imgpoints[0], cv2.COLOR_BGR2GRAY).shape[::-1]
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray_shape, None, None)
            return self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs
        else:
            return None, None, None, None

    def perform_hand_eye_calibration(self, robot_poses, rvecs, tvecs):
        '''
        Performs hand-eye calibration given robot poses and camera transforms.

        :param robot_poses: Array of robot poses.
        :param camera_transforms: Array of camera transformations corresponding to robot poses.
        '''
        camera_transforms = [self.rodrigues_trans2tr(rvec, tvec / 1000.) for rvec, tvec in zip(rvecs, tvecs)]
        rot, pos = cv2.calibrateHandEye(robot_poses[:, :3, :3], robot_poses[:, :3, 3], camera_transforms[:, :3, :3], camera_transforms[:, :3, 3])
        camT = np.identity(4)
        camT[:3, :3] = rot
        camT[:3, 3] = pos[:,0]
        return camT
        

if __name__ == "__main__":
    cam = RealSenseCamera()
    calib = Calibration(pattern_size=(8,6), square_size=15)
    
    """ Capture single picture for calibration """
    color, depth = cam.get_data()
    plt.imshow(color)
    plt.show()
    # calib.detectFeature(color)
    # mtx, dist, rvecs, tvecs = calib.perform_camera_calibration()
    # print(mtx, dist, rvecs, tvecs)

    """ Capture sequences of pictures with pose info for calibration """
    # pose_list = []
    # Assuming cam and panda are predefined objects
    # for joint in jointList:
    #     panda.moveJoint(joint)
    #     current_pose = np.array(panda.robot.read_once().O_T_EE).reshape(4, 4).T
    #     color, depth = cam.get_data()
    #     cam.detect_feature(color)
    #     pose_list.append(current_pose)


    # mtx, dist, rvecs, tvecs = calib.perform_camera_calibration()
    # camT = calib.perform_hand_eye_calibration(pose_list, rvecs, tvecs)
    # np.save('config/campose20210909_franka.npy', camT)