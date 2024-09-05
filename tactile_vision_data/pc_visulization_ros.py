import pcl
import rospy
from sensor_msgs.msg import PointCloud2,PointField

import numpy as np
import cv2
import os.path as osp

from test_calib import inpaint

import pdb

def getleft(obj1):
    index = np.bitwise_and(obj1[:, 0] < 1.2, obj1[:, 0] > 0.2)
    index = np.bitwise_and(obj1[:, 1] < 0.5, index)
    index = np.bitwise_and(obj1[:, 2] > -0.1,index)
    index = np.bitwise_and(obj1[:, 1] > -0.5, index)
    index = np.bitwise_and(obj1[:, 2] < 0.6, index)
    return obj1[index]

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg

def icp_pointcloud(pc_list):
    # import pcl
    assert len(pc_list)==3, "Only length of 3 is done."

    if pc_list[0].shape[-1] == 6:
        pc0, pc1, pc2 = pc_list[0][:, :3], pc_list[1][:, :3], pc_list[2][:, :3]
    elif pc_list[0].shape[-1] == 3:
        pc0, pc1, pc2 = pc_list[0], pc_list[1], pc_list[2]
    else:
        pdb.set_trace()

    def icp_pcl(source, target):
        '''
        transform target to source frame
        :param source:
        :param target:
        :return:
        '''
        source = pcl.PointCloud(source.astype(np.float32))
        target = pcl.PointCloud(target.astype(np.float32))
        icp = source.make_IterativeClosestPoint()
        # pdb.set_trace()
        converged, transf, estimate, fitness = icp.icp(source, target)
        # transf = np.linalg.inv(transf)
        source_t = np.dot(transf[:3,:3], source.to_array().T) + transf[:3,3].reshape((3,1))
        source_t = source_t.T
        return source_t

    pc0 = icp_pcl(pc0,pc1)
    pc2 = icp_pcl(pc2,pc1)
    if pc_list[0].shape[-1] == 6:
        pc0 = np.hstack([pc0,pc_list[0][:,3:]])
        pc1 = np.hstack([pc1,pc_list[1][:,3:]])
        pc2 = np.hstack([pc2,pc_list[2][:,3:]])
    return [pc0,pc1,pc2]

def cal_norm(xyzrgb):
    xyz = xyzrgb[:,:3]
    cloud = xyz.reshape(-1, 3).astype(np.float32)
    p = pcl.PointCloud(cloud)
    norm = p.make_NormalEstimation()
    norm.set_KSearch(10)
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]
    index = np.where(surface_normal[:,-1]<0)
    surface_normal[index] = -surface_normal[index]
    # pdb.set_trace()
    xyzrgbnorm = np.hstack((xyzrgb,surface_normal))
    return xyzrgbnorm


if __name__ == "__main__":
    rospy.init_node('Camera_pubROS_PointCloud2', anonymous=True)
    pub_fusion = rospy.Publisher('pc_fusion', PointCloud2, queue_size=1)
    stamp = rospy.Time.now()
    frame_id = '/base'

    NUM = 3
    root_dir = '/home/hai/PycharmProjects/calibration/images/20210901/weibolu10'
    ee_pose = np.load('ee0901_8.npy')
    # obj_robot = np.load(osp.join(root_dir, 'pose_list.npy'))
    obj_robot = np.load('/home/hai/PycharmProjects/calibration/images/20210901/pose_list_weibolu10.npy')
    INPAINT = True
    INPAINT = False

    camIntrinsics = np.load('franka_mtx.npy')

    obj_list = []
    pointcloud2_list = []
    pub_list = []
    # pub_1 = rospy.Publisher('pc_1', PointCloud2, queue_size=1)
    # pub_2 = rospy.Publisher('pc_2', PointCloud2, queue_size=1)
    # pub_3 = rospy.Publisher('pc_3', PointCloud2, queue_size=1)
    # pub_4 = rospy.Publisher('pc_4', PointCloud2, queue_size=1)
    # pub_5 = rospy.Publisher('pc_5', PointCloud2, queue_size=1)
    for i in range(NUM):
        pub_list.append(rospy.Publisher("pc_%d"%(i+1),PointCloud2,queue_size=1))

        color_path = osp.join(root_dir,'color', 'franka_%0.3d.png' % (i + 1))
        depth_path = osp.join(root_dir,'depth', 'franka_%0.3d.png' % (i + 1))

        color = cv2.imread(color_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        heightIMG = 720
        widthIMG = 1280
        depthImg = depth / 1000.
        if INPAINT:
            depthImg = inpaint(depthImg)

        robot_pose = obj_robot[i]
        robot_pose = np.dot(robot_pose, ee_pose)

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
        xyzrgb = getleft(xyzrgb)

        xyzrgb = cal_norm(xyzrgb)
        pointcloud2_list.append(xyzrgb_array_to_pointcloud2(xyzrgb[:, :3], xyzrgb[:, 3:6], stamp=stamp,frame_id=frame_id))
        obj_list.append(xyzrgb)
        np.save(osp.join(root_dir, 'obj%d.npy' % (i + 1)), xyzrgb)

        # pdb.set_trace()
    # obj_list = icp_pointcloud(obj_list)
    xyzrgb_fusion = np.vstack(obj_list)
    index = np.random.choice(np.arange(xyzrgb_fusion.shape[0]), size=int(xyzrgb_fusion.shape[0] * 0.8), replace=False)
    xyzrgb_fusion = xyzrgb_fusion[index]
    np.save(osp.join(root_dir, 'obj_fusion.npy'), xyzrgb_fusion)

    # pdb.set_trace()
    pointcloud2 = xyzrgb_array_to_pointcloud2(xyzrgb_fusion[:,:3], xyzrgb_fusion[:,3:6], stamp=stamp, frame_id=frame_id)
    print("Start showing in RVIZ")
    while True:
        pub_fusion.publish(pointcloud2)
        for i in range(NUM):
            pub_list[i].publish(pointcloud2_list[i])
    pdb.set_trace()