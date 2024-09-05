import pdb
import paramiko
from robot import Panda
from camera import CameraL515,Camera
import cv2
import numpy as np
import os
import os.path as osp
import open3d as o3d

def scp_file(pc, file_name="obj.npy"):
    transport = paramiko.Transport(('10.52.21.133', 22))
    transport.connect(username='lvjun', password='lvjun')
    sftp = paramiko.SFTPClient.from_transport(transport)
    np.save("tmp.npy", pc)
    sftp.put("tmp.npy", '/home/lvjun/robot/%s'%file_name)
    transport.close()

def vis_pc(pc):
    pc1 = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0,0,0])
    pc1.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([frame, pc1])

if __name__ == "__main__":
    camMtx = np.load('config/franka_d415mtx.npy')
    camPose = np.load('config/campose20210909_franka.npy')
    cam = Camera()
    # while True:
    #     color, depth = cam.get_data()
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    #     cv2.imshow('color',color)
    #     cv2.imshow('depth',depth_colormap)
    #     cv2.waitKey(1)
    panda = Panda()
    jointList = [[0.41219, -0.807, -0.922, -1.858, 0.168, 1.870, 0.0011],
                 [-0.314, -1.01202, 0.180, -2.315, 0.401, 2.122, 0.589],
                 [-1.364, -1.540, 0.583, -1.33, 1.012, 1.35, 1.38]]
    xyzrgbList = []
    obj_image_root = 'image/20210908/weibolu2'
    pose_list = []
    for i, joint in enumerate(jointList):
        panda.moveJoint(joint)
        color, depth = cam.get_data()
        # pdb.set_trace()
        current_pose = np.array(panda.robot.read_once().O_T_EE).reshape(4, 4).T

        os.makedirs(obj_image_root, exist_ok=True)
        os.makedirs(osp.join(obj_image_root, 'color'), exist_ok=True)
        os.makedirs(osp.join(obj_image_root, 'depth'), exist_ok=True)
        cv2.imwrite(osp.join(obj_image_root, 'depth', 'franka_%0.3d.png' % (i + 1)), depth)
        cv2.imwrite(osp.join(obj_image_root, 'color', 'franka_%0.3d.png' % (i + 1)), color)
        pose_list.append(current_pose)
        # pdb.set_trace()

        xyzrgb = cam.getXYZRGB(color, depth, current_pose, camPose, camMtx, False)
        xyzrgb_none = cam.getXYZRGB(color, depth, np.identity(4), np.identity(4), camMtx, False)
        xyzrgb_ttt = cam.getleft(xyzrgb)
        vis_pc(xyzrgb[:,:3])
        np.savetxt('xyz.xyz',xyzrgb[:,:3])
        np.save(osp.join(obj_image_root, 'obj_%d.npy'%(i+1)), np.array(xyzrgb))
        xyzrgbList.append(xyzrgb)
        #pdb.set_trace()
    xyzrgbfusion = np.vstack(xyzrgbList)
    np.save(osp.join(obj_image_root, 'pose_list.npy'), np.array(pose_list))
    #vis_pc(xyzrgbfusion[:,:3])
    vis_pc(xyzrgbList[2][:,:3])
    pdb.set_trace()
    scp_file(xyzrgbfusion)
    scp_file(xyzrgbfusion)
    scp_file(xyzrgbfusion)
    scp_file(xyzrgbfusion)