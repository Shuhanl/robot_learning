from camera import RealSenseCamera
import numpy as np
import pdb
import open3d as o3d

class SLAM(object):
    def __init__(self):
        self.point_cloud = []
        self.processed_point_cloud = []

    def icp_pointcloud(self, source_pcd, target_pcd):
        '''
        transform target to source frame
        :param source: numpy.ndarray
        :param target: numpy.ndarray
        :return: transformed source numpy.ndarray
        '''

        # Compute the ICP transformation
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance=0.02,
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        # Apply the transformation to the source point cloud
        source_pcd.transform(icp_result.transformation)
        aligned_source = np.asarray(source_pcd.points)

        return aligned_source

    def cal_norm(self, point_cloud):
                
        # Estimate normals using Open3D. The radius or the number of neighbors used to estimate the normals can be set here.
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Optionally, reorient the normals so that they all point in a consistent direction
        # This can be useful if the normals are oriented inconsistently across the cloud
        point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
        
        # Flip normals that point below the horizontal plane (assumed)
        normals = np.asarray(point_cloud.normals)
        if normals[2] < 0:
            normals = -normals
                
        return normals

    def process_point_cloud(self):
       
        point_cloud = o3d.geometry.PointCloud()
        if self.point_cloud.shape[-1] == 6:
            point_cloud.points = o3d.utility.Vector3dVector(self.point_cloud[:, :3])
        elif self.point_cloud.shape[-1] == 3:
            point_cloud.points = o3d.utility.Vector3dVector(self.point_cloud)
        else:
            pdb.set_trace()     

        for i in range(1, len(point_cloud.points)):
            aligned_pc = self.icp_pointcloud(point_cloud[i], point_cloud[i-1])
            normals = self.cal_norm(aligned_pc)
            if self.point_cloud.shape[-1] == 6:
                # Combine XYZ rgb and normals into a single array
                self.process_pointcloud = self.process_pointcloud.append(np.hstack((aligned_pc, self.point_cloud[i, 3:], normals)))
            elif self.point_cloud.shape[-1] == 3:
                # Combine XYZ normals into a single array
                self.process_pointcloud = np.hstack((aligned_pc, normals))
            else:
                pdb.set_trace()  
    
    def vis_pc(self, pc):
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pc1])

if __name__ == "__main__":
    cam = RealSenseCamera()
    slam = SLAM()

    while True:
        color, depth = cam.get_data(hole_filling=False)
        camIntrinsics = cam.getIntrinsics()
        xyzrgb = slam.getXYZRGB(color, depth, camIntrinsics, np.identity(4), np.identity(4), inpaint=False)
        slam.point_cloud.append(xyzrgb)

        if len(slam.point_cloud) % 10 == 0:  # Periodically process or visualize
            slam.process_point_cloud()
            slam.vis_pc(slam.processed_point_cloud[:,:3])

    # while True:
    #     color, depth = cam.get_data(hole_filling=False)
  
    #     # pdb.set_trace()
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03),cv2.COLORMAP_JET)
    #     cv2.imshow('depth', depth_colormap)
    #     cv2.imshow('color', color)
    #     cv2.waitKey(1)
