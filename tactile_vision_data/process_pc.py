import numpy as np
import open3d as o3d

class ProcessPointCloud(object):
    def __init__(self):
        print("init")

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

    def process(self, xyzrgbs):

        """ Converts numpy array to Open3D point cloud and stores it. """
        point_clouds = []
        for xyzrgb in xyzrgbs:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])  # Assumes xyz are the first three columns
            point_cloud.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:6])  # Assumes rgb are the next three columns
            point_clouds.append(point_cloud)

        for i in range(1, len(point_clouds)):
            point_clouds[i].points = self.icp_pointcloud(point_clouds[i], point_clouds[i-1])
            point_clouds[i].normals = self.cal_norm(point_clouds[i])
 
        # Combine all processed point clouds into one
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            combined_pcd += pcd

        return combined_pcd

    def vis_pc(self, pc):
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pc1])

if __name__ == "__main__":

    process_point_cloud = ProcessPointCloud()
    xyzrgbs = np.load("xyzrgb.npy")
    combined_pcd = process_point_cloud.process(xyzrgbs)
    process_point_cloud.vis_pc(combined_pcd.points)


