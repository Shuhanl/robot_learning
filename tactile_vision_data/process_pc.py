import numpy as np
import open3d as o3d
import copy
class ProcessVisionPointCloud(object):
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

        # Create a copy of the source point cloud
        source_pcd_transformed = copy.deepcopy(source_pcd)

        # Apply the transformation to the copied point cloud
        source_pcd_transformed.transform(icp_result.transformation)

        return source_pcd_transformed

    def cal_norm(self, point_cloud):
                
        # Estimate normals using Open3D. The radius or the number of neighbors used to estimate the normals can be set here.
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Optionally, reorient the normals so that they all point in a consistent direction
        # This can be useful if the normals are oriented inconsistently across the cloud
        point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
        
        # Convert the normals to a NumPy array for processing
        normals = np.asarray(point_cloud.normals)

        # Find indices where the z-component of the normal is negative
        negative_z = normals[:, 2] < 0

        # Flip the normals that have a negative z-component
        normals[negative_z] = -normals[negative_z]
        normals = o3d.utility.Vector3dVector(normals)
                
        return normals

    def process(self, data):
        """ Loads point clouds from an .npz file and converts them to Open3D point clouds. """
        point_clouds = []
        
        # Iterate through each point cloud in the .npz file
        for key in data:
            xyzrgb = data[key]  # Load each point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])  # xyz coordinates
            point_cloud.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:6])  # rgb colors
            point_clouds.append(point_cloud)

        for i in range(1, len(point_clouds)):
            point_clouds[i] = self.icp_pointcloud(point_clouds[i], point_clouds[i-1])
            point_clouds[i].normals = self.cal_norm(point_clouds[i])
 
        # Combine all processed point clouds into one
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            combined_pcd += pcd

        return combined_pcd

    def vis_pc(self, point_cloud):
        """ Visualizes the given point cloud using Open3D """
        o3d.visualization.draw_geometries([point_cloud])

    def save_pc(self, point_cloud, filename):
        """ Saves the processed point cloud as a .npz file in the format {'coords': ..., 'features': ...} """
        # Extract the points (coordinates) and colors (features)
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        # Create a dictionary with 'coords' and 'features'
        visual_data = {
            'points': points,
            'colors': colors,
        }
        
        # Save to npz file
        np.savez(filename, **visual_data)
        print(f"Point cloud saved to {filename}.npz")

if __name__ == "__main__":

    process_point_cloud = ProcessVisionPointCloud()
    data = np.load("vision.npz")
    combined_pcd = process_point_cloud.process(data)
    process_point_cloud.vis_pc(combined_pcd)

    # Save the processed point cloud as an .npz file
    process_point_cloud.save_pc(combined_pcd, "processed_vision")


