import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
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

    def visualize_point_cloud(self, point_cloud):
        """ Visualizes the given point cloud using Open3D """
        o3d.visualization.draw_geometries([point_cloud])
    
    def save_point_cloud(self, point_cloud, filename):
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

class ProcessTactilePointCloud:
    def __init__(self):
        """
        Initialize the visualizer by loading the point cloud data from a .npz file.
        """
        self.point_cloud = None
        self.coords = None
        self.friction = None
        self.stiffness = None

    def load_data(self, filename):
        """
        Load the point cloud data from the .npz file into memory.
        """
        try:
            data = np.load(filename)
            self.coords = data['coords']     # 3D coordinates of the points (x, y, z)
            self.friction = data['friction'] # Friction values for each point
            self.stiffness = data['stiffness'] # Stiffness values for each point
            print(f"Loaded point cloud data from {filename}")

        except Exception as e:
            print(f"Failed to load data: {e}")

    def normalize_to_rgb(self, values):
        """
        Normalize a scalar array to RGB values for coloring.
        
        :param values: Scalar array (e.g., friction or stiffness values)
        :return: Nx3 array of RGB values.
        """
        # Normalize the values to be between 0 and 1
        normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)

        # Create an RGB colormap (you can customize this to any colormap you prefer)
        cmap = plt.get_cmap('viridis')
        rgb_colors = cmap(normalized_values)[:, :3]  # Extract RGB from RGBA
        
        return rgb_colors

    def visualize_point_cloud(self, color_by='friction'):
        """
        Visualize the tactile point cloud in 3D using Open3D.
        
        :param color_by: Choose which attribute to color by ('friction' or 'stiffness').
        """
        if self.coords is None or self.friction is None or self.stiffness is None:
            print("No point cloud data available for visualization.")
            return
        
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()

        # Assign the coordinates (Nx3)
        pcd.points = o3d.utility.Vector3dVector(self.coords)

        # Color points by either friction or stiffness
        if color_by == 'friction':
            color = self.normalize_to_rgb(self.friction.flatten())  # Normalize friction to RGB colors
        elif color_by == 'stiffness':
            color = self.normalize_to_rgb(self.stiffness.flatten())  # Normalize stiffness to RGB colors
        else:
            color = np.zeros((self.coords.shape[0], 3))  # Default color (black)

        # Set point cloud colors
        pcd.colors = o3d.utility.Vector3dVector(color)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd], window_name=f"Tactile Point Cloud colored by {color_by}")

    def get_tactile_pcd(self, radius=0.0005):
        """
        Create an Open3D PointCloud object for the tactile data.
        The tactile points are colored red.
        """
        if self.coords is None:
            print("No point cloud data available.")
            return None

        color = np.array([1.0, 0.0, 0.0])
        spheres = []
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere_mesh.compute_vertex_normals()
        sphere_mesh.paint_uniform_color(color)

        for point in self.coords:
            sphere = copy.deepcopy(sphere_mesh)
            sphere.translate(point)
            spheres.append(sphere)

        return spheres
    


if __name__ == "__main__":

    process_vision_pc = ProcessVisionPointCloud()
    data = np.load("vision.npz")
    processed_pcd = process_vision_pc.process(data)
    process_vision_pc.visualize_point_cloud(processed_pcd)
    process_vision_pc.save_point_cloud(processed_pcd, "processed_vision_pcd")

    process_tactile_pc = ProcessTactilePointCloud()
    process_tactile_pc.load_data(filename="tactile.npz")
    process_tactile_pc.visualize_point_cloud(color_by='friction')
    process_tactile_pc.visualize_point_cloud(color_by='stiffness')

    # Get tactile point cloud colored by friction
    tactile_pcd = process_tactile_pc.get_tactile_pcd()

    o3d.visualization.draw_geometries([processed_pcd] + tactile_pcd, window_name="Combined Point Cloud with Tactile Points Highlighted")