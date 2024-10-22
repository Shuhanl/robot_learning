import torch
import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Custom Kernel Combining Spatial and Visual Features
class CombinedKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel_spatial, base_kernel_visual, **kwargs):
        super(CombinedKernel, self).__init__(**kwargs)
        self.base_kernel_spatial = base_kernel_spatial
        self.base_kernel_visual = base_kernel_visual

    def forward(self, x1, x2, diag=False, **params):
        # Split inputs into spatial and visual components
        x1_spatial = x1[:, :3]
        x1_visual = x1[:, 3:]
        x2_spatial = x2[:, :3]
        x2_visual = x2[:, 3:]

        # Compute kernels
        k_spatial = self.base_kernel_spatial(x1_spatial, x2_spatial, diag=diag, **params)
        k_visual = self.base_kernel_visual(x1_visual, x2_visual, diag=diag, **params)

        # Combine kernels (e.g., multiply)
        return k_spatial * k_visual

# GP Regression Model with Combined Kernel
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Base kernels for spatial and visual features
        self.base_kernel_spatial = gpytorch.kernels.RBFKernel(ard_num_dims=3)
        self.base_kernel_visual = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim - 3)

        # Combined kernel
        self.covar_module = CombinedKernel(
            base_kernel_spatial=self.base_kernel_spatial,
            base_kernel_visual=self.base_kernel_visual
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Class for Training and Inference
class GPRegression:
    def __init__(self, train_x, train_y, num_steps=100, device='cpu'):
        self.device = device
        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y.squeeze(), dtype=torch.float32).to(self.device)
        self.num_steps = num_steps
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood, train_x.shape[1]).to(self.device)

    def train(self):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.num_steps):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 or i == self.num_steps - 1:
                print(f"Iter {i + 1}/{self.num_steps} - Loss: {loss.item():.3f}")

    def predict(self, test_x):
        self.model.eval()
        self.likelihood.eval()
        test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
        mean = observed_pred.mean.cpu().numpy()
        variance = observed_pred.variance.cpu().numpy()
        return mean, variance

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_visual_data(self, filepath):
        # Load data from .npz file
        data = np.load(filepath)
        # Assuming the .npz file contains 'points' and 'colors' arrays
        if 'points' not in data or 'colors' not in data:
            raise ValueError("The .npz file must contain 'points' and 'colors' arrays.")
        
        visual_coords = data['points']
        visual_colors = data['colors']

        # Check if the arrays are not empty
        if visual_coords.size == 0:
            raise ValueError("The 'points' array in the .npz file is empty.")
        if visual_colors.size == 0:
            raise ValueError("The 'colors' array in the .npz file is empty.")

        # Normalize RGB values to [0, 1] if necessary
        if visual_colors.max() > 1.0:
            visual_colors = visual_colors / 255.0

        return visual_coords, visual_colors
    
    def load_tactile_data(self, filename):
        """
        Load the point cloud data from the .npz file into memory.
        """
        try:
            data = np.load(filename)
            coords = data['coords']     # 3D coordinates of the points (x, y, z)
            friction = data['friction'] # Friction values for each point
            stiffness = data['stiffness'] # Stiffness values for each point
            print(f"Loaded point cloud data from {filename}")

        except Exception as e:
            print(f"Failed to load data: {e}")
        
        return coords, friction, stiffness

    def get_visual_features_for_tactile_points(self, tactile_coords, visual_coords, visual_features):
        # Build KD-Tree from visual coordinates
        visual_tree = cKDTree(visual_coords)

        # Query nearest visual points for each tactile point
        distances, indices = visual_tree.query(tactile_coords, k=1)

        # Extract visual features for tactile points
        tactile_visual_features = visual_features[indices]

        return tactile_visual_features
    
    def normalize_data(self, train_x, test_x):
        self.scaler.fit(train_x)
        train_x_norm = self.scaler.transform(train_x)
        test_x_norm = self.scaler.transform(test_x)
        return train_x_norm, test_x_norm

# Main GP Class
class GPModel:
    def __init__(self, tactile_data, visual_data, device='cpu'):
        self.device = device
        self.tactile_coords = tactile_data['coords']
        self.tactile_visual_features = tactile_data['visual_features']
        self.tactile_friction = tactile_data['friction']
        self.tactile_stiffness = tactile_data['stiffness']

        self.visual_coords = visual_data['coords']
        self.visual_features = visual_data['colors']

        self.data_processor = DataProcessor()

    def train_tactile_gp(self, num_steps=100):
        # Prepare training data
        train_x = np.hstack((self.tactile_coords, self.tactile_visual_features))
        train_y_friction = self.tactile_friction
        train_y_stiffness = self.tactile_stiffness

        # Normalize data
        self.train_x, _ = self.data_processor.normalize_data(train_x, train_x)

        # Train GP model for friction
        self.tactile_gp_friction = GPRegression(self.train_x, train_y_friction, num_steps=num_steps, device=self.device)
        self.tactile_gp_friction.train()

        # Train GP model for stiffness
        self.tactile_gp_stiffness = GPRegression(self.train_x, train_y_stiffness, num_steps=num_steps, device=self.device)
        self.tactile_gp_stiffness.train()

    def infer_tactile_properties(self):
        # Prepare test data
        test_x = np.hstack((self.visual_coords, self.visual_features))
        _, self.test_x = self.data_processor.normalize_data(self.train_x, test_x)

        # Predict friction at visual points
        mean_friction, _ = self.tactile_gp_friction.predict(self.test_x)
        # Predict stiffness at visual points
        mean_stiffness, _ = self.tactile_gp_stiffness.predict(self.test_x)

        # Store predicted tactile properties
        self.predicted_friction = mean_friction
        self.predicted_stiffness = mean_stiffness

    def visualize_tactile_properties(self):
        # Visualize tactile properties on the point cloud
        fig = plt.figure(figsize=(12, 6))

        # Friction
        ax1 = fig.add_subplot(121, projection='3d')
        p = ax1.scatter(self.visual_coords[:, 0], self.visual_coords[:, 1], self.visual_coords[:, 2],
                        c=self.predicted_friction, cmap='viridis', marker='.', s=1)
        ax1.set_title('Predicted Friction')
        fig.colorbar(p, ax=ax1)

        # Stiffness
        ax2 = fig.add_subplot(122, projection='3d')
        p = ax2.scatter(self.visual_coords[:, 0], self.visual_coords[:, 1], self.visual_coords[:, 2],
                        c=self.predicted_stiffness, cmap='viridis', marker='.', s=1)
        ax2.set_title('Predicted Stiffness')
        fig.colorbar(p, ax=ax2)

        plt.show()

    def save_point_cloud_with_tactile(self, filename="point_cloud_with_tactile.npz"):
        # Combine all data
        combined_coords = self.visual_coords
        combined_colors = self.visual_features
        combined_friction = self.predicted_friction.reshape(-1, 1)
        combined_stiffness = self.predicted_stiffness.reshape(-1, 1)

        # Save to .npz file
        np.savez(filename,
                 coords=combined_coords,
                 colors=combined_colors,
                 friction=combined_friction,
                 stiffness=combined_stiffness)
        
if __name__ == "__main__":
    # Initialize data processor
    data_processor = DataProcessor()

    # Load visual data from .npz file
    visual_coords, visual_colors = data_processor.load_visual_data("processed_vision.npz")
    tactile_coords, friction, stiffness = data_processor.load_tactile_data("tactile.npz")

    # Get visual features for tactile points
    tactile_visual_features = data_processor.get_visual_features_for_tactile_points(
        tactile_coords, visual_coords, visual_colors
    )

    # Prepare tactile and visual data dictionaries
    tactile_data = {
        'coords': tactile_coords,
        'visual_features': tactile_visual_features,
        'friction': friction,
        'stiffness': stiffness,
    }

    visual_data = {
        'coords': visual_coords,
        'colors': visual_colors,
    }

    # Initialize GP model
    gp_model = GPModel(tactile_data, visual_data, device='cpu')

    # Train tactile GP models
    gp_model.train_tactile_gp(num_steps=100)

    # Infer tactile properties at visual points
    gp_model.infer_tactile_properties()

    # Visualize tactile properties
    gp_model.visualize_tactile_properties()

    # Save point cloud with inferred tactile properties
    gp_model.save_point_cloud_with_tactile(filename="point_cloud_with_tactile.npz")

