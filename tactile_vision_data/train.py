import torch
import numpy as np
import os
import random
from gaussian_splatting.render import Renderer
from gaussian_splatting.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss
from parameters import GSParams, CameraParams


def train_gs(dataset_dir="dataset"):

    gs_params = GSParams()
    gaussians = GaussianModel(gs_params.sh_degree, gs_params.distill_feature_dim)

    camera_params = CameraParams()
    gaussians.training_setup(gs_params)
    render = Renderer(gaussians)

    camera_calib = np.load('config/camera_calib.npy')  # Extrinsic calibration matrix
    camera_pose = camera_calib['extrinsics']
    intrinsics = camera_calib['intrinsics']

    # Get the list of all available data files
    data_files = [f.split('_')[1].split('.')[0] for f in os.listdir(dataset_dir) if f.startswith("rgb")]
    data_indices = sorted(list(set(data_files)))  # Ensure unique and sorted indices

    white_background = False

    for iteration in range(0, gs_params.iterations):

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Randomly sample an index from the available data
        sampled_index = random.choice(data_indices)

        # Load the sampled RGB, depth, and pose data
        rgb_path = os.path.join(dataset_dir, f"rgb_{sampled_index}.npy")
        depth_path = os.path.join(dataset_dir, f"depth_{sampled_index}.npy")
        pose_path = os.path.join(dataset_dir, f"pose_{sampled_index}.npy")
        
        rgb = np.load(rgb_path)
        depth = np.load(depth_path)
        robot_pose = np.load(pose_path)

        # Combine robot pose and camera pose to get the extrinsics (world-to-camera transform)
        extrinsics  = torch.tensor(robot_pose @ camera_pose, dtype=torch.float32, device="cuda")

        render_pkg  = render.render(intrinsics, extrinsics, camera_params.image_width, camera_params.image_height)

        image = render_pkg["render"]
        rendered_depth = render_pkg["render_depth"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Reconstruction Loss
        gt_image = torch.tensor(rgb, dtype=torch.float32, device="cuda")
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 

        # Depth loss
        gt_depth = torch.tensor(depth, dtype=torch.float32, device="cuda")
        depth_loss = l1_loss(rendered_depth, gt_depth)
        loss = loss + depth_loss

        loss.backward()

        with torch.no_grad():

            if iteration < gs_params.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > gs_params.densify_from_iter and iteration % gs_params.densification_interval == 0:
                    size_threshold = 20 if iteration > gs_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(gs_params.densify_grad_threshold, 0.005, camera_params.cameras_extent, size_threshold)
                
                if iteration % gs_params.opacity_reset_interval == 0 or (white_background and iteration == gs_params.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)