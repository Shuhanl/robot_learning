import torch
import numpy as np
import os
from gaussian_splatting.render import Renderer
from gaussian_splatting.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss
from parameters import OptimizationParams, CameraParams


def train_gs(dataset_dir="dataset"):
    sh_degree  = 2
    distill_feature_dim = 3
    gaussians = GaussianModel(sh_degree, distill_feature_dim)

    opt = OptimizationParams()
    camera_params = CameraParams()
    gaussians.training_setup(opt)
    render = Renderer(gaussians)

    camera_pose = np.load('config/camera_calib.npy')

    white_background = False

    for iteration in range(0, opt.iterations):

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Load RGB, depth, and pose data for the current iteration
        rgb = np.load(os.path.join(dataset_dir, f"rgb_{iteration:04d}.npy"))
        depth = np.load(os.path.join(dataset_dir, f"depth_{iteration:04d}.npy"))
        robot_pose = np.load(os.path.join(dataset_dir, f"pose_{iteration:04d}.npy"))

        # Combine robot pose and camera pose to get the extrinsics (world-to-camera transform)
        extrinsics  = torch.tensor(robot_pose @ camera_pose, dtype=torch.float32, device="cuda")

        render_pkg  = render.render(extrinsics , camera_params.image_width, camera_params.image_height)

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

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, camera_params.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)