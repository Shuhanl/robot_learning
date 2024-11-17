import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh


class Renderer:
    def __init__(self, GaussianModel):
        """
        Initialize the Renderer.

        Args:
        - GaussianModel: An instance of the GaussianModel class.
        """
        self.pc = GaussianModel  # Point cloud Gaussian model

    def compute_camera(intrinsics, robot_pose, camera_pose, image_width, image_height, znear=0.1, zfar=1000.0):

        # Compute field of view
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        FoVx = 2 * torch.atan(image_width / (2 * fx))
        FoVy = 2 * torch.atan(image_height / (2 * fy))

        # Combine robot pose and camera pose to get the extrinsics (world-to-camera transform)
        world_view_transform = torch.tensor(robot_pose @ camera_pose, dtype=torch.float32, device="cuda")

        # Compute projection matrix
        full_proj_transform = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        full_proj_transform[0, 0] = 2 * fx / image_width
        full_proj_transform[1, 1] = 2 * fy / image_height
        full_proj_transform[0, 2] = 1 - (2 * cx / image_width)
        full_proj_transform[1, 2] = (2 * cy / image_height) - 1
        full_proj_transform[2, 2] = -(zfar + znear) / (zfar - znear)
        full_proj_transform[2, 3] = -(2 * zfar * znear) / (zfar - znear)
        full_proj_transform[3, 2] = -1.0

        view_inv = torch.inverse(world_view_transform)
        camera_center = view_inv[3][:3]

        # Return MiniCam object
        return image_width,  image_height, FoVy, FoVx, world_view_transform, full_proj_transform, camera_center

    def render(self,
               intrinsics, robot_pose, camera_pose, image_width, image_height,
               pipe,
               bg_color: torch.Tensor,
               scaling_modifier=1.0,
               override_color=None,
               render_features=False,
               render_gaussian_idx=False):
        """
        Render the scene. 

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.pc.get_xyz, dtype=self.pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        image_width,  image_height, FoVy, FoVx, world_view_transform, full_proj_transform, camera_center = self.compute_camera(intrinsics, robot_pose, camera_pose, image_width, image_height)

        # Set up rasterization configuration
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.pc.active_sh_degree,
            campos=camera_center
            prefiltered=False,
            render_features=render_features,
            render_gaussian_idx=render_gaussian_idx,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.pc.get_xyz
        means2D = screenspace_points
        opacity = self.pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.pc.get_covariance(scaling_modifier)
        else:
            scales = self.pc.get_scaling
            rotations = self.pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = self.pc.get_features.transpose(
                    1, 2).view(-1, 3, (self.pc.max_sh_degree+1)**2)
                dir_pp = (
                    self.pc.get_xyz - self.viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.pc.active_sh_degree,
                                 shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
            else:
                shs = self.pc.get_features  # (N, 16 ,3)
        else:
            colors_precomp = override_color

        # Get view-independent features (distill features) for each Gaussian for rendering.
        distill_feats = self.pc.get_distill_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            distill_feats=distill_feats)

        return {"render": rendered_image,
                "render_feat": rendered_feat,
                "render_depth": rendered_depth,
                "render_gaussian_idx": rendered_gaussian_idx,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii}
