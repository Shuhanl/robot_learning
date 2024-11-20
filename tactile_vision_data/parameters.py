import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GSParams():
    def __init__(self):
        self.sh_degree  = 2
        self.distill_feature_dim = 3
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.distill_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 2000
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002

        # Feature splatting optimization parameters
        self.update_decoder_until_iter = 2000
        self.update_features_until_iter = 2500

class CameraParams():
    def __init__(self):
        self.image_width = 1280
        self.image_height = 720
        self.cameras_extent = [(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)]
