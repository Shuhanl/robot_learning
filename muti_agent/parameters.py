import torch

# Data parameters
arm_state_dim = 14
leg_state_dim = 12
body_state_dim = 6

arm_act_dim = 14
leg_act_dim = 12
body_act_dim = 3

latent_dim = 64
num_distribs = 10
batch_size = 100
num_workers = 2

# Network parameters

# Training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
grad_norm_clipping = 0.5
beta = 0.01
tau = 0.01
num_episodes = 100
