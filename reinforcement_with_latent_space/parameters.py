import torch

# Sensing parameters 
vision_dim = (8, 128, 128)
proprioception_dim = 7

# Latent space parameters
latent_dim = 512

# Action parameters
action_dim = 7
num_distribs = 10
qbits = 8

# Transformer parameters
n_heads = 8
d_model = 512
sequence_length = 100

# PPO parameters
epsilon =  0.2 # clip parameter for PPO
gamma = 0.99
lmbda = 0.95
entropy_weight = 1e-4
tau = 0.01
rollout_length = 2048

# Target RL parameters
target_tau = 0.01
target_gamma = 0.99

# Training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
grad_norm_clipping = 0.5
beta = 0.01
memory_size = int(1e3)
num_episodes = 100
batch_size = 5
num_workers = 2
