import torch

# Data parameters
vision_dim = (3, 128, 128)
proprioception_dim = 7
action_dim = 7
num_distribs = 10
batch_size = 2

# Network parameters
vision_embedding_dim = 64
latent_dim = 256
memory_size = int(1e6)
qbits = 8

# Training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
grad_norm_clipping = 0.5

N_EPOCH = 50
WINDOW_SIZE = 32
BS = 64
BETA = 0.01
NUM_WORKERS = 12
