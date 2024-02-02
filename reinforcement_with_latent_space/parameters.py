import torch

# Data parameters
vision_dim = (3, 299, 299)
proprioception_dim = 8
action_dim = 8
sequence_length = 100

# Network parameters
vision_embedding_dim = 64
latent_dim = 256
memory_size = int(1e6)
batch_size=1024

# Training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
grad_norm_clipping = 0.5

N_EPOCH = 50
WINDOW_SIZE = 32
BS = 64
LR = 2e-4
BETA = 0.01
NUM_WORKERS = 12
