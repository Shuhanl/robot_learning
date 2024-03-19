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
mamba_model_dim = 128

# Training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
grad_norm_clipping = 0.5


# User-defined hyperparameters
d_model = 8
state_size = 128  # Example state size
seq_len = 100  # Example sequence length
batch_size = 256  # Example batch size
last_batch_size = 81  # only for the very last batch of the dataset
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None