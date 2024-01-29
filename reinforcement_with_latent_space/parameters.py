import torch

# Trainer flags
GPUS = 3
NUM_NODES = 1
ACCELERATOR = "ddp"
PLUGINS = "ddp_sharded"
PRECISION = 16
MAX_STEPS = 10
LIMIT_TRAIN_BATCHES = 10
VAL_CHECK_INTERVAL = 0.5  # i.e. 0.1 means 10 times every epoch
SWA = True
FAST_DEV_RUN = False
LOG_GPU_MEMORY = True
PROFILER = "simple"  # one of 'simple' or 'advanced' (i.e. function level) or 'pytorch' (https://pytorch-lightning.readthedocs.io/en/stable/profiler.html)

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
