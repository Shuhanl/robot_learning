import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical, Distribution, AffineTransform, TransformedDistribution, SigmoidTransform
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform
import parameters as params


def init_linear(module):
    """
    Initialize linear layers with Xavier uniform initialization and set biases to a small constant.
    This function now handles Sequential modules containing linear layers.
    """
    if isinstance(module, nn.Linear):  # Check if the module is a linear layer
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)  # Small constant to avoid dead neurons
    elif isinstance(module, nn.Sequential):  # Check if it's a Sequential module
        for sub_module in module:
            init_linear(sub_module)  # Recursively apply to sub-modules


class VisionNetwork(nn.Module):
    def __init__(self, ):
        super(VisionNetwork, self).__init__()

        img_channels = params.vision_dim[0]
        img_height = params.vision_dim[1]
        img_width = params.vision_dim[2]
        self.device = params.device
        self.out_dim = params.d_model

        # Encoder Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        test_tensor = torch.zeros(
            [img_channels, img_height, img_width]
        )
        with torch.no_grad():
            pre_softmax = self.cnn(test_tensor[None])
            N, C, H, W = pre_softmax.shape

            self.fc = nn.Sequential(nn.Linear(2*C, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.out_dim))

    def _spatial_softmax(self, pre_softmax):
        N, C, H, W = pre_softmax.shape
        pre_softmax = pre_softmax.view(N*C, H * W)
        softmax = F.softmax(pre_softmax, dim=1)
        softmax = softmax.view(N, C, H, W)

        # Create normalized meshgrid
        x_coords = torch.linspace(0, 1, W, device=self.device)
        y_coords = torch.linspace(0, 1, H, device=self.device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        image_coords = torch.stack([X, Y], dim=-1).to(self.device)  # [H, W, 2]

        image_coords = image_coords.unsqueeze(0)  # [1, H, W, 2]
        image_coords = image_coords.unsqueeze(0)  # [1, H, W, 2] -> [1, 1, H, W, 2]

        # Reshape softmax for broadcasting
        softmax = softmax.unsqueeze(-1)  # [N, C, H, W, 1]

        # Compute spatial soft argmax
        # This tensor represents the 'center of mass' for each channel of each feature map in the batch
        spatial_soft_argmax = torch.sum(
            softmax * image_coords, dim=[2, 3])  # [N, C, 2]
        x = nn.Flatten()(spatial_soft_argmax)  # [N, C, 2] -> [N, 2*C]
        return x

    def forward(self, x):
        pre_softmax = self.cnn(x)
        spatial_soft_argmax = self._spatial_softmax(pre_softmax)
        x = self.fc(spatial_soft_argmax)
        return x


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.proprioception_dim = params.proprioception_dim
        self.action_dim = params.action_dim
        self.d_model = params.d_model
        self.sequence_length = params.sequence_length

        self.vision_embedding = VisionNetwork()
        self.proprioception_embedding = torch.nn.Linear(self.proprioception_dim, self.d_model)
        self.action_embedding = torch.nn.Linear(self.action_dim, self.d_model)
        self.position_embedding = nn.Embedding(self.sequence_length, self.d_model)

        init_linear(self.proprioception_embedding)
        init_linear(self.action_embedding)  

    def vision_embed(self, x):
        return self.vision_embedding(x)

    def proprioception_embed(self, x):
        return self.proprioception_embedding(x)

    def action_embed(self, x):
        return self.action_embedding(x)

    def position_embed(self, x):
        return self.position_embedding(x)

class PlanRecognition(nn.Module):
    def __init__(self):
        super(PlanRecognition, self).__init__()
        self.layer_size = 2048
        self.nhead = params.n_heads
        self.epsilon = 1e-4
        self.sequence_length = params.sequence_length
        self.proprioception_dim = params.proprioception_dim
        self.latent_dim = params.latent_dim
        self.d_model = params.d_model
        self.batch_size = params.batch_size
        self.device = params.device
        self.embedding = EmbeddingNetwork()

        # Define the Transformer encoder layer (self-attention on the same sensor embedding sequence)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.layer_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

        # Linear layers for the latent variables
        self.fc_mu = nn.Linear(self.d_model, self.latent_dim)
        self.fc_sigma = nn.Linear(self.d_model, self.latent_dim)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)  # Xavier initialization for weight matrices
            else:
                init.constant_(p, 0)  # Zero initialization for biases
        
        init_linear(self.fc_mu)
        init_linear(self.fc_sigma)

    def latent_normal(self, mu, sigma):
        dist = Normal(loc=mu, scale=sigma)
        return dist

    def forward(self, vision_embedded, proprioception_embedded):

        vision_embedded = vision_embedded[:, -self.sequence_length:, :]
        proprioception_embedded = proprioception_embedded[:, -self.sequence_length:, :]

        position = torch.arange(vision_embedded.shape[1], device=self.device).expand((self.batch_size, vision_embedded.shape[1])).contiguous()
        position_embedded = self.embedding.position_embed(position)

        # pad all tokens to sequence length
        vision_embedded = torch.cat([torch.zeros((vision_embedded.shape[0], self.sequence_length -
                                    vision_embedded.shape[1], self.d_model), device=self.device), vision_embedded], dim=1)
        proprioception_embedded = torch.cat([torch.zeros((proprioception_embedded.shape[0], self.sequence_length -
                                            proprioception_embedded.shape[1], self.d_model), device=self.device), proprioception_embedded], dim=1)
        position_embedded = torch.cat([torch.zeros((position_embedded.shape[0], self.sequence_length -
                                      position_embedded.shape[1], self.d_model), device=self.device), position_embedded], dim=1)
        
        # Add position embeddings
        vision_embedded += position_embedded
        proprioception_embedded += position_embedded

        # this makes the sequence look like (vision_1, pro_1, vision_2, pro_2, ...)
        # which works nice in an autoregressive sense 
        x = torch.stack((vision_embedded, proprioception_embedded), dim=1
        ).permute(0, 2, 1, 3).reshape(-1, 2*self.sequence_length, self.d_model)  # (bs, 2*seq_len, d_model)

        # Encoder
        x = self.transformer_encoder(x)

        # Process output for latent variables as before
        x = x[:, -1, :]
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x) + self.epsilon)
        dist = self.latent_normal(mu, sigma)

        return dist


class PlanProposal(nn.Module):
    def __init__(self):
        super(PlanProposal, self).__init__()
        self.layer_size = 2048
        self.epsilon = 1e-4

        # Encoder Layers
        self.in_features = 3*params.d_model
        self.latent_dim = params.latent_dim

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.layer_size),
                                nn.ReLU(),
                                nn.Linear(self.layer_size, self.layer_size),
                                nn.ReLU(),
                                nn.Linear(self.layer_size, self.layer_size),
                                nn.ReLU(),
                                nn.Linear(self.layer_size, self.layer_size),
                                nn.ReLU())
        self.fc_mu = nn.Linear(self.layer_size, self.latent_dim)
        self.fc_sigma = nn.Linear(self.layer_size, self.latent_dim)

        init_linear(self.fc)
        init_linear(self.fc_mu)
        init_linear(self.fc_sigma)

    def latent_normal(self, mu, sigma):
        dist = Normal(loc=mu, scale=sigma)
        return dist

    def forward(self, vision_embedded, proprioception_embedded, goal_embedded):

        x = torch.cat(
            [vision_embedded, proprioception_embedded, goal_embedded], dim=1)  # (bs, 3*d_model)
        x = self.fc(x)
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x)+self.epsilon)
        dist = self.latent_normal(mu, sigma)

        return dist


class LogisticMixture(Distribution):
    def __init__(self, weightings, mu, scale, qbits=None):
        """
        Initializes the logistic mixture model.
        :param weightings: The logits for the categorical distribution.
        :param mu: The means of the logistic distributions.
        :param scale: The scales of the logistic distributions.
        :param qbits: Number of quantization bits.
        """
        super(LogisticMixture, self).__init__()
        arg_constraints = {}

        # Create a uniform distribution as the base for the logistic transformation
        base_distributions = Uniform(torch.zeros_like(mu), torch.ones_like(mu))

        # Apply the logistic transformation to the uniform base
        # Logistic(x; mu, scale) = mu + scale * log(x / (1-x))
        # Define transforms: Inverse of sigmoid followed by an affine transformation
        transforms = [SigmoidTransform().inv, AffineTransform(
            loc=mu, scale=scale)]

        # Create the transformed distribution representing the logistic distribution
        logistic_distributions = TransformedDistribution(
            base_distributions, transforms)

        # Create the mixture distribution
        self.mixture_dist = MixtureSameFamily(
            mixture_distribution=Categorical(logits=weightings),
            component_distribution=logistic_distributions,
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Samples from the logistic mixture model.
        """
        x = self.mixture_dist.sample(sample_shape)
        x = torch.clamp(x, -1, 1)
        return x

    def log_prob(self, value):
        log_prob = self.mixture_dist.log_prob(value)
        return log_prob

class Actor(nn.Module):
    def __init__(
        self,
        layer_size=1024,
    ):
        super(Actor, self).__init__()
        self.sequence_length = params.sequence_length
        self.nhead = params.n_heads
        self.latent_dim = params.latent_dim
        self.d_model = params.d_model
        self.action_dim = params.action_dim
        self.num_distribs = params.num_distribs
        self.device = params.device
        self.batch_size = params.batch_size
        self.epsilon = params.epsilon
        self.embedding = EmbeddingNetwork()

        # Transformer Encoder Layer for self-attention on the same embedding sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=layer_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Transformer Decoder Layer for cross-attention on different embedding sequences
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=layer_size, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # self.fc = nn.Sequential(
        #     nn.Linear(self.d_model, self.action_dim), nn.Tanh())

        self.alpha = nn.Linear(self.d_model, self.action_dim * self.num_distribs)
        self.mu = nn.Linear(self.d_model, self.action_dim * self.num_distribs)
        self.sigma = nn.Linear(self.d_model, self.action_dim * self.num_distribs)

        # Initialize transformer encoder and decoder weights
        for module in [self.transformer_encoder, self.transformer_decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    init.xavier_uniform_(p)  # Xavier initialization for weight matrices
                else:
                    init.constant_(p, 0)  # Zero initialization for biases      

        init_linear(self.alpha)
        init_linear(self.mu)
        init_linear(self.sigma)

    def causal_mask(self, size):
        """
        Create a causal mask to prevent positions from attending to future positions.
        """
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1).bool()
        return mask

    def forward(self, vision_embedded, proprioception_embedded, latent, goal_embedded, action_embedded, position_embedded):

        sequence_length = vision_embedded.size(1)
        mask = self.causal_mask(sequence_length)

        vision_embedded += position_embedded
        proprioception_embedded += position_embedded
        action_embedded += position_embedded
        
        # this makes the sequence look like (vision_1, pro_1, vision_2, pro_2, ... goal)
        # which works nice in an autoregressive sense since states predict actions

        x = torch.stack((vision_embedded, proprioception_embedded), dim=1
        ).permute(0, 2, 1, 3).reshape(-1, 2*self.sequence_length, self.d_model)  # (bs, 2*seq_len, d_model)

        x = torch.cat((x, latent, goal_embedded), dim=1)  # (bs, 2*seq_len+2, d_model)

        x = self.transformer_encoder(x)  # (bs, 2*seq_len+2, d_model)

        # x = self.transformer_decoder(action_embedded, x, tgt_mask=mask) # (bs, seq_len, d_model)

        # print("NaN in x:", torch.isnan(x).any())

        # Use the last decoder output for generating the action
        x = x[:, -1, :]  # (bs, d_model)

        weightings = self.alpha(x).view(-1, self.action_dim, self.num_distribs)
        mu = self.mu(x).view(-1, self.action_dim, self.num_distribs)
        scale = nn.functional.softplus(self.sigma(x)+self.epsilon).view(-1, self.action_dim, self.num_distribs)
        logistic_mixture = LogisticMixture(weightings, mu, scale)

        return logistic_mixture

    def get_action(self, vision_embedded, proprioception_embedded, latent, goal_embedded, action_embedded):
        """
        Get the action for the current state
        """

        vision_embedded = vision_embedded[:, -self.sequence_length:, :]
        proprioception_embedded = proprioception_embedded[:, -self.sequence_length:, :]
        action_embedded = action_embedded[:, -self.sequence_length:, :]
        latent = latent.unsqueeze(1)
        goal_embedded = goal_embedded.unsqueeze(1)

        position = torch.arange(vision_embedded.shape[1], device=self.device).expand((1, vision_embedded.shape[1])).contiguous()
        position_embedded = self.embedding.position_embed(position)

        # pad all tokens to sequence length
        vision_embedded = torch.cat([torch.zeros((vision_embedded.shape[0], self.sequence_length -
                                    vision_embedded.shape[1], self.d_model), device=self.device), vision_embedded], dim=1)
        proprioception_embedded = torch.cat([torch.zeros((proprioception_embedded.shape[0], self.sequence_length -
                                            proprioception_embedded.shape[1], self.d_model), device=self.device), proprioception_embedded], dim=1)
        action_embedded = torch.cat([torch.zeros((action_embedded.shape[0], self.sequence_length -
                                    action_embedded.shape[1], self.d_model), device=self.device), action_embedded], dim=1)

        position_embedded = torch.cat([torch.zeros((position_embedded.shape[0], self.sequence_length -
                                      position_embedded.shape[1], self.d_model), device=self.device), position_embedded], dim=1)

        logistic_mixture = self.forward(vision_embedded, proprioception_embedded, latent, goal_embedded, action_embedded, position_embedded)

        action = logistic_mixture.sample()
        action_log_prob = logistic_mixture.log_prob(action)

        return action, action_log_prob


class Critic(nn.Module):
    def __init__(self, layer_size=1024):
        super(Critic, self).__init__()
        self.in_features = 3*params.d_model
        # Fully Connected Layers
        self.fc = nn.Sequential(nn.Linear(self.in_features, layer_size),
                                nn.ReLU(),
                                nn.Linear(layer_size, layer_size),
                                nn.ReLU(),
                                nn.Linear(layer_size, layer_size),
                                nn.ReLU(),
                                nn.Linear(layer_size, 1),
                                nn.ReLU())

    def forward(self, vision_embedded, proprioception_embedded, action_embedded):

        x = torch.cat([vision_embedded, proprioception_embedded,
                      action_embedded], dim=-1)
        q_value = self.fc(x)

        return q_value
