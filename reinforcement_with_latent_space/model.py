import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
import parameters as params
    
class VisionNetwork(nn.Module):
    def __init__(self, ):
        super(VisionNetwork, self).__init__()

        img_channels=params.vision_dim[0]
        img_height=params.vision_dim[1]
        img_width=params.vision_dim[2]

        # Encoder Layers
        # In PyTorch, normalization is usually done as a preprocessing step, but it can be included in the model
        self.rescaling = lambda x: x / 255.0  
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4, padding=2)  # 'same' padding in PyTorch needs calculation
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

        # Calculate size after convolutions
        size_after_conv1 = self._size_in_features(img_height, 8, 4, 2)
        size_after_conv2 = self._size_in_features(size_after_conv1, 4, 2, 1)
        size_after_conv3 = self._size_in_features(size_after_conv2, 3, 1, 1)

        fc_input_size = 64 * size_after_conv3 * size_after_conv3
        self.dense1 = nn.Linear(fc_input_size, 512)
        self.dense2 = nn.Linear(512, params.vision_embedding_dim)

    def _size_in_features(self, size, kernel_size, stride, padding):
        return (size + 2*padding - kernel_size) // stride + 1
    
    def _spatial_softmax(self, pre_softmax):
        N, C, H, W = pre_softmax.shape
        pre_softmax = pre_softmax.view(N * C, H * W)
        softmax = F.softmax(pre_softmax, dim=1)
        softmax = softmax.view(N, C, H, W)

        # Create normalized meshgrid
        x_coords = torch.linspace(0, 1, W, device=params.device)
        y_coords = torch.linspace(0, 1, H, device=params.device)
        X, Y = torch.meshgrid(x_coords, y_coords)
        image_coords = torch.stack([X, Y], dim=-1).to(params.device)  # [H, W, 2]
        image_coords = image_coords.unsqueeze(2)  # [H, W, 1, 2]

        # Reshape softmax for broadcasting
        softmax = softmax.unsqueeze(-1)  # [N, C, H, W, 1]

        # Compute spatial soft argmax
        # This tensor represents the 'center of mass' for each channel of each feature map in the batch
        spatial_soft_argmax = torch.sum(softmax * image_coords, dim=[2, 3])  # [N, C, 2]

        return spatial_soft_argmax

    def forward(self, x):
        x = self.rescaling(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pre_softmax = F.relu(self.conv3(x))

        spatial_soft_argmax = self._spatial_softmax(pre_softmax)

        x = self.flatten(spatial_soft_argmax)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

        
class PlanRecognition(nn.Module):
    def __init__(self):
        super(PlanRecognition, self).__init__()
        self.layer_size=2048
        self.epsilon=1e-4

        # Encoder Layers
        self.lstm1 = nn.LSTM(params.sequence_length*(params.vision_embedding_dim + params.proprioception_dim + params.action_dim), self.layer_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * self.layer_size, self.layer_size, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * self.layer_size, params.latent_dim)
        self.sigma = nn.Linear(2 * self.layer_size, params.latent_dim)

    def latent_normal(self, mu, sigma):
        dist = Normal(loc=mu, scale=sigma)
        return dist

    def forward(self, x):

        # LSTM Layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Latent variable
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x+self.epsilon))
        dist = self.latent_normal(mu, sigma)

        return dist

    
class PlanProposal(nn.Module):
    def __init__(self):
        super(PlanProposal, self).__init__()
        self.layer_size=2048
        self.epsilon=1e-4

        # Encoder Layers
        self.in_features = params.vision_dim + params.proprioception_dim

        self.fc1 = nn.Linear(self.in_features, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, self.layer_size)
        self.fc4 = nn.Linear(self.layer_size, self.layer_size)
        self.fc5 = nn.Linear(self.layer_size, params.latent_dim)

    def latent_normal(self, mu, sigma):
        dist = Normal(loc=mu, scale=sigma)
        return dist

    def forward(self, x):
        """
        x: (bs, input_size) -> input_size: goal (vision only) + current (visuo-proprio)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = self.fc5(x)
        sigma = F.softplus(self.fc5(x+self.epsilon))
        dist = self.latent_normal(mu, sigma)

        return dist


class Actor(nn.Module):
  def __init__(self, layer_size=1024, epsilon=1e-4, 
                num_distribs=None, qbits=None, return_state=False, 
                discrete=False, disc_embed_size=64):
    super(Actor, self).__init__()
    self.return_state = return_state
    self.discrete = discrete
    self.num_distribs = num_distribs
    self.epsilon = epsilon
    self.qbits = qbits

    self.disc_embed = nn.Sequential(
            nn.Linear(params.latent_dim, disc_embed_size),
            nn.ReLU()
        )

    self.lstm1 = nn.LSTM(input_size=params.vision_embedding_dim + params.latent_dim + params.vision_embedding_dim,
                          hidden_size=layer_size, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=layer_size, hidden_size=layer_size, batch_first=True)

    self.alpha = nn.Linear(layer_size, params.action_dim * num_distribs)
    self.mu = nn.Linear(layer_size, params.action_dim * num_distribs)
    self.sigma = nn.Linear(layer_size, params.action_dim * num_distribs)

  def logistic_mixture(self, inputs, qbits=None):
      """
      :param inputs:
      :param qbits: number of quantisation bits, total quantisation intervals = 2 ** qbits
      :return:
      """
      weightings, mu, scale = inputs
      dist = tfd.Logistic(loc=mu, scale=scale)
      dist = tfd.QuantizedDistribution(
          distribution=tfd.TransformedDistribution(
              distribution=dist,
              bijector=tfb.Shift(shift=-0.5)),
          low=-2 ** qbits / 2.,
          high=2 ** qbits / 2.,
      )

      mixture_dist = tfd.MixtureSameFamily(
              mixture_distribution=tfd.Categorical(logits=weightings),
              components_distribution=dist,
              validate_args=True
      )

      action_limits = tf.constant([1.5, 1.5, 2.2, 3.2, 3.2, 3.2, 1.1])
      mixture_dist = tfd.TransformedDistribution(
          distribution=mixture_dist,
          bijector=tfb.Scale(scale=action_limits / (2 ** qbits / 2.)) # scale to action limits
      )

      return mixture_dist
  
  def forward(self, o, z, g):

    z = self.disc_embed(z)
    x = torch.cat([o, z, g], dim=-1)

    x = self.lstm1(x)
    x = self.lstm2(x)

    weightings = self.alpha(x).view(-1, params.action_dim, self.num_distribs)
    mu = self.mu(x).view(-1, params.action_dim, self.num_distribs)
    scale = torch.nn.functional.softplus(self.sigma(x + self.epsilon)).view(-1, params.action_dim, self.num_distribs)
    actions = tfpl.DistributionLambda(self.logistic_mixture, name='logistic_mix')([weightings, mu, scale], self.qbits)

    return actions

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.in_features = params.vision_embedding_dim + params.proprioception_dim + params.action_dim
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.in_features, self.layer_size)
        self.fc2 = nn.Linear(self.layer_size, self.layer_size)
        self.fc3 = nn.Linear(self.layer_size, self.layer_size)
        self.fc4 = nn.Linear(self.layer_size, self.layer_size)
        self.fc5 = nn.Linear(self.layer_size, 1)


    def forward(self, vision_embed, proprioception, action):

        if not isinstance(vision_embed, torch.Tensor):
            vision_embed = torch.tensor(vision_embed, dtype=torch.float32)
        if not isinstance(proprioception, torch.Tensor):
            proprioception = torch.tensor(proprioception, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        x = torch.cat([vision_embed, proprioception, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = F.relu(self.fc4(x))

        return q_value
    

        

 

