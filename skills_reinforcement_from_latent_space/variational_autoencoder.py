import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ModifiedCNN(nn.Module):
    def __init__(self, img_height=128, img_width=128, img_channels=3, embedding_size=64, is_decoder=False):
        super(ModifiedCNN, self).__init__()
        self.is_decoder = is_decoder

        # Encoder Layers
        if not is_decoder:
            # In PyTorch, normalization is usually done as a preprocessing step, but it can be included in the model
            self.rescaling = lambda x: x / 255.0  
            self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4, padding=2)  # 'same' padding in PyTorch needs calculation
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(64 * (img_height // 8) * (img_width // 8), 512)
            self.dense2 = nn.Linear(512, embedding_size)

        # Decoder Layers
        if is_decoder:
            self.dense1_rev = nn.Linear(embedding_size, 512)
            self.dense2_rev = nn.Linear(512, 64 * (img_height // 8) * (img_width // 8))
            self.reshape = lambda x: x.view(-1, 64, img_height // 8, img_width // 8)
            self.conv3_rev = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv2_rev = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.conv1_rev = nn.ConvTranspose2d(64, 32, kernel_size=8, stride=4, padding=2)
            self.rescale_back = lambda x: x * 255

    def forward(self, x):
        x = self.rescaling(x)

        if not self.is_decoder:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.flatten(x)
            x = F.relu(self.dense1(x))
            x = self.dense2(x)

        if self.is_decoder:
            x = F.relu(self.dense1_rev(x))
            x = F.relu(self.dense2_rev(x))
            x = self.reshape(x)
            x = F.relu(self.conv3_rev(x))
            x = F.relu(self.conv2_rev(x))
            x = torch.sigmoid(self.conv1_rev(x))
            x = self.rescale_back(x)

        return x

        
class VAE():
    def __init__(self, image_shape, proprioception_dim, sequence_length):
        self.layer_size=2048
        self.latent_dim=256
        self.epsilon=1e-4
        self.sequence_length = sequence_length
        self.image_shape = image_shape
        self.proprioception_dim = proprioception_dim

        # Encoder Layers
        self.cnn_encoder = ModifiedCNN(img_height=image_shape[0], img_width=image_shape[1], img_channels=image_shape[2])
        self.lstm1 = nn.LSTM(image_shape[0] + proprioception_dim, self.layer_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * self.layer_size, self.layer_size, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * self.layer_size, self.latent_dim)
        self.sigma = nn.Linear(2 * self.layer_size, self.latent_dim)

        # Decoder Layers
        self.cnn_decoder = ModifiedCNN(img_height=image_shape[0], img_width=image_shape[1], img_channels=image_shape[2], is_decoder=True)
        self.decode_dense1 = nn.Linear(self.latent_dim, self.layer_size)
        self.decode_lstm1 = nn.LSTM(self.layer_size, self.layer_size, batch_first=True, bidirectional=True)
        self.decode_lstm2 = nn.LSTM(2 * self.layer_size, self.layer_size, batch_first=True, bidirectional=True)
        self.decode_proprioception = nn.Linear(2 * self.layer_size, proprioception_dim)


    def latent_normal(self, mu, sigma):
        dist = Normal(loc=mu, scale=sigma)
        return dist

    def encode(self, images, proprioception):

        # Image encoder
        cnn_out = torch.stack([self.cnn_encoder(images[:, i]) for i in range(self.sequence_length)], dim=1)
        # Combine CNN output with proprioception data
        combined = torch.cat([cnn_out, proprioception], dim=-1)

        # LSTM Layers
        encoded, _ = self.lstm1(combined)
        encoded, _ = self.lstm2(encoded)

        # Latent variable
        mu = self.mu(encoded)
        sigma = F.softplus(self.sigma(encoded+self.epsilon))
        dist = self.latent_normal(mu, sigma)

        return dist
    
    def decode(self, latent):

        # Dense and Repeat
        decoded = F.relu(self.decode_dense1(latent))
        decoded = decoded.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # LSTM Decoding
        decoded, _ = self.decode_lstm1(decoded)
        decoded, _ = self.decode_lstm2(decoded)

        # CNN Decoding and Proprioception Decoding
        image_decoded = torch.stack([self.cnn_decoder(decoded[:, i]) for i in range(self.sequence_length)], dim=1)
        proprioception_decoded = self.decode_proprioception(decoded)

        return image_decoded, proprioception_decoded

        

 

