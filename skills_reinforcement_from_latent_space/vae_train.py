import torch
import torch.nn.functional as F
import torch.optim as optim
from variational_autoencoder import VAE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class VAETrainer():
    def __init__(self, image_shape, robot_state_shape, sequence_length):
        self.image_shape = image_shape
        self.robot_state_shape = robot_state_shape
        self.sequence_length = sequence_length
        self.visualize_batch = 100
        self.vae_model = VAE(self.image_shape, self.robot_state_shape, self.sequence_length)
        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=0.001)  # Learning rate can be adjusted as needed
        self.tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)  # 2D t-SNE, adjust parameters as needed
    
    def train_step(self, images, proprioception):
        dist = self.vae_model.encode(images, proprioception)
        latent = dist.resample()
        image_decoded, proprioception_decoded = self.vae_model.decode(latent)
        reconstruction_loss = F.mse_loss(image_decoded, images) + F.mse_loss(proprioception_decoded, proprioception)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + dist.scale**2 - dist.loc**2 - torch.exp(dist.scale**2), dim=1), dim=0)
        loss = reconstruction_loss + kl_loss

        # Backpropagation
        self.optimizer.zero_grad()  # Clear existing gradients
        loss.backward()        # Calculate gradients
        self.optimizer.step()  # Update parameters

        return loss.item()

    def plot_latent_space(self, images, proprioception):
        """ Visualize the latent space using t-SNE """
        latent_samples = []
        for i in range(self.visualize_batch):
            dist = self.vae_model.encode(images[i], proprioception[i])
            latent = dist.resample()
            latent_samples.append(latent)
        
        latent_samples = torch.cat(latent_samples, dim=0)
        latent_samples = latent_samples.cpu().numpy()
        tsne_results = self.tsne.fit_transform(latent_samples)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.title('VAE Latent Space Represented using t-SNE')
        plt.show()

 

