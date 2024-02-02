import torch
import torch.distributions.kl as kl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def image_process(image):
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    image = image.unsqueeze(1)
    image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
    return image

def compute_loss(labels, predictions, mask, seq_lens):
    nll = -predictions.log_prob(labels).sum(dim=2)
    # Apply the mask
    masked_nll = nll * mask
    # Sum over the time dimension and divide by sequence lengths
    per_example_loss = torch.sum(masked_nll, dim=1) / seq_lens
    # Compute the average loss across the batch
    average_loss = torch.mean(per_example_loss)

    return average_loss

def compute_regularisation_loss(recognition, proposal):
    # Reverse KL(enc|plan): we want recognition to map to proposal 
    reg_loss = kl.kl_divergence(recognition, proposal) # + KL(plan, encoding)
    average_loss = reg_loss.mean()
    return average_loss

def plot_latent_space(encoder, vision_network, video_batch, proprioception_batch, action_batch):
    """ Visualize the latent space using t-SNE """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)  # 2D t-SNE, adjust parameters as needed
    batch_size = video_batch.shape[0]
    sequence_length = video_batch.shape[1]
    latent_samples = []
    for i in range(batch_size):
        video_embeded = torch.stack([vision_network(video_batch[i, j, :]) for j in range(sequence_length)], dim=1)
        combined = torch.cat([video_embeded, proprioception_batch[i], action_batch[i]], dim=-1)
        dist = encoder(combined)
        latent = dist.resample()
        latent_samples.append(latent)

    latent_samples = torch.cat(latent_samples, dim=0)
    latent_samples = latent_samples.cpu().numpy()
    tsne_results = tsne.fit_transform(latent_samples)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('Latent Space Represented using t-SNE')
    plt.show()

def save_checkpoint(model, model_name, filename):
    
    checkpoint = torch.load(filename)
    model.save_state_dict(checkpoint[model_name])
    print('Model Saved')

def load_checkpoint(model, model_name, filename):

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[model_name])



