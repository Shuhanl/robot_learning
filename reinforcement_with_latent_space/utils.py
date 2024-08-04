import torch
import torch.distributions.kl as kl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.common import flatten_state_dict
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np

class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = self.load_h5_data(trajectory)

            # convert the original raw observation with our batch-aware function
            obs = self.convert_observation(trajectory["obs"])
            # we use :-1 to ignore the last obs as terminal observations are included
            # and they don't have actions
            self.obs_rgbd.append(obs['rgbd'][:-1])
            self.obs_state.append(obs['state'][:-1])
            self.actions.append(trajectory["actions"])

    # loads h5 data into memory for faster access
    def load_h5_data(self, data):
        out = dict()
        for k in data.keys():
            if isinstance(data[k], h5py.Dataset):
                out[k] = data[k][:]
            else:
                out[k] = self.load_h5_data(data[k])
        return out

    def convert_observation(self, observation):
        # flattens the original observation by flattening the state dictionaries
        # and combining the rgb and depth images

        # image data is not scaled here and is kept as uint16 to save space
        image_obs = observation["sensor_data"]
        rgb = image_obs["base_camera"]["rgb"]
        depth = image_obs["base_camera"]["depth"]
        tcp = observation["extra"]["tcp_pose"]

        # combine the RGB and depth images
        rgbd = np.concatenate([rgb, depth], axis=-1)
        obs = dict(rgbd=rgbd, state=tcp)
        return obs

    def rescale_rgbd(self, rgbd):
        # rescales rgbd data and changes them to floats
        rgb = rgbd[..., 0:3] / 255.0
        depth = rgbd[..., 3:4] / (2**10)
        return np.concatenate([rgb, depth], axis=-1)

    def __len__(self):
        return len(self.obs_rgbd)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float()

        rgbd = self.obs_rgbd[idx]
        rgbd = self.rescale_rgbd(rgbd)
        rgbd = torch.from_numpy(rgbd).squeeze(1).permute(0, 3, 1, 2).float()

        state = self.obs_state[idx]
        state = torch.from_numpy(state).squeeze(1)
        state = torch.FloatTensor(state)

        return dict(rgbd=rgbd, state=state, action=action)
    
def convert_demonstration(data_bacth):

    actions = data_bacth["action"]
    video = data_bacth["rgbd"]
    proprioception = data_bacth['state']

    return actions, video, proprioception

def convert_observation(observation):

    image_obs = observation["sensor_data"]
    rgb = image_obs["base_camera"]["rgb"] / 255.0
    depth = image_obs["base_camera"]["depth"] / (2**10)
    vision = np.concatenate([rgb, depth], axis=-1)
    vision = vision.transpose(0, 3, 1, 2)    # [batch_size, channel, height, width]
    proprioception = observation['extra']['tcp_pose']

    return vision, proprioception

def compute_loss(labels, predictions):
    nll = torch.sum(-predictions.log_prob(labels), dim=1)  # sum over action space
    nll = torch.mean(nll, dim=0)   # average over batch
    return nll

def compute_regularisation_loss(recognition, proposal):
    """ Reverse KL(enc|plan): we want recognition to map to proposal """
    reg_loss = kl.kl_divergence(recognition, proposal) # [batch_size, latent_dim]
    reg_loss = torch.sum(reg_loss, dim=1) # sum over latent space
    average_loss = torch.mean(reg_loss, dim=0) # average over batch
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
        latent = dist.sample()
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



