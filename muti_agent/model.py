import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as params
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class VisionNetwork(nn.Module):
    def __init__(self, ):
        super(VisionNetwork, self).__init__()

        img_channels=params.vision_dim[0]
        img_height=params.vision_dim[1]
        img_width=params.vision_dim[2]
        self.device = params.device
        self.out_dim = params.vision_embedding_dim

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
        spatial_soft_argmax = torch.sum(softmax * image_coords, dim=[2, 3])  # [N, C, 2]
        x = nn.Flatten()(spatial_soft_argmax)  # [N, C, 2] -> [N, 2*C]
        return x

    def forward(self, x):
        pre_softmax = self.cnn(x)
        spatial_soft_argmax = self._spatial_softmax(pre_softmax)
        x = self.fc(spatial_soft_argmax)
        return x
        
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None, max_ep_len=4096, action_tanh=True, nhead=8, num_encoder_layers=6):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # Transformer Encoder configuration
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer_encoder(stacked_inputs, src_key_padding_mask=stacked_attention_mask==0)

        x = x.reshape(batch_size, seq_length, self.hidden_size)

        # get predictions
        action_preds = self.predict_action(x)  # predict next action given state

        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds
    

class Critic(nn.Module):
    def __init__(self, layer_size=1024):
        super(Critic, self).__init__()
        self.in_features = params.vision_embedding_dim + params.proprioception_dim + params.action_dim
        # Fully Connected Layers
        self.fc = nn.Sequential(nn.Linear(self.in_features, layer_size), 
                                nn.ReLU(),
                                nn.Linear(layer_size, layer_size),
                                nn.ReLU(),
                                nn.Linear(layer_size, layer_size),
                                nn.ReLU(),
                                nn.Linear(layer_size, 1), 
                                nn.ReLU())

    def forward(self, vision_embed, proprioception, action):

        x = torch.cat([vision_embed, proprioception, action], dim=-1)
        q_value = self.fc(x)

        return q_value
    

        

 

