import torch
import torch.nn as nn
from mamba import MambaModel
import parameters as params


class DecisionMamba(nn.Module):

    """
    This model uses Mamba to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self):
        super().__init__()
        self.mamba_model_dim = params.mamba_model_dim
        self.action_dim = params.action_dim
        self.state_dim = params.vision_embedding_dim + params.proprioception_dim
        self.device = params.device
        self.embedding_dim = params.embedding_dim


        self.mamba = MambaModel(self.embedding_dim, self.mamba_model_dim)

        self.embed_timestep = nn.Embedding(self.embedding_dim)
        self.embed_return = torch.nn.Linear(1, self.embedding_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, self.embedding_dim)
        self.embed_action = torch.nn.Linear(self.action_dim, self.embedding_dim)

        self.embed_ln = nn.LayerNorm(self.embedding_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.mamba_model_dim, self.state_dim)
        self.predict_action = nn.Sequential(nn.Linear(self.mamba_model_dim, self.action_dim), nn.Tanh())
        self.predict_return = torch.nn.Linear(self.mamba_model_dim, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps):

        batch_size, seq_length = states.shape[0], states.shape[1]

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
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.embedding_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        mamba_outputs = self.mamba(stacked_inputs)
        x = mamba_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for states_t  
        x = x.reshape(batch_size, seq_length, 3, self.embedding_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,0])  # predict next return given state and action
        state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        action_preds = self.predict_action(x[:,2])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            # truncate sequences to max length
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
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


        state_preds, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, **kwargs)

        return action_preds[0,-1]
