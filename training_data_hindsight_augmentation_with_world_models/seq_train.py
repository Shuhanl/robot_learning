import torch
import time
import numpy as np
from decision_transformer import DecisionTransformer

class SequenceTrainer():

    def __init__(self, state_dim, act_dim, batch_size, pri_buffer, num_trajectories, device):        
        self.batch_size = batch_size
        self.loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((s_hat - s)**2) + torch.mean((a_hat - a)**2) + torch.mean((r_hat - r)**2)
        self.diagnostics = dict()
        self.device = device
        self.num_trajectories = num_trajectories
        self.pri_buffer = pri_buffer
        self.dropout = 0.1
        self.n_head = 1
        self.n_layer = 3
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.warmup_steps = 10000
        self.activation_function = 'relu'
        self.embed_dim = 128
        self.K = 20
        self.max_ep_len = 1000
        self.num_eval_episodes = 100

        self.model =  DecisionTransformer(state_dim=state_dim, act_dim=act_dim, max_length=self.K, max_ep_len=self.max_ep_len, hidden_size=self.embed_dim,
            n_layer=self.n_layer, n_head=self.n_head, n_inner=self.embed_dim, activation_function=self.activation_function,
            n_positions=1024, resid_pdrop=self.dropout, attn_pdrop=self.dropout)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps+1)/self.warmup_steps, 1)
    )

        self.start_time = time.time()

    def get_batch(self, batch_size=256, max_len=K):

        batch_inds = self.pri_buffer.sample_indices(self.num_trajectories)

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rew'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['obs'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['act'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rew'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rew'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
    
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
        

