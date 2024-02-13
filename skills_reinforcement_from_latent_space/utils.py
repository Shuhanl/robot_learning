import torch
import torch.nn.functional as F

def save_checkpoint(self, episode):
    checkpoint = {'episode': episode,
    'actor_state_dict': self.agent_trainer.actor.state_dict(),
    'target_actor_state_dict': self.agent_trainer.target_actor.state_dict(),
    'actor_optimizer_state_dict': self.agent_trainer.actor_optimizer.state_dict(),
    'critic_state_dict': self.agent_trainer.critic.state_dict(),
    'target_critic_state_dict': self.agent_trainer.target_critic.state_dict(),
    'critic_optimizer_state_dict': self.agent_trainer.critic_optimizer.state_dict()}

    torch.save(checkpoint, '/content/drive/My Drive/check_point')
    print('Model Saved')

