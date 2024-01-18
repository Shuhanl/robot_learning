import torch
import torch.nn.functional as F

def image_process(image):
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    image = image.unsqueeze(1)
    image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
    return image


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

