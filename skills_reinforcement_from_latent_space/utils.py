import torch
import torch.nn.functional as F

def image_process(image):
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    image = image.unsqueeze(1)
    image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
    return image

