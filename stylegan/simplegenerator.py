import os
import time

import torch
from torchvision import utils

from stylegan.model import StyledGenerator


class SimpleGenerator:
    def __init__(self, model_file=None):
        if model_file is None:
            model_file = os.environ['STYLEGAN_MODEL']  # 'checkpoint/style-gan-256-140k.model'
        
        self.device = 'cpu'
        self.generator = StyledGenerator(512).to(self.device)
        self.generator.load_state_dict(torch.load(model_file, map_location=self.device))
        
        self.mean_style = None
        mean_steps = 1
        for i in range(mean_steps):
            style = self.generator.mean_style(torch.randn(10, 512).to(self.device))
            if self.mean_style is None:
                self.mean_style = style
            
            else:
                self.mean_style += style
        self.mean_style /= mean_steps
    
    def generate(self, latent_vec):
        image = self.generator(
            latent_vec.unsqueeze(0).to(self.device),
            step=6
        )
        # Fit range into [0, 1]
        image.clamp_(-1, 1)
        image = (image + 1.0) / 2.0
        # Remove batch dim
        return image.squeeze(0)


def save_image(image):
    utils.save_image(image, 'sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(-1, 1))
