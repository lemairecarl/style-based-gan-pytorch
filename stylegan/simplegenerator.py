import os
import time
from collections import OrderedDict

import torch
import torch.jit
from torchvision import utils

from stylegan.model import StyledGenerator

_model_path = 'stylegan/checkpoint/style-gan-256-140k.model'


class SimpleGenerator:
    def __init__(self, model_file=None):
        if model_file is None:
            if os.path.isfile(_model_path):
                model_file = _model_path
            else:
                model_file = os.environ['STYLEGAN_MODEL']
        
        self.device = 'cpu'
        self.generator = StyledGenerator(512).to(self.device)
        self.generator.eval()
        
        # Fix and load state dict
        sd = torch.load(model_file, map_location=self.device)
        new_sd = OrderedDict()
        for k, v in sd.items():
            if 'weight_orig' in k:
                k = k.replace('weight_orig', 'weight')
                fan_in = v.size(1) * v[0][0].numel()
                v *= torch.sqrt(torch.tensor(2 / fan_in))
            new_sd[k] = v
        del sd
        self.generator.load_state_dict(new_sd)
        
        # Trace
        self.traced_model = torch.jit.trace(self.generator, torch.randn(1, 512).to(self.device), check_trace=False)
        del self.generator
    
    def generate(self, latent_vec):
        image = self.traced_model(
            latent_vec.unsqueeze(0).to(self.device)
        )
        # Fit range into [0, 1]
        image.clamp_(-1, 1)
        image = (image + 1.0) / 2.0
        # Remove batch dim
        return image.squeeze(0)


def save_image(image):
    utils.save_image(image, 'sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(0, 1))
