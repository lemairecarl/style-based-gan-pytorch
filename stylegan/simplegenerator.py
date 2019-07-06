import os
import time
from collections import OrderedDict

import torch
import torch.jit
from torchvision import utils

from stylegan.model import StyledGenerator

_weights_path = 'stylegan/checkpoint/style-gan-256-140k.model'
_traced_model_path = 'stylegan/checkpoint/trace.pt'
_traced_model_path_dir = 'stylegan/checkpoint'


class SimpleGenerator:
    def __init__(self, model_file=None):
        self.device = 'cpu'
        self.model = None
        
        if model_file is None:
            if os.path.isfile(_weights_path):
                model_file = _weights_path
            else:
                model_file = os.environ['STYLEGAN_MODEL']
        
        generator = StyledGenerator(512).to(self.device)
        generator.eval()
        
        # Fix and load state dict TODO cache
        sd = torch.load(model_file, map_location=self.device)
        new_sd = OrderedDict()
        for k, v in sd.items():
            if 'weight_orig' in k:
                k = k.replace('weight_orig', 'weight')
                fan_in = v.size(1) * v[0][0].numel()
                v *= torch.sqrt(torch.tensor(2 / fan_in))
            new_sd[k] = v
        del sd
        generator.load_state_dict(new_sd)
        
        self.model = generator
    
    def generate(self, latent_vecs):
        images = self.model(
            latent_vecs.to(self.device)
        )
        # Fit range into [0, 1]
        images.clamp_(-1, 1)
        images = (images + 1.0) / 2.0
        # Remove batch dim
        return images


def save_image(image):
    utils.save_image(image, 'sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(0, 1))