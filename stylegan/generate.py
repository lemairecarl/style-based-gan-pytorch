import time
from collections import OrderedDict

import torch
from torchvision import utils

from model import StyledGenerator


generate_mixing = False

device = 'cpu'
generator = StyledGenerator(512).to(device)
# generator.load_state_dict(torch.load('checkpoint/style-gan-256-140k.model', map_location=device))
sd = torch.load('checkpoint/style-gan-256-140k.model', map_location=device)
new_sd = OrderedDict()
for k, v in sd.items():
    if 'weight_orig' in k:
        k = k.replace('weight_orig', 'weight')
        fan_in = v.size(1) * v[0][0].numel()
        v *= torch.sqrt(torch.tensor(2 / fan_in))
    new_sd[k] = v
del sd
generator.load_state_dict(new_sd)

mean_style = None

step = 6

shape = 4 * 2 ** step

mean_steps = 1
for i in range(mean_steps):
    style = generator.mean_style(torch.randn(10, 512).to(device))

    if mean_style is None:
        mean_style = style

    else:
        mean_style += style

mean_style /= mean_steps

n_samples = 20
image = generator(
    torch.randn(n_samples, 512).to(device),
    step=step,
    alpha=1,
    mean_style=mean_style,
    style_weight=-3,
)

utils.save_image(image, 'output/sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(-1, 1))

if generate_mixing:
    for j in range(20):
        source_code = torch.randn(9, 512).to(device)
        target_code = torch.randn(5, 512).to(device)
    
        images = [torch.ones(1, 3, shape, shape).to(device) * -1]
    
        source_image = generator(
            source_code, step=step, alpha=1, mean_style=mean_style, style_weight=0.7
        )
        target_image = generator(
            target_code, step=step, alpha=1, mean_style=mean_style, style_weight=0.7
        )
    
        images.append(source_image)
    
        for i in range(5):
            image = generator(
                [target_code[i].unsqueeze(0).repeat(9, 1), source_code],
                step=step,
                alpha=1,
                mean_style=mean_style,
                style_weight=0.7,
                mixing_range=(0, 1),
            )
            images.append(target_image[i].unsqueeze(0))
            images.append(image)
    
        # print([i.shape for i in images])
    
        images = torch.cat(images, 0)
    
        utils.save_image(
            images, f'sample_mixing_{j}.png', nrow=10, normalize=True, range=(-1, 1)
        )
