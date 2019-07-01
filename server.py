import time
import sys

import numpy as np
import torch
from torchvision import utils

from model import StyledGenerator


img_byte_size = 3 * 256 * 256 * 4

device = 'cuda'
generator = StyledGenerator(512).to(device)
generator.load_state_dict(torch.load('checkpoint/style-gan-256-140k.model', map_location=device))

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


def generate_image():
    with torch.no_grad():
        image = generator(
            torch.randn(1, 512).to(device),
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=-3,
        )
    return image.squeeze(0)


while True:
    print('[Server] Ready', file=sys.stderr)
    a = sys.stdin.buffer.read(1)
    print('[Server] I has read', file=sys.stderr)
    im = generate_image()
    bytes = im.cpu().numpy().tobytes()
    assert len(bytes) == img_byte_size
    sys.stdout.buffer.write(bytes)
    print('[Server] I has written', file=sys.stderr)
