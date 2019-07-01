import time

import numpy as np
import torch
from torchvision import utils

from model import StyledGenerator


generate_mixing = False

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


# while True:
#     a = input()
#     im = generate_image()
#     im = im[:, :4, :4]
#     print(im.cpu().numpy().tobytes())

image = generate_image()
orig_shape = image.size()
print(orig_shape)
expected_len = np.prod(orig_shape) * 4
bytes = image.cpu().numpy().tobytes()
assert len(bytes) == expected_len
image = np.frombuffer(bytes, dtype=np.float32).reshape(orig_shape)
image = torch.from_numpy(image)

utils.save_image(image, 'sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(-1, 1))
