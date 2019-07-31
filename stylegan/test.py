import stylegan
import stylegan.client
import torch

from stylegan import save_image

g = stylegan.client.GeneratorClient()

p = torch.randn(512)
save_image(g.generate(p, True))
