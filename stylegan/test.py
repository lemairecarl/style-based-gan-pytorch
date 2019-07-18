import stylegan
import torch

from stylegan import save_image

g = stylegan.GeneratorClient()

p = torch.randn(512)
save_image(g.generate(p, True))
