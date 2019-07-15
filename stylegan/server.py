import sys

import torch

from stylegan import SimpleGenerator

img_byte_size = 3 * 256 * 256 * 4
generator = SimpleGenerator()

while True:
    print('[Server] Ready', file=sys.stderr)
    a = sys.stdin.buffer.read(1)
    print('[Server] I has read', file=sys.stderr)
    im = generator.generate(torch.randn(1, 512))
    bytes = im.cpu().numpy().tobytes()
    assert len(bytes) == img_byte_size
    sys.stdout.buffer.write(bytes)
    print('[Server] I has written', file=sys.stderr)
