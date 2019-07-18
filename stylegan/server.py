import sys

import numpy as np
import torch

from stylegan import SimpleGenerator
from stylegan.utils import read_stream, write_stream


def decode_message(message):
    vec_data = message[:-1]
    latent_vec = torch.from_numpy(np.frombuffer(vec_data, dtype=np.float32))
    
    mode_byte = message[-1]
    is_h_mode = mode_byte == 49  # 49 = 1, 48 = 0
    
    return latent_vec, is_h_mode


img_byte_size = 3 * 256 * 256 * 4
generator = SimpleGenerator()

while True:
    print('[Server] Ready', file=sys.stderr)
    input_bytes = read_stream(sys.stdin.buffer, 2049)
    latent_vec, is_h_mode = decode_message(input_bytes)
    print('[Server] I has read', file=sys.stderr)
    im = generator.generate(latent_vec.unsqueeze(0), haunted=is_h_mode)
    bytes = im.cpu().numpy().tobytes()
    write_stream(sys.stdout.buffer, bytes, check_len=img_byte_size)
    print('[Server] I has written', file=sys.stderr)
