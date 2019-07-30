import os
import subprocess

import numpy as np
import torch

from stylegan.utils import read_stream, write_stream


class GeneratorClient:
    img_size = 256
    img_byte_size = 3 * img_size * img_size * 4
    
    def __init__(self):
        server_script_path = os.environ.get('CESTUNVRAIREGAL', '/Users/carl/recherche/style-based-gan-pytorch/stylegan/server.py')
        self.server = subprocess.Popen(['python', server_script_path],
                                       bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def generate(self, latent_vec, is_h_mode=True):
        message = latent_vec.cpu().numpy().tobytes() + str(int(is_h_mode)).encode()
        write_stream(self.server.stdin, message, check_len=2049)

        data = read_stream(self.server.stdout, self.img_byte_size)
    
        image = np.frombuffer(data, dtype=np.float32).reshape((3, self.img_size, self.img_size))
        image = torch.from_numpy(image)
        return image


