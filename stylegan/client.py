import subprocess

import numpy as np
import torch


class GeneratorClient:
    img_size = 256
    img_byte_size = 3 * img_size * img_size * 4
    
    def __init__(self):
        self.server = subprocess.Popen(['python', '/Users/carl/recherche/style-based-gan-pytorch/stylegan/server.py'],
                                       bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def generate(self):
        self.server.stdin.write(b'0')
    
        # print('Reading data')
        data = b''
        # n_reads = 0
        while len(data) < self.img_byte_size:
            received_data = self.server.stdout.read(self.img_byte_size)
            # print(len(received_data))
            data += received_data
            # n_reads += 1
        assert len(data) == self.img_byte_size
        # print('Num reads', n_reads)
        # print('Received:', data[:10])
    
        image = np.frombuffer(data, dtype=np.float32).reshape((3, self.img_size, self.img_size))
        image = torch.from_numpy(image)
        # utils.save_image(image, 'output/sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(-1, 1))
        return image
