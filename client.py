import subprocess
import time

import numpy as np
import torch
from torchvision import utils

img_size = 256
img_byte_size = 3 * img_size * img_size * 4

p = subprocess.Popen(['python', 'server.py'], bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

while True:
    a = input('Press key...')
    print('Sending byte')
    t0 = time.time()
    p.stdin.write(b'0')

    # print('Reading data')
    data = p.stdout.read(img_byte_size)
    # print('Received:', data[:10])

    image = np.frombuffer(data, dtype=np.float32).reshape((3, img_size, img_size))
    image = torch.from_numpy(image)
    t1 = time.time()
    print('Took', t1 - t0)

    utils.save_image(image, 'sample_{}.png'.format(time.time()), nrow=10, normalize=True, range=(-1, 1))
