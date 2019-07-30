import os
import subprocess

import numpy as np
import torch
import paramiko

from stylegan.utils import read_stream, write_stream


class GeneratorClient:
    img_size = 256
    img_byte_size = 3 * img_size * img_size * 4
    
    def __init__(self):
        #server_script_path = os.environ.get('CESTUNVRAIREGAL', '/Users/carl/recherche/style-based-gan-pytorch/stylegan/server.py')
        #self.server = subprocess.Popen(['python', server_script_path],
        #                               bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.client = paramiko.client.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        self.client.connect('lemc2220-desktop.ccs.usherbrooke.ca', port=22, username='carl', password=os.environ['DANGSON'],
                       look_for_keys=False)
        self.client.invoke_shell()
        self.stdin, self.stdout, self.stderr = self.client.exec_command('source venv/bin/activate; STYLEGAN_MODEL=/home/carl/source/_perso/style-based-gan-pytorch/stylegan/checkpoint/style-gan-256-140k.model python /home/carl/source/_perso/style-based-gan-pytorch/stylegan/server.py')

    def generate(self, latent_vec, is_h_mode=True):
        # stdin = self.server.stdin
        # stdout = self.server.stdout
        stdin = self.stdin
        stdout = self.stdout
        
        message = latent_vec.cpu().numpy().tobytes() + str(int(is_h_mode)).encode()
        write_stream(stdin, message, check_len=2049)

        data = read_stream(stdout, self.img_byte_size)
    
        image = np.frombuffer(data, dtype=np.float32).reshape((3, self.img_size, self.img_size))
        image = torch.from_numpy(image)
        return image

    def __del__(self):
        # self.client.send('\x03')
        self.client.close()
