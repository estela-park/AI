# Version Issue: conda activate TF114
#                cd D:\study\GAN

import tensorflow as tf
print(tf.__version__)
# 1.15.0

# Import needed Python libraries
import os
import pickle
# 4.0
import warnings
import numpy as np
import PIL

# PIL: python image libarary
#      installed with pip install image
#      processes varias types of image .ext
# module wrapper helps with deprecation warnning and global renaming attr. and methods

from tensorflow.python.util import module_wrapper
module_wrapper._PER_MODULE_WARNING_LIMIT = 0

# Import the official StyleGAN repo
import dnnlib
from dnnlib import tflib
from dnnlib import util
import config

# Initialize TensorFlow
tflib.init_tf()

print(os.getcwd())
# D:\study\GAN

import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

import io

# Move into the StyleGAN directory, if you're not in it already
path = 'D:/study/GAN/stylegan/'
if "stylegan" not in os.getcwd():
    # chdir() changes the current working directory to the given path.
    # It returns None in all the cases.
    os.chdir(path)
    # @config.py cache_dir = 'cache'
print(os.getcwd())

# when path = './GAN/stylegan/'
#   D:\study\GAN
# when path = 'D:/study/GAN/stylegan'
#   D:\study\GAN\stylegan


# Load pre-trained StyleGAN network
url = 'https://bitbucket.org/ezelikman/gans/downloads/karras2019stylegan-ffhq-1024x1024.pkl' 
      # karras2019stylegan-ffhq-1024x1024.pkl

# with util.open_url(url, cache_dir=config.cache_dir) as f:
  # You'll load 3 components, and use the last one Gs for sampling images.
  #   _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
  #   _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
  #   Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
  # f: <class '_io.BufferedReader'>
    #_G, _D, Gs = pickle.load(f)#, encoding='byte')


with open('D:\study\GAN\stylegan\cache\d2a38577d3c883e2bc684fb7d32c3bab_karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as file: 
    # type(file): <class '_io.BufferedReader'>
    # file.readline(): b'\xef\xbf\xbd\x04...': <class 'bytes'>
    # file.readlines(): <class 'list'>
    file = file.readlines()
    print('=====================Read file then decoding=================')
    for line in range(5):
        # print(file[line].decode('utf-8'))
        # print(file[line].decode('latin1'))
        # print(file[line].decode('cp1252'))
        print(file[line].decode('cp949'))
        print(file[line])
        # print(file[line].decode('utf-16'))
        # print(file[line].decode('utf-32'))
        # print(file[line].decode('ascii'))
        # print(file[line].decode('MacRoman'))
        # if type(line) != "<class 'bytes'>":
        #     print(type(line))
    _G, _D, Gs = pickle.load(file, encoding='utf-8')

print('StyleGAN package loaded successfully!')

# Truncation trades off fidelity (quality) and diversity of the generated images - play with it!
# @title Generate faces with StyleGAN
# @markdown Double click here to see the code. After setting truncation, run the cells below to generate images. This adjusts the truncation, you will learn more about this soon! Truncation trades off fidelity (quality) and diversity of the generated images - play with it!
# truncation==0.1 same facial features with different skin color
# truncation==1 different face with different angle and everything, but quiet distorted
Truncation = 0.5 #@param {type:"slider", min:0.1, max:1, step:0.1}

print(f'Truncation set to {Truncation}. \nNow run the cells below to generate images with this truncation value.')

# Set the random state. Nothing special about 42,
#   except that it's the meaning of life.
rnd = np.random.RandomState(42)

print(f'Random state is set.')

# batch_size means how many images for the run

batch_size = 4 #@param {type:"slider", min:1, max:10, step:1}

print(f'Batch size is {batch_size}...')

# setting noise vector
input_shape = Gs.input_shape[1]
noise_vectors = rnd.randn(batch_size, input_shape)

print(f'There are {noise_vectors.shape[0]} noise vectors, each with {noise_vectors.shape[1]} random values between -{Truncation} and {Truncation}.')

# image generation: Gs seems like a generator from GAN network
fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
images = Gs.run(noise_vectors, None, truncation_psi=Truncation, randomize_noise=False, output_transform=fmt)

print(f'Successfully sampled {batch_size} images from the model.')

# visualizing serialized image

# Save the images
os.makedirs(config.result_dir, exist_ok=True)
png_filename = os.path.join(config.result_dir, 'stylegan-example.png')
if batch_size > 1:
  img = np.concatenate(images, axis=1)
else:
  img = images[0]
PIL.Image.fromarray(img, 'RGB').save(png_filename)

# Check the images out!
from IPython.display import Image
Image(png_filename, width=256*batch_size, height=256)