# Import Python packages
import numpy as np
import os
from io import StringIO
from tqdm import tqdm
from random import random
from PIL import ImageFont, ImageDraw, ImageEnhance
from scipy.stats import truncnorm
from google.colab import files
import IPython.display
import tensorflow as tf
import tensorflow_hub as hub

print(f'Successfully imported packages.')

# Load BigGAN from the official repo (Coursera: remove and load pkl file)

# tf.reset_default_graph()
module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
print('Loading BigGAN module from:', module_path)
module = hub.Module(module_path)
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print('Loaded the BigGAN module. Here are its input and outputs sizes:')
print('Inputs:\n', '\n'.join(
    '  {}: {}'.format(*kv) for kv in inputs.items()))
print('\nOutput:', output)

# Inputs:
#    truncation: Tensor("truncation:0", shape=(), dtype=float32)
#   z: Tensor("z:0", shape=(?, 128), dtype=float32)
#   y: Tensor("y:0", shape=(?, 1000), dtype=float32)

# Output: Tensor("module_apply_default/G_trunc_output:0", shape=(?, 256, 256, 3), dtype=float32)

# Get the different components of the input
noise_vector = input_z = inputs['z']
label = input_y = inputs['y']
input_trunc = inputs['truncation']

# Get the sizes of the noise vector and the label
noise_vector_size = input_z.shape.as_list()[1]
label_size = input_y.shape.as_list()[1]

print(f'Components of input are set.')
print(f'Noise vector is size {noise_vector_size}. Label is size {label_size}.')

# Components of input are set.
# Noise vector is size 128. Label is size 1000.

# Function to truncate the noise vector
def truncated_noise_vector(batch_size, truncation=1., seed=42):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, noise_vector_size), random_state=state)
  return truncation * values

print(f'Function declared.')

def one_hot(label, label_size=label_size):
  '''
  Function to turn label into a one-hot vector.
  This means that all values in the vector are 0, except one value that is 1, 
  which represents the class label, e.g. [0 0 0 0 1 0 0].
  '''
  label = np.asarray(label)
  if len(label.shape) <= 1:
    index = label
    index = np.asarray(index)
    if len(index.shape) == 0:
      index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    label = np.zeros((num, label_size), dtype=np.float32)
    label[np.arange(num), index] = 1
  assert len(label.shape) == 2
  return label

print(f'Function declared.')

def sample(sess, noise, label, truncation=1., batch_size=8,
           label_size=label_size):
  '''
  Function to sample images from the model.
  Inputs include the noise vector, label, truncation, 
  and batch size (number of images to generate).
  '''
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot(label, label_size)
  ims = []
  print(f"Generating images...")
  for batch_start in tqdm(range(0, num, batch_size)):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims.append(sess.run(output, feed_dict=feed_dict))
  ims = np.concatenate(ims, axis=0)
  assert ims.shape[0] == num
  ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
  ims = np.uint8(ims)
  return ims

print(f'Function declared.')

'''
Functions for saving and visualizing images in a grid.
'''
def imgrid(imarray, cols=5, pad=1):
  if imarray.dtype != np.uint8:
    raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = int(np.ceil(N / float(cols)))
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  path = 'results/biggan-example.png'
  img = PIL.Image.fromarray(a)
  img.save(path, format)
  try:
    disp = IPython.display.display(IPython.display.Image(path))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

print(f'Functions declared.')

# Initialize TensorFlow
initializer = tf.global_variables_initializer()
sess = tf.Session()
sess.run(initializer)

print('TensorFlow initialized.')

#@title Select the class and truncation { display-mode: "form", run: "auto" }
#@markdown ##### The id next to each class is taken from ImageNet, a 1000-class dataset of that BigGAN was trained on.
#@markdown ##### Double click to see all values in a code format.

# @param ["0) tench, Tinca tinca", "1) goldfish, Carassius auratus",... "999) toilet tissue, toilet paper, bathroom tissue"]
Truncation = 1 #@param {type:"slider", min:0.02, max:1, step:0.02}

# Set number of samples
num_samples = 4

# Create the noise vector with truncation (you'll learn about this later!)
noise_vector = truncated_noise_vector(num_samples, Truncation)

# Select the class to generate
label = int(Class.split(')')[0])

# Sample the images with the noise vector and label as inputs
ims = sample(sess, noise_vector, label, truncation=Truncation)

# Display generated images
imshow(imgrid(ims, cols=min(num_samples, 5)))