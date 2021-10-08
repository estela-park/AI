import urllib
import tensorflow as tf
import pandas as pd
import csv


# series refers to pd.Seires
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dadaset.from_tensor_slices(series)
    ds = ds.window(window_size+1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size+1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'

urllib.request.urlretrieve(url, 'sunspots.csv')

time_step = []
sunspots = []

with open('sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(# YOUR CODE HERE)
        time_step.append(# YOUR CODE HERE)

    series = # YOUR CODE HERE

# DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

