import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

