import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model


def encoder_block(filters, inputs):
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  s = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  p = MaxPooling2D(pool_size = (2,2), padding = 'same')(s)
  return s, p #p provides the input to the next encoder block and s provides the context to the symmetrically opposte decoder block

#Baseline layer is just a binch on Convolutional Layers to extract high level features from the downsampled Image
#As the number of baselines to include is highly related to the task, we don't necessarily need a function for it
def baseline_layer(filters, inputs):
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  return x

def decoder_block(filters, connections, inputs):
  x = Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
  skip_connections = concatenate([x, connections], axis = -1)
  x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skip_connections)
  x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
  return x

def unet():
  #Defining the input layer and specifying the shape of the images
  inputs = Input(shape = (128,128,1))

  #defining the encoder
  s1, p1 = encoder_block(64, inputs = inputs)
  s2, p2 = encoder_block(128, inputs = p1)
  s3, p3 = encoder_block(256, inputs = p2)
  s4, p4 = encoder_block(512, inputs = p3)

  #Setting up the baseline
  baseline = baseline_layer(1024, p4)

  #Defining the entire decoder
  d1 = decoder_block(512, s4, baseline)
  d2 = decoder_block(256, s3, d1)
  d3 = decoder_block(128, s2, d2)
  d4 = decoder_block(64, s1, d3)

  #Setting up the output function for binary classification of pixels
  outputs = Conv2D(1, 1, activation = 'sigmoid')(d4)

  #Finalizing the model
  model = Model(inputs = inputs, outputs = outputs, name = 'Unet')

  return model

def augmentData(dataset: list, masks: list, brightness: float = 0.5,
                contrast: float = 2, hue: float = 200,
                saturation: float = 200, mean: float = 0, std: float = 1) -> tuple[list, list]:
  """
  Augments images in a dataset by flipping horizontally, changing brightness, contrast,
  hue, saturation and adding noise.

  Parameters
      ----------
      dataset : list
          the dataset to augment
      masks : list
          original list of labels
      brightness : float
          brightness coefficient value
      contrast : float
          contrast coefficient value
      hue : float
          the amount of hue increase
      saturation : float
          the amount of saturation increase
      mean : float
          mean value for noise
      std : float
          standard deviation for noise

  Returns
      ----------
      tuple[list, list]
      tuple of augmented data and labels
  """
  data_flip_hor = []
  masks_flip_hor = []
  data_flip_ver = []
  masks_flip_ver = []
  data_90_r = []
  masks_90_r = []
  data_90_l = []
  masks_90_l = []
  data_brightened = []
  data_hsv = []
  data_noisy = []

  for image,mask in zip(dataset,masks):
    data_flip_hor.append(cv2.flip(image, 1))
    masks_flip_hor.append(cv2.flip(mask, 1))

    data_flip_ver.append(cv2.flip(image, 0))
    masks_flip_ver.append(cv2.flip(mask, 0))

    data_90_r.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    masks_90_r.append(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))

    data_90_l.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    masks_90_l.append(cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE))

    data_brightened.append(cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype),
                                           0, brightness))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h += hue
    s += saturation
    hsv = cv2.merge((h, s, v))
    data_hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    data_hsv.append(data_hue)

    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    data_noise = cv2.add(image, noise)
    data_noisy.append(data_noise)

  dataset_augmented = dataset + data_flip_ver + data_90_r + data_90_l + data_brightened + data_hsv + data_noisy
  masks_augmented = masks + masks_flip_ver + masks_90_r + masks_90_l + masks*3

  data_flip_hor.clear()
  data_flip_ver.clear()
  data_90_r.clear()
  data_90_l.clear()


  for image in data_hsv:
    data_flip_hor.append(cv2.flip(image, 1))

    data_flip_ver.append(cv2.flip(image, 0))

    data_90_r.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    data_90_l.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))


  for image in data_noisy:
    data_flip_hor.append(cv2.flip(image, 1))

    data_flip_ver.append(cv2.flip(image, 0))

    data_90_r.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    data_90_l.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

  for image in data_brightened:
    data_flip_hor.append(cv2.flip(image, 1))

    data_flip_ver.append(cv2.flip(image, 0))

    data_90_r.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    data_90_l.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

  dataset_augmented +=  data_flip_ver + data_90_r + data_90_l
  masks_augmented += 3*masks_flip_ver  + 3*masks_90_r + 3*masks_90_l

  return dataset_augmented, masks_augmented


