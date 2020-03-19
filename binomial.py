import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import get_accuracy

# threshold images
def preprocess_dataset(images, INTENSITY_THRESHOLD=127):
  images_threshold = np.where(images > INTENSITY_THRESHOLD, 1, 0)

  return images_threshold

def get_binomial_parameters(X, labels, NUM_DIGITS=10):
  digit_freq = np.empty((NUM_DIGITS), dtype=np.int32)
  pixel_freq_ones = np.empty((NUM_DIGITS, X.shape[1]),dtype=np.int32)
  
  # compute probability of each digit in train set
  digit_freq = np.bincount(labels)
  idx = np.insert(np.cumsum(digit_freq), 0, 0)
  
  # compute pixel probabilities
  for digit in range(0,NUM_DIGITS):
      pixel_freq_ones[digit,:] = np.sum(X[labels == digit], axis=0)
  
  pixel_freq_zeros = (np.reshape(digit_freq,(digit_freq.size,1)).T - pixel_freq_ones.T).T
  pixel_freq = np.stack((pixel_freq_zeros / digit_freq[:,None], pixel_freq_ones / digit_freq[:,None]), axis=-1)

  return [digit_freq / labels.size, pixel_freq]

def binomialNB(X, p, prior, NUM_DIGITS=10):
  eps = 1e-8
  NUM_TEST, NUM_PIXELS = X.shape[0], X.shape[1]

  posterior = np.empty((NUM_TEST, NUM_DIGITS))
  label_predicted = np.empty((NUM_TEST))

  for idx in range(0, NUM_TEST):
    curr_image = X[idx,:].astype(np.int32)
    inv_curr_image = 1 - curr_image
    
    # compute the likelihood
    image_stack = np.stack((inv_curr_image, curr_image), axis=-1)
    pixel_intensity = np.repeat(image_stack[np.newaxis], NUM_DIGITS, axis=0)
    likelihood = np.sum(np.multiply(p, pixel_intensity), axis=2)
    
    # compute marginal probability
    marginal = np.sum(np.prod(likelihood,axis=1) * prior)
    
    # compute posterior probabilities and get predicted label
    posterior[idx] = np.prod(likelihood,axis=1) * prior / (np.prod(marginal) + eps)
    label_predicted[idx] = np.argmax(posterior[idx])

  return label_predicted

def predictBinomial(X_train, X_test, y_train, y_test, NUM_PLOTS=10, NUM_DIGITS=10):
  # preprocess the dataset
  X_threshold_train = preprocess_dataset(X_train)
  X_threshold_test = preprocess_dataset(X_test)

  # compute parameters
  [prior, binomial_p] = get_binomial_parameters(
      X_threshold_train, y_train, NUM_DIGITS)

  labels_predicted = binomialNB(
    X_threshold_test, binomial_p, prior, NUM_DIGITS)
  accuracy = get_accuracy(y_test, labels_predicted)

  print('Accuracy computed using binarized images is {}'.format(accuracy))
