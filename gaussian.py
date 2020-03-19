import numpy as np
import cv2

from utils import uint2float, performPCA, get_accuracy

def compute_parameters(X, labels, NUM_DIGITS, NUM_PIXELS):
  # compute mean and variance of training set for Gaussian Naive Bayes
  mu = np.zeros((NUM_DIGITS, NUM_PIXELS))
  var = np.zeros((NUM_DIGITS, NUM_PIXELS))

  for i in range(NUM_DIGITS):
    idxs = labels == i
    mu[i] = np.mean(X[idxs], axis=0)
    var[i] = np.mean((X[idxs] - mu[i])**2, axis=0)
  
  return mu, var

def gaussianNB(X, mu, var, prior, NUM_IMAGES, NUM_DIGITS):
  eps = 1e-8

  loglikelihood = np.zeros((NUM_IMAGES, NUM_DIGITS))
  logposterior = np.zeros((NUM_IMAGES, NUM_DIGITS))
  
  # compute likelihood and posterior
  for i in range(NUM_DIGITS):
    loglikelihood[:, i] = - (
      np.sum(np.log(var[i] + eps) / 2) \
      + 0.5 * np.sum((X - mu[i])**2 / (var[i] + eps), axis=1) / 2
    )
    logposterior[:, i] = np.log(prior[i]) + loglikelihood[:, i]

  labels_predicted = np.argmax(logposterior, axis=1)
  
  return labels_predicted

def predictGrayscale(X_train, X_test, y_train, y_test, NUM_DIGITS=10):
  # compute parameters
  X_train = uint2float(X_train)
  X_test = uint2float(X_test)
  mu, var = compute_parameters(X_train, y_train, NUM_DIGITS, X_train.shape[1])

  # compute prior
  prior = np.bincount(y_train) / y_train.size

  # predicted
  labels_predicted = gaussianNB(X_test, mu, var, prior, y_test.size, NUM_DIGITS)
  accuracy = get_accuracy(y_test, labels_predicted)

  print('Accuracy using original pixel intensity values as features {}'.format(accuracy))

def predictPCA(X_train, X_test, y_train, y_test, n_components=25, NUM_DIGITS=10):
  # after PCA
  Xp_train, Xp_test = performPCA(X_train, X_test, n_components)
  mu_p, var_p = compute_parameters(Xp_train, y_train, NUM_DIGITS, n_components)

  # compute prior
  prior = np.bincount(y_train) / y_train.size

  # predicted
  labels_predicted = gaussianNB(Xp_test, mu_p, var_p, prior, y_test.size, NUM_DIGITS)
  accuracy = get_accuracy(y_test, labels_predicted)

  print('Accuracy after PCA is {}'.format(accuracy))

def predictHoG(X_train, X_test, y_train, y_test, NUM_DIGITS=10):
  IMG_H, IMG_W = 28, 28
  NUM_TRAIN = y_train.size
  NUM_TEST = y_test.size
  hog_feats = 324

  Xh_train = np.zeros((NUM_TRAIN, hog_feats))
  Xh_test = np.zeros((NUM_TEST, hog_feats))
  hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
  for i in range(NUM_TRAIN):
    Xh_train[i] = np.squeeze(hog.compute(X_train[i].reshape(IMG_H, IMG_W)))
  for i in range(NUM_TEST):
    Xh_test[i] = np.squeeze(hog.compute(X_test[i].reshape(IMG_H, IMG_W)))
    
  # compute prior
  prior = np.bincount(y_train) / y_train.size

  # using HoG
  mu_h, var_h = compute_parameters(Xh_train, y_train, NUM_DIGITS, hog_feats)
  labels_predicted = gaussianNB(Xh_test, mu_h, var_h, prior, NUM_TEST, NUM_DIGITS)
  accuracy = get_accuracy(y_test, labels_predicted)

  print('Accuracy using HoG as features is {}'.format(accuracy))