import parser

import numpy as np
import pandas as pd
import cv2

from binomial2 import predictBinomial
from gaussian import predictGrayscale, predictPCA, predictHoG
from utils import reformat_data, plot_data
from options import Options

def main(args):
  # read training data
  data_frame_train = pd.read_csv(args.trainpath, dtype=np.uint8, header=None)
  NUM_TRAIN = data_frame_train.shape[0]
  NUM_PIXELS = data_frame_train.shape[1]-1

  # read test data
  data_frame_test = pd.read_csv(args.testpath, dtype=np.uint8, header=None)
  NUM_TEST = data_frame_test.shape[0]

  # reformat data
  [images_train, labels_train] = reformat_data(data_frame_train)
  [images_test, labels_test] = reformat_data(data_frame_test)

  # plot data as sanity check
  plot_data(images_train)

  if(args.algorithm == 'B'):
    predictBinomial(images_train, images_test, labels_train, labels_test)
  elif(args.algorithm == 'G_gray'):
    predictGrayscale(images_train, images_test, labels_train, labels_test)
  elif(args.algorithm == 'G_pca'):
    predictPCA(images_train, images_test, labels_train, labels_test)
  elif(args.algorithm == 'G_hog'):
    predictHoG(images_train, images_test, labels_train, labels_test)

if __name__ == "__main__":
  parser = Options()
  args = parser.parse()
  main(args)