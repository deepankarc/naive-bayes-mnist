# import options_base as base options
import argparse

class Options():
  def __init__(self):
    self.parser = argparse.ArgumentParser(description='declare options for experiment')
    self.initialized = False

  def initialize(self):
    self.parser.add_argument('--trainpath', type=str, default='./data/mnist_train.csv', help="path of training dataset")
    self.parser.add_argument('--testpath', type=str, default='./data/mnist_test.csv', help="path of test dataset")
    self.parser.add_argument('--algorithm', type=str, default='G_gray', help="algorithm to use")

  def parse(self):
    # initialize parser
    if(not self.initialized):
      self.initialize()
      self.initialized = True
    self.args = self.parser.parse_args()

    return self.args