import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA

# compute accuracy
def get_accuracy(actual, predicted):
  accuracy = np.sum(predicted == actual) / actual.size * 100

  return accuracy

# re-format data
def reformat_data(data_frame):
  np_data_frame = data_frame.values
  images = np_data_frame[:,1:]
  labels = np_data_frame[:,0]

  return [images, labels]

def plot_data(X, NUM_PLOTS=10):
  # plot data as a sanity check
  sample_idx = np.random.randint(0, X.shape[0], NUM_PLOTS)

  plt.figure(figsize=(12, 4))
  gs = gridspec.GridSpec(2, 5)
  gs.update(wspace=0.05, hspace=0.05)

  for i in range(0,NUM_PLOTS):
    plt.subplot(gs[i])
    plt.imshow(np.reshape(X[sample_idx[i],:],(28,28)),cmap='gray')
    plt.axis('off')

  plt.show()

def uint2float(x):
  return x.astype(np.float64) / 255

def performPCA(X_train, X_test, n_components=70):
  pca = PCA(n_components=n_components, whiten=True, random_state=42)
  pca.fit(X_train)
  Xp_train = pca.transform(X_train)
  Xp_test = pca.transform(X_test)
  
  return Xp_train, Xp_test