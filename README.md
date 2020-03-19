# Classification using Naive Bayes

Classification using Naive Bayes on the MNIST dataset. For project explanation see [1]

### Run

You can run the code using `python runNaiveBayes.py --trainpath PATH_TO_TRAIN --testpath PATH_TO_TEST --algorithm ALGORITHM_CODE`. There are four options for `algorithm`:  
1. `B` - Image is binarized and features are modeled as Bernoulli random variables.
2. `G_gray` - Original grayscale values of the image are used as features and modeled as jointly Gaussian random variables.
3. `G_pca` - PCA is used for dimensionality reduction of images before classification.
4. `G_hog` - Histogram of oriented gradients are computed from the image and used as features for classification.

### Dataset

You'll require access to the MNIST dataset which can be obtained from [2]. For ease of use, the code has been written to load the data in `.csv` format which can be obtained from [3]. After downloading store the datasets in `./data/`.

### References

[1] - [Deepankar C. - A Primer to Bayes' Classifier](https://deepankarc.github.io/2020/02/04/infogan/)  
[2] - [Y. LeCun, C. Cortes, C. Burges - The MNIST Database](http://yann.lecun.com/exdb/mnist/)  
[3] - [MNIST in CSV](https://www.kaggle.com/oddrationale/mnist-in-csv)
