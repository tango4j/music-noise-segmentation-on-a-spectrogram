from __future__ import absolute_import
from __future__ import print_function
# import numpy as np
from numpy import *
random.seed(0) # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.normalization import LRN2D
from keras.layers.core import Dense, Dropout, Activation, Flatten,Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from six.moves import range
from pylab import *
from numpy.random import normal

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


import h5py
from h5py import *
import pickle
import finish_alarm
import set_file_func
import set_file_func_novalid
from tts import *
from time import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from requests.exceptions import ConnectionError
# from keras.utils.dot_utils import Grapher


# Compute ROC curve and ROC area for each class

with open('pr_val_box_matrix'+'.pickle') as f:
    fpr_box, tpr_box, roc_auc_box, fnn_test_box, fnn_test_box_SNR, pr_val_box= pickle.load(f)

img_path = 'rqa_img/'
file_name = 'ROC_' + fnn_test_box[0]
# figure()
M = len(fnn_test_box)
color_box = ['red', 'magenta', 'orange', 'green', 'blue', 'purple']
color_box = color_box[::-1]
linestyles = ['-', '-', '-', '-', '-', '-']

for k in [0,1,2,3,5]:

    LW = 2
    plot(fpr_box[k][0], tpr_box[k][0], color = color_box[k], linestyle=linestyles[k], label='ROC of SNR '+ fnn_test_box_SNR[k] + '(area = %0.3f)'% roc_auc_box[k][0], linewidth = LW )
    plot([0, 1], [0, 1], 'k--')
    xlim([0.0, 1.0])
    ylim([0.0, 1.01])
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title('Receiver Operating Characteristic (ROC)')
    legend(loc="lower right")


savefig(img_path + file_name + '.png', dpi=100)

index = 5

pr_val_hist = log10(pr_val_box[index])
# pr_val_hist = (pr_val_box[index])

figure()

gaussian_numbers = normal(size=1000)
print ('shape of gn', shape(pr_val_hist))
print ('pr_val_hist', pr_val_hist)


bins = 50
plt.hist(pr_val_hist, bins=bins, range=[-12, 0])
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


