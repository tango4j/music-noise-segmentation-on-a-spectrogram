__author__ = 'mac'

from numpy import *
from pylab import *
# random.seed(0) # for reproducibility


# from h5py import *
import pickle

cm_num = 1
cl_num = 4
confusion_matrix_sum = zeros((cl_num,cl_num))
for k in range(0,10):
    pickle_path = 'confusion_matrix/'
    # path = pickle_path + 'confusion_matrix_HYBRID_RP&Spg__MIS_f0_j13_8080808080808_SR44kHz_cyanmare_testset_'
    path = pickle_path + 'confusion_matrix_HYBRID_RP&Spg__MIS_f5_j13_8080808080808_SR44kHz_sharpdog_testset_'
    # path = pickle_path + 'confusion_matrix_HYBRID_RP&Spg__MIS_f4_j13_8080808080808_SR44kHz_bigcat_testset_'

    print cm_num
    with open(path + str(cm_num) + '.pickle') as f:
        confusion_matrix, confusion_matrix_mv = pickle.load(f)
    confusion_matrix_sum = confusion_matrix_sum + confusion_matrix
    cm_num = cm_num + 1


# print confusion_matrix

confusion_matrix_norm = confusion_matrix

for k in range(0,cl_num):
    confusion_matrix_norm[k, :]= confusion_matrix_norm[k, :]/sum(confusion_matrix[k,:])


# print confusion_matrix_norm

confusion_matrix = squeeze(confusion_matrix)

figure()
imshow(confusion_matrix, interpolation = 'None')
colorbar()
# set_cmap('gray')

img_path = 'rqa_img/'
file_name = 'CM4cl'
savefig(img_path + file_name + '.png', dpi=100)
# print "The feature data per one inst sample is (=jump_num) :", jump_num

# show()