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


def call_data(fnn, test_set_num, valid_set_num, speak):

    X_train, X_test, X_valid, y_train, y_test, y_valid, D_train, D_test, D_valid, jump_num, FN, nb_classes, open_data = set_file_func.set_data(fnn, test_set_num, valid_set_num, 0)

    return X_train, X_test, X_valid, y_train, y_test, y_valid, D_train, D_test, D_valid

def num10(num):

    if remainder(num,10) == 0 :
        p_num = 10
    else :
        p_num = remainder(num,10)

    return p_num


## Don't forget to edit nb_classes and nb_epoches
# open_data = [ 'MIS_f1_j7_0108080808_SR24kHz_foxyowl']
# open_data = ['MIS_f4_j7_0108080808_SR24kHz_wetlynx']
# open_data = ['MIS_f4_j9_000010010010010_SR44kHz_goodpig']
# open_data = ['MIS_f4_j7_8080808080808_SR44kHz_sharpcat']

'''
Edited 18:44, 9, Sep, 2015


'''
weight_file = 'NNM_cla_ss_0__music_noise__j26_0000000000002_SR11kHz_goodmare_merged_epoch10_32_last_weights'

# open_data = ['NNM_cla_ss_1_6_j102_0000000000002_SR11kHz_bigshep']
fnn_train = 'NNM_cla_ss_0__music_noise__j26_0000000000002_SR11kHz_goodmare'


fnn_test_box = ['NNM_cla_ss_0_mix_test_pm10_j26_0000000000002_SR11kHz_koalapaw',
                'NNM_cla_ss_0_mix_test_p0_j26_0000000000002_SR11kHz_badhorse',
                'NNM_cla_ss_0_mix_test_p10_j26_0000000000002_SR11kHz_muskypup',
                'NNM_cla_ss_0_mix_test_p20_j26_0000000000002_SR11kHz_wetpup',
                'NNM_cla_ss_0_mix_test_p30_j26_0000000000002_SR11kHz_tinyorca',
                'NNM_cla_ss_0_mix_test_pinf_j26_0000000000002_SR11kHz_redshark']

# fnn_test = 'NNM_cla_ss_1_mix_test_p20_j102_0000000000002_SR11kHz_aquadeer'

fnn_test_box_SNR = ['-10dB', '0dB', '10dB', '20dB', '30dB', 'clean']


nb_epoch = 50

patience = nb_epoch

batch_size = 256

result_data = 'result_data/'
foldername_confusion_matrix = 'confusion_matrix/'
foldername_pr_matrix = 'pr_matrix/'

pickle_path = 'pickle_folder/'
test_result_key = '_merged_epoch10_32'
speak = 0

# set_list = [1,2,3,4,5, 6,7,8,9,10]
set_list = [1,2,3,4,5, 6,7,8,9,10]
set_list = [1]
# set_list = [6,7,8,9,10]

start = clock()

pr_val_box = []
fpr_box = []
tpr_box = []
roc_auc_box = []

for counter, fnn_test in enumerate(fnn_test_box):

    test_size = 10

    total_score = zeros((test_size, 1), dtype = float32 )
    sample_score = zeros((test_size, 1), dtype = float32 )

    test_key = fnn_train
    tn = 1

    start_tset = clock()

    test_set_num = tn

    valid_set_index = [1,2,3,4,5,6,7,8,9,10]
    valid_set_num = num10(tn+1)

    X_train, X_valid, y_train, y_valid, D_train, D_valid, jump_num, FN, nb_classes, open_data = set_file_func_novalid.set_data_onlytrain(fnn_train, test_set_num, valid_set_num, speak)

    X_test, X_valid, y_test, y_valid, D_test, D_valid, jump_num, FN, nb_classes, open_data= set_file_func_novalid.set_data_onlytest(fnn_test, test_set_num, valid_set_num, speak)

    def most_common(lst):
        return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[0])[0]


    print ('Data file : '+ fnn_train, 'and', 'test set - ', tn)
    print ('Patience : ', patience)

    ## Confusion Matrix
    confusion_matrix = zeros((nb_classes, nb_classes))
    confusion_matrix_mv = zeros((nb_classes, nb_classes))


    # last_node = 1024  #1024 for foxyroo j13 107
    last_node = 1024  #1024 for foxyroo j13 107
    # units = [32,64,64,64,64,64,64,64] #32 for foxyroo j13 107
    units = [32,32,32,32,32,32] #32 for foxyroo j13 107
    units = [32]*8
    # units = [16,32,16,32,16,32] #32 for foxyroo j13 107
    # filter = [5, 5, 5, 5, 5, 5]
    filter_A = [3, 3, 3, 3, 3, 3]
    filter_B = [3, 3, 3, 3, 3, 3]
    # filter_C = [6,6,6,6,6,6]


    img_sz = [64, 64*4]
    img_pH = img_sz[0]/8
    img_pW = img_sz[1]/8
    # optimizer = 'RMSprop'
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    optimizer =  'RMSprop'
    # the data, shuffled and split between tran and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()


    '''
        Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

        GPU run command:
            THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

        It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
        (it's still underfitting at that point, though).

        Note: the data was pickled with Python 2, and some encoding issues might prevent you
        from loading it in Python 3. You might have to load it in Python 2,
        save it in a different format, load it in Python 3 and repickle it.
    '''


    data_augmentation = True

    # the data, shuffled and split between tran and test sets
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    # print('X_valid shape:', X_valid.shape )
    print('X_test shape:', X_test.shape)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test =  np_utils.to_categorical(y_test,  nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)


    model_f3 = Sequential()

    model_f3.add(Convolution2D(units[0], FN, filter_A[0], filter_A[0], border_mode='full')) # (32, 3, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(Convolution2D(units[1], units[0], filter_A[1], filter_A[1])) # (32, 32, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.25))   #0.25

    model_f3.add(Convolution2D(units[2], units[1], filter_A[2], filter_A[2], border_mode='full')) # (64, 32, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(Convolution2D(units[3], units[2], filter_A[3], filter_A[3])) # (64, 64, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.25))   #0.25

    model_f3.add(Convolution2D(units[4], units[3], filter_A[4], filter_A[4], border_mode='full')) # (64, 32, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(Convolution2D(units[5], units[4], filter_A[5], filter_A[5])) # (64, 64, 3, 3)
    model_f3.add(Activation('relu'))
    # model_f3.add(LRN2D())
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.5))   #0.25

    model_f3.add(Flatten())

    # model.add(Dense(units[5]*img_p2*img_p2, last_node)) # (64*8*8, 512)
    model_f3.add(Dense(units[5]*img_pW*img_pH, last_node)) # (64*8*8, 512)
    model_f3.add(Activation('relu'))
    model_f3.add(Dropout(0.5))  # 0.5

    model_f3.add(Dense(last_node, nb_classes)) # (512,10)
    model_f3.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model_f3.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    # for ec in range(0,20):
    # speak_str('Initiate fitting process')

    checkpointer = ModelCheckpoint(filepath='tmp/'+ str(test_key) + test_result_key +'_best_weights.hdf5', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    # out = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer, early_stop])
    # out =model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=1, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid), callbacks=[checkpointer, early_stop])
    # [('acc', 0.63989637305699487), ('loss', 0.91340676840516977), ('batch', 53), ('val_acc', array(0.5056689342403629)), ('val_loss', array(1.4115771055221558, dtype=float32)), ('size', 31)]

    val_accs = [0]
    count = 0
    #
    #
    # for p in range(0,nb_epoch):
    #     print ('############### EPOCH: ['+str(p)+'] ##################', 'set_list :', str(set_list), 'test set:', str(test_set_num))
    #     count += 1
    #     out =model_f3.fit([X_train], Y_train, batch_size=batch_size, nb_epoch=1, shuffle=1, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer, early_stop])
    #
    #
    # model_f3.save_weights('tmp/'+ str(test_key) + test_result_key +'_last_weights.hdf5', overwrite=True)
    #
    #

    model_f3.load_weights('tmp/'+ weight_file +'.hdf5')
    score_last = model_f3.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)

    print('Last weights-Test score:', score_last[0])
    print('Last weights-Test accuracy:', score_last[1])

    print ('Value Prediction')
    pr_val = model_f3.predict(X_test, verbose=1)
    pr_val_box.append(pr_val)
    print (pr_val, type(pr_val))

    print ('Class Prediction')
    pr = model_f3.predict_classes(X_test, verbose=1)
    print (pr, type(pr))

    score = score_last

    # print('Predict List:', predict)
    file = open(result_data + "result " + test_result_key + '_'+ open_data + '_test_sets' + ".txt", 'a')
    file.write('Data : '+ str(open_data) + ': test set ' + str(tn) + '\n\n')
    file.write('Note : '+ 'Valid set = Valid set version' + '\n')
    file.write(' Test score:' + str(score[0]) + '\n' + ' Test accuracy:' + str(score[1]) + "\n\n")
    file.write('Epoch :'+ str(nb_epoch) + '\n' )
    file.write('List of the incorrect predictions \n')

    ## Write down wrongly classified samples and confusion matrix
    for k in range(0, len(pr) ):

        if squeeze(y_test[k]) != int(pr[k]) :
            out_txt = str(transpose(D_test[k])) + "- Label : " + str(squeeze(y_test[k])) + " Prediction : "+ str(int(pr[k]) ) +"\n"
            # print (out_txt)
            file.write(out_txt)

        confusion_matrix[int(y_test[k]), int(pr[k])] += 1


    ## The number of label class
    len_pr = len(pr)

    ## The number of samples
    len_pr_mi = len(pr) / jump_num

    count_mv = 0

    y_test_sq = squeeze(y_test)
    y_test_sq = y_test_sq.tolist()

    for q in range(0, len_pr_mi):

        ## Get the list from prediction output matrix
        in_list = pr[q*jump_num:(q+1)*jump_num].tolist()

        ## Most common element in pr
        mce = most_common(in_list)

        ## Update Confusion Matrix
        confusion_matrix_mv[int(y_test_sq[q*jump_num]), int(mce)] += 1

        ## If the prediction is correct, count it.
        if y_test_sq[q*jump_num] == mce :
            count_mv += 1

    print ('count_mv:', count_mv)
    mv_score = float( float(count_mv) / float(len_pr_mi))
    total_score[tn - 1] = mv_score
    sample_score[tn - 1] = score[1]

    print ('\nData : '+ str(open_data) + ': test set ' + str(tn) + '\n')
    print ("Majority Vote Score :" + str(mv_score)+'\n')
    file.write('\nData : '+ str(open_data) + ': test set ' + str(tn) + '\n')
    file.write("Majority Vote Score :" + str(mv_score) + '\n\n')

    file.write("Patience :" + str(patience) + '\n')
    file.write("Batch Size :" + str(batch_size) + '\n')
    file.write("nb_epoch :" + str(nb_epoch) + '\n')
    file.write("Last Node :" + str(last_node) + '\n')
    file.write("Units :" + str(units) + '\n')
    file.write("filter_A :" + str(filter_A) + '\n')
    file.write("filter_B :" + str(filter_B) + '\n')
    # file.write("filter_C :" + str(filter_C) + '\n')
    file.write("Optimizer :" + str(optimizer) + '\n\n')

    stop_tset = clock()
    elap_tset = stop_tset - start_tset
    Total_time_tset = int(elap_tset / 60)
    file.write("The Elapsed Time for this test set :" + str(Total_time_tset) + ' min' + '\n')
    print ("The Elapsed Time for this test set :" + str(Total_time_tset) + ' min' + '\n')
    file.write('===================================================================' + '\n\n')

    file.close()
    # finish_alarm.ring('piano01')

    print('Test set ' + str(tn) + ' is complete.')

    score_recog = '{0:.3f}'.format(score[1])
    mv_score = '{0:.3f}'.format(mv_score)

    if speak ==1:
        try:
            speak_str('Test set. ' + str(tn) + ' is complete.')
            speak_str('The recognition score is, ' + score_recog +'.' + score_recog +'.')
            speak_str('and the majority vote score is, ' + mv_score +'.' + mv_score +'.')

        finally:
            print ()

    if nb_classes < 11 :
        print ('Confusion Matrix :\n\n' + str(confusion_matrix) + '\n' )

    with open( foldername_confusion_matrix + 'confusion_matrix' + test_result_key + '_' + open_data + '_testset_' + str(test_set_num) + '.pickle', 'w') as f:
        pickle.dump([confusion_matrix, confusion_matrix_mv], f)

    with open( foldername_pr_matrix + 'pr_matrix'  + '_' + fnn_test + '.pickle', 'w') as f:
        pickle.dump([pr, pr_val], f)


    print ('Average score :', str(average(total_score)))

    file2 = open(result_data + "result " + test_result_key + '_'+ open_data + "_total_score" + ".txt", 'a')
    file2.write('Data : '+ str(open_data) + ': test sets ' + str(squeeze(set_list)) + '\n\n')

    file2.write('Sample Scores :\n')
    for p in sample_score :  file2.write(str(squeeze(p)) + '\n')

    file2.write('Standard deviation of the scores :' + str(std(sample_score)) + '\n')
    file2.write('Average of the scores :' + str(average(sample_score)) + '\n\n')

    print ('\n\n\n')
    file2.write('\n\n\n')
    file2.write('Major Voting Scores :\n')
    for p in total_score :  file2.write(str(squeeze(p)) + '\n')


    file2.write('Standard deviation of the scores :' + str(std(total_score)) + '\n')
    file2.write('Average of the scores :' + str(average(total_score)) + '\n\n')

    file2.write("Patience :" + str(patience) + '\n')
    file2.write("Batch Size :" + str(batch_size) + '\n')
    file2.write("nb_epoch :" + str(nb_epoch) + '\n')
    file2.write("Last Node :" + str(last_node) + '\n')
    file2.write("Units :" + str(units) + '\n')
    file2.write("filter_A :" + str(filter_A) + '\n')
    file2.write("filter_B :" + str(filter_B) + '\n')
    # file2.write("filter_C :" + str(filter_C) + '\n')
    file2.write("Optimizer :" + str(optimizer) + '\n\n')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = nb_classes

    print ( 'shape of y_test and pr_val', shape(y_test), shape(pr_val) )
    print ('y_test', squeeze(squeeze(y_test)))
    print ('shape y_test', shape(y_test))
    y_testM = concatenate( (1-y_test, y_test), axis = 1)
    print (y_testM)
    print ('shape y_testM', shape(y_testM))

    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_testM[:,i], pr_val[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_box.append(fpr)
    tpr_box.append(tpr)
    roc_auc_box.append(roc_auc)


    stop = clock()
    elap_t = stop - start
    Total_time = int(elap_t / 60)
    file2.write("Total Elapsed Time :" + str(Total_time) + ' min' + '\n')
    print ("Total Elapsed Time :" + str(Total_time) + ' min' + '\n')
    avg_total_score = average(total_score)
    total_score_str = '{0:.3f}'.format(avg_total_score)
    file2.close()

    if speak == 1:
        try:
            speak_str('All the tests are complete.')
            speak_str('The average recognition rate is' + total_score_str +'.')
        except ConnectionError as e:
            print ('ConnectionError \n')


# Compute ROC curve and ROC area for each class
with open('pr_val_box_matrix' + '.pickle', 'w') as f:
    pickle.dump([fpr_box, tpr_box, roc_auc_box, fnn_test_box, fnn_test_box_SNR, pr_val_box], f)


img_path = 'rqa_img/'
file_name = 'ROC_' + fnn_test
figure()


for k in range (len(fnn_test_box)):

    LW = 2
    plot(fpr_box[k][0], tpr_box[k][0],  color = (0, k/len(fnn_test_box), 0, 1), label='ROC curve of SNR '+ fnn_test_box_SNR[k] + '(area = %0.4f)'% roc_auc_box[k][0], linewidth = LW )
    plot([0, 1], [0, 1], 'k--')
    xlim([0.0, 1.0])
    ylim([0.0, 1.01])
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title('Receiver operating characteristic example')
    legend(loc="lower right")


savefig(img_path + file_name + '.png', dpi=100)

# figure()
# plot(fpr[1], tpr[1], 'r-',label='ROC curve (area = %0.4f)' % roc_auc[1], linewidth = LW )
# plot([0, 1], [0, 1], 'k--' )
# xlim([0.0, 1.0])
# ylim([0.0, 1.01])
# xlabel('False Positive Rate')
# ylabel('True Positive Rate')
# title('Receiver operating characteristic example')
# legend(loc="lower right")
#
# show()