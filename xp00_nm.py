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

open_data = ['NNM_cla_ss_1_6_j102_0000000000002_SR11kHz_bigshep']

fnn_test = 'NNM_cla_ss_1_6_j102_0000000000002_SR11kHz_'
# open_data = ['MIS_f0_j5_08080808_SR44kHz_ragelynx']
# open_data = ['MIS_f0_j50_0000000001110_SR16kHz_aquashep']
#open_data = ['MIS_f5_j13_8000008080808_SR44kHz_bluefox']
#open_data = ['MIS_f4_j13_8000008080808_SR44kHz_pigwing']

nb_epoch = 30

patience = 5
batch_size = 256

result_data = 'result_data/'
foldername_confusion_matrix = 'confusion_matrix/'

pickle_path = 'pickle_folder/'
test_result_key = '_merged_epoch10_32'
speak = 0

# set_list = [1,2,3,4,5, 6,7,8,9,10]
set_list = [1,2,3,4,5, 6,7,8,9,10]
set_list = [1]
# set_list = [6,7,8,9,10]

start = clock()

for fnn in open_data:

    test_size = 10

    total_score = zeros((test_size, 1), dtype = float32 )
    sample_score = zeros((test_size, 1), dtype = float32 )

    test_key = fnn

    for tn in set_list:

        start_tset = clock()

        test_set_num = tn

        valid_set_index = [1,2,3,4,5,6,7,8,9,10]
        valid_set_num = num10(tn+1)
        # X_train, X_test, X_valid, y_train, y_test, y_valid, D_train, D_test, D_valid, jump_num, FN, nb_classes, open_data = set_file_func.set_data(fnn, test_set_num, valid_set_num, speak)
        X_train, X_valid, y_train, y_valid, k_train, k_valid, D_train, D_valid  = set_file_func_novalid.set_data_onlytrain(fnn, test_set_num, valid_set_num, speak)

        X_test, X_valid, y_test, y_valid, k_test, k_valid, D_test, D_valid = set_file_func_novalid.set_data_onlytest(fnn_test, test_set_num, valid_set_num, speak)

        def most_common(lst):
            return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[0])[0]


        print ('Data file : '+ fnn, 'and', 'test set - ', tn)
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


        image_size = 64
        img_p2 = image_size/8
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


        model_f5 = Sequential()

        model_f5.add(Convolution2D(units[0], FN, filter_B[0], filter_B[0], border_mode='full')) # (32, 3, 3, 3)
        model_f5.add(Activation('relu'))
        # model_f5.add(LRN2D())
        model_f5.add(Convolution2D(units[1], units[0], filter_B[1], filter_B[1])) # (32, 32, 3, 3)
        model_f5.add(Activation('relu'))
        # model_f5.add(LRN2D())
        model_f5.add(MaxPooling2D(poolsize=(2, 2)))
        model_f5.add(Dropout(0.25))   #0.25

        model_f5.add(Convolution2D(units[2], units[1], filter_B[2], filter_B[2], border_mode='full')) # (64, 32, 3, 3)
        model_f5.add(Activation('relu'))
        # model_f5.add(LRN2D())
        model_f5.add(Convolution2D(units[3], units[2], filter_B[3], filter_B[3])) # (64, 64, 3, 3)
        model_f5.add(Activation('relu'))
        # model_f5.add(LRN2D())
        model_f5.add(MaxPooling2D(poolsize=(2, 2)))
        model_f5.add(Dropout(0.25))   #0.25

        model_f5.add(Convolution2D(units[4], units[3], filter_A[4], filter_A[4], border_mode='full')) # (64, 32, 3, 3)
        model_f5.add(Activation('relu'))
        # model_f5.add(LRN2D())
        model_f5.add(Convolution2D(units[5], units[4], filter_A[5], filter_A[5])) # (64, 64, 3, 3)
        model_f5.add(Activation('relu'))
        model_f5.add(MaxPooling2D(poolsize=(2, 2)))
        # model_f5.add(LRN2D())
        model_f5.add(Dropout(0.5))   #0.25

        model_f5.add(Flatten())



        # model_f7 = Sequential()
        # 
        # model_f7.add(Convolution2D(units[0], FN, filter_B[0], filter_B[0], border_mode='full')) # (32, 3, 3, 3)
        # model_f7.add(Activation('relu'))
        # model_f7.add(Convolution2D(units[1], units[0], filter_B[1], filter_B[1])) # (32, 32, 3, 3)
        # model_f7.add(Activation('relu'))
        # model_f7.add(MaxPooling2D(poolsize=(2, 2)))
        # model_f7.add(Dropout(0.25))   #0.25
        # 
        # # model_f7.add(Convolution2D(units[2], units[1], filter_B[2], filter_B[2], border_mode='full')) # (64, 32, 3, 3)
        # # model_f7.add(Activation('relu'))
        # # model_f7.add(Convolution2D(units[3], units[2], filter_B[3], filter_B[3])) # (64, 64, 3, 3)
        # # model_f7.add(Activation('relu'))
        # # model_f7.add(MaxPooling2D(poolsize=(2, 2)))
        # # model_f7.add(Dropout(0.25))   #0.25
        # 
        # model_f7.add(Flatten())

        model = Sequential()
        model.add(Merge([model_f3, model_f5], mode='sum'))

        model.add(Flatten())
        # model.add(Dense(units[5]*img_p2*img_p2, last_node)) # (64*8*8, 512)
        model.add(Dense(units[5]*img_p2*img_p2, last_node)) # (64*8*8, 512)
        model.add(Activation('relu'))
        model.add(Dropout(0.5))  # 0.5

        model.add(Dense(last_node, nb_classes)) # (512,10)
        model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
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


        for p in range(0,nb_epoch):
            print ('############### EPOCH: ['+str(p)+'] ##################', 'set_list :', str(set_list), 'test set:', str(test_set_num))
            count += 1
            # out =model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=1, shuffle=1, show_accuracy=True, verbose=1, validation_data=([X_test, X_test], Y_test), callbacks=[checkpointer, early_stop])
            out =model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=1, shuffle=1, show_accuracy=True, verbose=1, validation_data=([X_test, X_test], Y_test), callbacks=[checkpointer, early_stop])

        model.save_weights('tmp/'+ str(test_key) + test_result_key +'_last_weights.hdf5', overwrite=True)
        model.load_weights('tmp/'+ str(test_key) + test_result_key + '_last_weights.hdf5')
        score_last = model.evaluate([X_test, X_test], Y_test, show_accuracy=True, verbose=1)

        print('Last weights-Test score:', score_last[0])
        print('Last weights-Test accuracy:', score_last[1])

        pr = model.predict_classes([X_test, X_test], verbose=1)

        # print ('Value Prediction')
        # pr_val = model.predict([X_test, X_test], verbose=1)
        # print (pr_val, type(pr_val))

        print ('Class Prediction')
        pr = model.predict_classes([X_test, X_test], verbose=1)
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
        finish_alarm.ring('piano01')

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


