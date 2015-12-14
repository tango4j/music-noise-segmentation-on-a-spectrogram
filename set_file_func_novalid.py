__author__ = 'inctrl'

from numpy import *
import pickle
import test_set
import test_set_val
import finish_alarm
from tts import *
from requests.exceptions import ConnectionError

def set_data(open_data,test_set_num, valid_set_num, speak):
    '''
        set_file.py
        This python file divides pickle data into Train set, Valid set, Test set.
    '''
    #############################################################################
    # open_data = 'data_j1_01010101_SR48kHz'
    # open_data = 'data_j1_11111111_SR48kHz'
    # open_data = 'class13_data_j1_01010101_SR48kHz'
    #############################################################################
    pickle_path = 'pickle_folder/'
    with open(pickle_path + open_data + '.pickle') as f:
        sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes = pickle.load(f)

    print "The feature data per one inst sample is (=jump_num) :", jump_num

    ## Set the key for test set and validation set.
    ## The whole data is divided into 10 different data sets.
    ## You can choose a key number from 1 to 10.


    # val_key = [[8],[9]]
    # # val_key = []
    # tst_key = [[7],[1]]
    val_key = []
    # val_key =[]
    tst_key = [test_set_num]

    # print sum_mat_k_data

    ## Call "test_set.set_train_data" function to divide data
    X_train, X_test, X_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, D_train, D_test, D_valid \
        = test_set_val.set_train_data(sum_mat_X_data,sum_mat_y_data,sum_mat_k_data, sum_mat_D_data, tst_key, val_key)


    print "\n" + open_data + " : DATA set ready! \n"



    # speak_str('Data seperation for test set '+ str(test_set_num) +' is complete.')
    if speak == 1 :
        try:
            speak_str('Initiate classification for test set '+ str(test_set_num)+ '.'+'\n')
        except ConnectionError as e:
            print ('ConnectionError \n')

    # finish_alarm.ring('guitar_c3_04')

    return X_train, X_test, X_valid, y_train, y_test, y_valid, D_train, D_test, D_valid, jump_num, FN, nb_classes, open_data


def set_data_novalid(open_data,test_set_num, valid_set_num, speak):
    '''
        set_file.py
        This python file divides pickle data into Train set, Valid set, Test set.
    '''

    pickle_path = 'pickle_folder/'
    with open(pickle_path + open_data + '.pickle') as f:
        sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes = pickle.load(f)

    print "The feature data per one inst sample is (=jump_num) :", jump_num

    ## Set the key for test set and validation set.
    ## The whole data is divided into 10 different data sets.
    ## You can choose a key number from 1 to 10.


    # val_key = [[8],[9]]
    # # val_key = []
    # tst_key = [[7],[1]]
    val_key = []
    # val_key =[]
    tst_key = [test_set_num]

    # print sum_mat_k_data

    ## Call "test_set.set_train_data" function to divide data
    X_train, X_test, X_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, D_train, D_test, D_valid \
        = test_set_val.set_train_data_novalid(sum_mat_X_data,sum_mat_y_data,sum_mat_k_data, sum_mat_D_data, tst_key, val_key)


    print "\n" + open_data + " : DATA set ready! \n"

    # speak_str('Data seperation for test set '+ str(test_set_num) +' is complete.')
    if speak == 1 :
        try:
            speak_str('Initiate classification for test set '+ str(test_set_num)+ '.'+'\n')
        except ConnectionError as e:
            print ('ConnectionError \n')

    # finish_alarm.ring('guitar_c3_04')

    return X_train, X_test, X_valid, y_train, y_test, y_valid, D_train, D_test, D_valid, jump_num, FN, nb_classes, open_data


def set_data_onlytrain(open_data,test_set_num, valid_set_num, speak):
    '''
        set_file.py
        This python file divides pickle data into Train set, Valid set, Test set.
    '''

    pickle_path = 'pickle_folder/'
    with open(pickle_path + open_data + '.pickle') as f:
        sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes = pickle.load(f)

    print "The feature data per one inst sample is (=jump_num) :", jump_num

    ## Set the key for test set and validation set.
    ## The whole data is divided into 10 different data sets.
    ## You can choose a key number from 1 to 10.


    # val_key = [[8],[9]]
    # # val_key = []
    # tst_key = [[7],[1]]
    val_key = []
    # val_key =[]
    tst_key = []

    # print sum_mat_k_data

    ## Call "test_set.set_train_data" function to divide data
    X_train, X_valid, y_train, y_valid, k_train, k_valid, D_train, D_valid \
        = test_set_val.set_train_data_trainonly(sum_mat_X_data,sum_mat_y_data,sum_mat_k_data, sum_mat_D_data, tst_key, val_key)


    print "\n" + open_data + " : DATA set ready! \n"

    # speak_str('Data seperation for test set '+ str(test_set_num) +' is complete.')
    if speak == 1 :
        try:
            speak_str('Initiate classification for test set '+ str(test_set_num)+ '.'+'\n')
        except ConnectionError as e:
            print ('ConnectionError \n')

    # finish_alarm.ring('guitar_c3_04')

    return X_train, X_valid, y_train, y_valid, D_train, D_valid, jump_num, FN, nb_classes, open_data

def set_data_onlytest(open_data,test_set_num, valid_set_num, speak):
    '''
        set_file.py
        This python file divides pickle data into Train set, Valid set, Test set.
    '''
    #############################################################################
    # open_data = 'data_j1_01010101_SR48kHz'
    # open_data = 'data_j1_11111111_SR48kHz'
    # open_data = 'class13_data_j1_01010101_SR48kHz'
    #############################################################################
    pickle_path = 'pickle_folder/'
    with open(pickle_path + open_data + '.pickle') as f:
        sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes = pickle.load(f)

    print "The feature data per one inst sample is (=jump_num) :", jump_num

    ## Set the key for test set and validation set.
    ## The whole data is divided into 10 different data sets.
    ## You can choose a key number from 1 to 10.


    # val_key = [[8],[9]]
    # # val_key = []
    # tst_key = [[7],[1]]
    val_key = []
    # val_key =[]
    tst_key = [test_set_num]

    # print sum_mat_k_data

    ## Call "test_set.set_train_data" function to divide data
    X_test, X_valid, y_test, y_valid, k_test, k_valid, D_test, D_valid \
        = test_set_val.set_train_data_testonly(sum_mat_X_data,sum_mat_y_data,sum_mat_k_data, sum_mat_D_data, tst_key, val_key)


    print "\n" + open_data + " : DATA set ready! \n"

    # speak_str('Data seperation for test set '+ str(test_set_num) +' is complete.')
    if speak == 1 :
        try:
            speak_str('Initiate classification for test set '+ str(test_set_num)+ '.'+'\n')
        except ConnectionError as e:
            print ('ConnectionError \n')

    # finish_alarm.ring('guitar_c3_04')

    return X_test, X_valid, y_test, y_valid, D_test, D_valid, jump_num, FN, nb_classes, open_data
