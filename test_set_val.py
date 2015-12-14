__author__ = 'inctrl'

from numpy import *
'''
    test_set.py
    This python file divides pickle data into Train set, Valid set, Test set.
'''


def set_train_data(sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_d_data, tst_key, val_key):

    dim_x = shape(sum_mat_X_data)
    dim = dim_x[0]

    x_train = []
    y_train = []
    k_train = []
    d_train = []


    x_test = []
    y_test = []
    k_test = []
    d_test = []

    x_valid = []
    y_valid = []
    k_valid = []
    d_valid = []




    all_key = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    print 'tst_key is ', tst_key

    all_key.remove(tst_key)


    id = nonzero(sum_mat_k_data[:]==array([tst_key] ) )

    print 'Generating testing key:', tst_key
    # print str(id)

    x_test = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    y_test = array(sum_mat_y_data)[id[0]]
    k_test = array(sum_mat_k_data)[id[0]]
    d_test = array(sum_mat_d_data)[id[0]]

    print 'val_key is ', val_key

    all_key.remove(val_key)

    id = nonzero(sum_mat_k_data[:]==array([val_key] ) )

    print 'Generating validating key:', val_key
    # print str(id)

    x_valid = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    y_valid = array(sum_mat_y_data)[id[0]]
    k_valid = array(sum_mat_k_data)[id[0]]
    d_valid = array(sum_mat_d_data)[id[0]]

    print 'Generating training key:',

    for key_idx, key in enumerate(all_key):

        x_train_each = []
        y_train_each = []
        k_train_each = []
        d_train_each = []

        print key,' ',

        id = nonzero(sum_mat_k_data[:]==array([key] ) )

        if key_idx == 0 :

            x_train = array(sum_mat_X_data)[id[0]]
            y_train = array(sum_mat_y_data)[id[0]]
            k_train = array(sum_mat_k_data)[id[0]]
            d_train = array(sum_mat_d_data)[id[0]]

            x_train_each = array(sum_mat_X_data)[id[0]]
            y_train_each = array(sum_mat_y_data)[id[0]]
            k_train_each = array(sum_mat_k_data)[id[0]]
            d_train_each = array(sum_mat_d_data)[id[0]]

        elif key_idx > 0 :

            x_train = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

            x_train_each = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train_each = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train_each = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train_each = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))


    print '\n'


    return x_train, x_test, x_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, d_train, d_test, d_valid


def set_train_data_novalid(sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_d_data, tst_key, val_key):

    dim_x = shape(sum_mat_X_data)
    dim = dim_x[0]

    x_train = []
    y_train = []
    k_train = []
    d_train = []


    x_test = []
    y_test = []
    k_test = []
    d_test = []

    x_valid = []
    y_valid = []
    k_valid = []
    d_valid = []


    all_key = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    print 'tst_key is ', tst_key

    all_key.remove(tst_key)


    id = nonzero(sum_mat_k_data[:]==array([tst_key] ) )

    print 'Generating testing key:', tst_key
    # print str(id)

    x_test = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    y_test = array(sum_mat_y_data)[id[0]]
    k_test = array(sum_mat_k_data)[id[0]]
    d_test = array(sum_mat_d_data)[id[0]]

    print 'Generating training key:',



    for key_idx, key in enumerate(all_key):

        x_train_each = []
        y_train_each = []
        k_train_each = []
        d_train_each = []

        print key,' ',

        id = nonzero(sum_mat_k_data[:]==array([key] ) )

        if key_idx == 0 :

            x_train = array(sum_mat_X_data)[id[0]]
            y_train = array(sum_mat_y_data)[id[0]]
            k_train = array(sum_mat_k_data)[id[0]]
            d_train = array(sum_mat_d_data)[id[0]]

            x_train_each = array(sum_mat_X_data)[id[0]]
            y_train_each = array(sum_mat_y_data)[id[0]]
            k_train_each = array(sum_mat_k_data)[id[0]]
            d_train_each = array(sum_mat_d_data)[id[0]]

        elif key_idx > 0 :

            x_train = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

            x_train_each = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train_each = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train_each = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train_each = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

        # x_valid.append(x_train_each)
        # y_valid.append(y_train_each)
        # k_valid.append(k_train_each)
        # d_valid.append(d_train_each)

    print '\n'


    return x_train, x_test, x_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, d_train, d_test, d_valid



def set_train_data_trainonly(sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_d_data, tst_key, val_key):

    dim_x = shape(sum_mat_X_data)
    dim = dim_x[0]

    x_train = []
    y_train = []
    k_train = []
    d_train = []


    x_test = []
    y_test = []
    k_test = []
    d_test = []

    x_valid = []
    y_valid = []
    k_valid = []
    d_valid = []


    all_key = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    # print 'tst_key is ', tst_key
    #
    # all_key.remove(tst_key)
    #
    #
    # id = nonzero(sum_mat_k_data[:]==array([tst_key] ) )
    #
    # print 'Generating testing key:', tst_key
    # # print str(id)
    #
    # x_test = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    # y_test = array(sum_mat_y_data)[id[0]]
    # k_test = array(sum_mat_k_data)[id[0]]
    # d_test = array(sum_mat_d_data)[id[0]]

    print 'Generating training key:',



    for key_idx, key in enumerate(all_key):

        x_train_each = []
        y_train_each = []
        k_train_each = []
        d_train_each = []

        print key,' ',

        id = nonzero(sum_mat_k_data[:]==array([key] ) )

        if key_idx == 0 :

            x_train = array(sum_mat_X_data)[id[0]]
            y_train = array(sum_mat_y_data)[id[0]]
            k_train = array(sum_mat_k_data)[id[0]]
            d_train = array(sum_mat_d_data)[id[0]]

            x_train_each = array(sum_mat_X_data)[id[0]]
            y_train_each = array(sum_mat_y_data)[id[0]]
            k_train_each = array(sum_mat_k_data)[id[0]]
            d_train_each = array(sum_mat_d_data)[id[0]]

        elif key_idx > 0 :

            x_train = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

            x_train_each = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train_each = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train_each = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train_each = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

        # x_valid.append(x_train_each)
        # y_valid.append(y_train_each)
        # k_valid.append(k_train_each)
        # d_valid.append(d_train_each)

    print '\n'


    return x_train,  x_valid, y_train, y_valid, k_train, k_valid, d_train, d_valid



def set_train_data_testonly(sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_d_data, tst_key, val_key):

    dim_x = shape(sum_mat_X_data)
    dim = dim_x[0]

    x_train = []
    y_train = []
    k_train = []
    d_train = []


    x_test = []
    y_test = []
    k_test = []
    d_test = []

    x_valid = []
    y_valid = []
    k_valid = []
    d_valid = []


    all_key = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    # print 'tst_key is ', tst_key
    #
    # all_key.remove(tst_key)
    #
    #
    # id = nonzero(sum_mat_k_data[:]==array([tst_key] ) )
    #
    # print 'Generating testing key:', tst_key
    # # print str(id)
    #
    # x_test = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    # y_test = array(sum_mat_y_data)[id[0]]
    # k_test = array(sum_mat_k_data)[id[0]]
    # d_test = array(sum_mat_d_data)[id[0]]
    #
    print 'Generating all-test key:',



    for key_idx, key in enumerate(all_key):

        x_train_each = []
        y_train_each = []
        k_train_each = []
        d_train_each = []

        print key,' ',

        id = nonzero(sum_mat_k_data[:]==array([key] ) )

        if key_idx == 0 :

            x_train = array(sum_mat_X_data)[id[0]]
            y_train = array(sum_mat_y_data)[id[0]]
            k_train = array(sum_mat_k_data)[id[0]]
            d_train = array(sum_mat_d_data)[id[0]]

            x_train_each = array(sum_mat_X_data)[id[0]]
            y_train_each = array(sum_mat_y_data)[id[0]]
            k_train_each = array(sum_mat_k_data)[id[0]]
            d_train_each = array(sum_mat_d_data)[id[0]]

        elif key_idx > 0 :

            x_train = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

            x_train_each = vstack((x_train, array(sum_mat_X_data)[id[0]]  ))   #[sum_mat_X_data[k][:][:][:]]
            y_train_each = vstack((y_train, array(sum_mat_y_data)[id[0]]  ))
            k_train_each = vstack((k_train, array(sum_mat_k_data)[id[0]]  ))
            d_train_each = vstack((d_train, array(sum_mat_d_data)[id[0]]  ))

        # x_valid.append(x_train_each)
        # y_valid.append(y_train_each)
        # k_valid.append(k_train_each)
        # d_valid.append(d_train_each)

    print '\n'
    x_test = x_train
    y_test = y_train
    k_test = k_train
    d_test = d_train

    return x_test, x_valid, y_test, y_valid, k_test, k_valid, d_test, d_valid
