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


    id = nonzero(sum_mat_k_data[:] == array([tst_key] ) )

    print 'Generating testing key:', tst_key

    x_test = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    y_test = array(sum_mat_y_data)[id[0]]
    k_test = array(sum_mat_k_data)[id[0]]
    d_test = array(sum_mat_d_data)[id[0]]

    #
    # print 'shape check, x_test:', shape(x_test)
    # print 'shape check, y_test:', shape(y_test)






    #
    # print 'val_key is ', val_key
    #
    # all_key.remove(val_key)
    #
    # id = nonzero(sum_mat_k_data[:]==array([val_key] ) )
    #
    # print 'Generating validating key:', val_key
    #
    # x_valid = array(sum_mat_X_data)[id[0]]   #[sum_mat_X_data[k][:][:][:]]
    # y_valid = array(sum_mat_y_data)[id[0]]
    # k_valid = array(sum_mat_k_data)[id[0]]
    # d_valid = array(sum_mat_d_data)[id[0]]
    #
    # print 'shape check, x_valid:', shape(x_valid)
    # print 'shape check, y_valid:', shape(y_valid)








    for key_idx, key in enumerate(all_key):

        x_train_each = []
        y_train_each = []
        k_train_each = []
        d_train_each = []

        print 'Generating training key:', key

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


    return x_train, x_test, x_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, d_train, d_test, d_valid


    # ct_tr = 0
    # ct_val = 0
    # ct_tst = 0
    #
    # print "Test_set check", sum_mat_d_data
    #
    # for k in range(0, dim):
    #
    #     if mod(k,100) == 0:
    #         print k, "-th sample"
    #     # print "type of k", type(k)
    #
    #     if sum_mat_k_data[k] in tst_key:
    #
    #         if ct_tst == 0:
    #             # print "shape of SMXD", shape(sum_mat_X_data[k][:][:][:])
    #             x_test = [sum_mat_X_data[k][:][:][:]]
    #             y_test = [sum_mat_y_data[k]]
    #             k_test = [sum_mat_k_data[k]]
    #             d_test = [sum_mat_d_data[k]]
    #
    #         elif ct_tst > 0:
    #             x_test = vstack((x_test, [sum_mat_X_data[k][:][:][:]]))
    #             y_test = vstack((y_test, [sum_mat_y_data[k]]))
    #             k_test = vstack((k_test, [sum_mat_k_data[k]]))
    #             d_test = vstack((d_test, [sum_mat_d_data[k]]))
    #
    #         ct_tst += 1
    #
    #     elif sum_mat_k_data[k] in val_key:
    #
    #         if ct_val == 0:
    #             # print "shape of SMXD", shape(sum_mat_X_data[k][:][:][:])
    #             x_valid = [sum_mat_X_data[k][:][:][:]]
    #             y_valid = [sum_mat_y_data[k]]
    #             k_valid = [sum_mat_k_data[k]]
    #             d_valid = [sum_mat_d_data[k]]
    #
    #
    #         elif ct_val > 0:
    #             x_valid = vstack((x_valid, [sum_mat_X_data[k][:][:][:]]))
    #             y_valid = vstack((y_valid, [sum_mat_y_data[k]]))
    #             k_valid = vstack((k_valid, [sum_mat_k_data[k]]))
    #             d_valid = vstack((d_valid, [sum_mat_d_data[k]]))
    #
    #         ct_val += 1
    #
    #     else:
    #
    #         if ct_tr == 0:
    #             # print "shape of SMXD", shape(sum_mat_X_data[k][:][:][:])
    #             x_train = [sum_mat_X_data[k][:][:][:]]
    #             y_train = [sum_mat_y_data[k]]
    #             k_train = [sum_mat_k_data[k]]
    #             d_train = [sum_mat_d_data[k]]
    #
    #
    #         elif ct_tr > 0:
    #
    #             x_train = vstack((x_train,[sum_mat_X_data[k][:][:][:]]))
    #             y_train = vstack((y_train,[sum_mat_y_data[k]]))
    #             k_train = vstack((k_train,[sum_mat_k_data[k]]))
    #             d_train = vstack((d_train,[sum_mat_d_data[k]]))
    #
    #         ct_tr += 1
    # # print d_test, "This is d_test"
    # return x_train, x_test, x_valid, y_train, y_test, y_valid, k_train, k_test, k_valid, d_train, d_test, d_valid