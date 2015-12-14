from __future__ import absolute_import
from __future__ import print_function

from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from vis_func import *

from random import random
'''
Edited 18:44, 9, Sep, 2015
Taejin Park
inctrljinee@gmail.com
'''
# random.seed(0) # for reproducibility


# The function that deals with decimal numbering
def num10(num):

    if remainder(num,10) == 0 :
        p_num = 10
    else :
        p_num = remainder(num,10)

    return p_num

# The main function that returns probability outputs.
def NN_judge2(X_test, y_test):

    weight_file = 'NNM_cla_ss_0__music_noise__j26_0000000000002_SR11kHz_goodmare_merged_epoch10_32_last_weights'

    nb_epoch = 50

    patience = nb_epoch

    pr_val_box = []

    tn = 1

    nb_classes = 2
    FN = 1

    print ('Patience : ', patience)

    # last_node = 1024  #1024 for foxyroo j13 107
    last_node = 1024  #1024 for foxyroo j13 107
    # units = [32,64,64,64,64,64,64,64] #32 for foxyroo j13 107

    units = [32]*8
    filter_A = [3, 3, 3, 3, 3, 3]

    img_sz = [64, 64*4]
    img_pH = img_sz[0]/8
    img_pW = img_sz[1]/8


    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    optimizer =  'RMSprop'


    print('X_test shape:', X_test.shape)

    model_f3 = Sequential()

    model_f3.add(Convolution2D(units[0], FN, filter_A[0], filter_A[0], border_mode='full')) # (32, 3, 3, 3)
    convout1 = Activation('relu')
    model_f3.add(convout1)

    model_f3.add(Convolution2D(units[1], units[0], filter_A[1], filter_A[1])) # (32, 32, 3, 3)
    model_f3.add(Activation('relu'))
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.25))   #0.25

    model_f3.add(Convolution2D(units[2], units[1], filter_A[2], filter_A[2], border_mode='full')) # (64, 32, 3, 3)
    model_f3.add(Activation('relu'))
    model_f3.add(Convolution2D(units[3], units[2], filter_A[3], filter_A[3])) # (64, 64, 3, 3)
    model_f3.add(Activation('relu'))
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.25))   #0.25

    model_f3.add(Convolution2D(units[4], units[3], filter_A[4], filter_A[4], border_mode='full')) # (64, 32, 3, 3)
    model_f3.add(Activation('relu'))
    model_f3.add(Convolution2D(units[5], units[4], filter_A[5], filter_A[5])) # (64, 64, 3, 3)
    model_f3.add(Activation('relu'))
    model_f3.add(MaxPooling2D(poolsize=(2, 2)))
    model_f3.add(Dropout(0.5))   #0.25

    model_f3.add(Flatten())

    model_f3.add(Dense(units[5]*img_pW*img_pH, last_node)) # (64*8*8, 512)
    model_f3.add(Activation('relu'))
    model_f3.add(Dropout(0.5))  # 0.5

    model_f3.add(Dense(last_node, nb_classes)) # (512,10)
    model_f3.add(Activation('softmax'))


    model_f3.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model_f3.load_weights('tmp/'+ weight_file +'.hdf5')


    print ('Value Prediction')
    pr_val = model_f3.predict(X_test, verbose=1)
    pr_val_box.append(pr_val)

    i = 1

    convout1_f = theano.function([model_f3.get_input(train=False)], convout1.get_output(train=False))
    #convout2_f = theano.function([model.get_input(train=False)], convout2.get_output(train=False))

    # utility functions
    # Visualize the first layer of convolutions on an input image
    X = X_test[i:i+1]

    print ('shape(X),', shape(X))

    pl.figure()
    pl.title('input')
    nice_imshow(pl.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)

    show()

    # Visualize weights
    W = model_f3.layers[0].W.get_value(borrow=True)
    W = np.squeeze(W)
    print("W shape : ", W.shape)

    pl.figure(figsize=(15, 15))
    pl.title('conv1 weights')
    nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)

    C1 = convout1_f(X)
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)

    pl.figure(figsize=(15, 15))
    pl.suptitle('convout1')
    nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)

    show()

    return pr_val