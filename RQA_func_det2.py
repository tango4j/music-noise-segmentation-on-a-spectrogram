__author__ = 'mac'

from pylab import *
import scipy.ndimage.interpolation as scindim
import numpy as np
import matplotlib.gridspec as gridspec
import librosa
import librosa.decompose
from matplotlib.colors import LogNorm
import numpy
from numpy import *
from skimage import img_as_uint
import skimage
import skimage.exposure
import numpy.random
import matplotlib as plt

## Max pooling function
def search_max(search_fsout):
    max_arg_ts = int(argmax(search_fsout))
    return max_arg_ts


## Max pooling function
def max_pooling2d(image, pool_size):

    dim_img = shape(image)

    pooled_img = zeros(( int(dim_img[0]/pool_size), int(dim_img[0]/pool_size) ), dtype=numpy.float32)

    for p in range(0, dim_img[0]/pool_size):
        for q in range(0, dim_img[1]/pool_size):

            set = image[ p*pool_size: (p+1)*pool_size ,q*pool_size : (q+1)*pool_size  ]
            cal = amax( reshape(set, (pool_size**2, 1)) )

            pooled_img[p, q] = cal

        ## FL2 END
    ## FL1 END

    return pooled_img

def max_pooling1d(image, pool_size):

    dim_img = shape(image)

    pooled_img = zeros((int(dim_img[0]/pool_size),), dtype=numpy.float32)

    for p in range(0, dim_img[0]/pool_size):

        set = image[ p*pool_size: (p+1)*pool_size ]
        cal_arg = argmax( abs( reshape(set, (pool_size, 1)) ))
        #
        # print cal_arg
        # print set
        # print int(squeeze(cal_arg))

        cal = set[ int(squeeze(cal_arg)) ]
        pooled_img[p] = cal

        ## FL2 END
    ## FL1 END

    return pooled_img


def max_pooling_avg(image, pool_size):

    dim_img = shape(image)

    pooled_img = zeros((int(dim_img[0]/pool_size),), dtype=numpy.float32)

    for p in range(0, dim_img[0]/pool_size):

        set = image[ p*pool_size: (p+1)*pool_size ]

        # print cal_arg
        # print set
        # print int(squeeze(cal_arg))

        cal = average(set[:])

        pooled_img[p] = cal

        ## FL2 END
    ## FL1 END

    return pooled_img

def gaussian(x, mu, sig):

    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def form_gs_mat(sz,mu,sig):

    gs_mat = zeros((sz, sz), dtype=float32)

    for y in range(1,sz+1):

        for x in range(1,sz+1):

            if y >= (-1*x):
                gs_mat[x-1, y-1]=gaussian( (-x+y), mu, sig)

            elif y < (-1*x):
                gs_mat[x-1, y-1]=gaussian( -(-x+y), mu, sig)

    return gs_mat



def MR_RQA(file_name, audio_sig, SR, offset, jump_num, TR, FR, jump, FN, RPpow, img_process, th, th_ts, gain, clim, I_mode, img_sz, img_plot, img_path, RPs_in_level, RPs_interval, ss_mode):

    if ss_mode == 1:
        num_s = 2
    elif ss_mode == 0:
        num_s = 1

    num_level = num_s

    ## List for output variable
    depth = 3
    total_bin = np.zeros((jump_num * depth, num_s, img_sz[0], img_sz[1]), dtype="float32")
    print shape(total_bin), 'of totalbin shape'
    img_orsz = img_sz

    # Set NFFT, HOP_LENGTH for spectrogram
    NFFT = 128
    HOP_LENGTH = NFFT/4

    # Plot per a line
    PL = int(ceil(FN/2))

    # Open a main window
    if img_plot in (1, 2, 3):

        fig_sz_H = 15
        fig_sz_W = 5
        fig = figure(figsize=(fig_sz_H,fig_sz_W ))
        fig = gridspec.GridSpec(depth, 1)
        # fig = gridspec.GridSpec(num_level + 2, PL, height_ratios=[0.4,   0.4, 0.4, 0.4, 0.4,    0.4, 0.4, 0.4, 0.4,    0.4, 0.4, 0.4, 0.4, 0.8, 0.8])

    # specgram(audio_sig, NFFT = NFFT, scale_by_freq=True,sides='default', noverlap = 0)
    # xticks([])
    print 'FN', FN
    pic_bin = np.zeros((num_s, img_sz[0], img_sz[1]), dtype="float32")

    print "\n"

    for index in range(0, jump_num):


        # Parameter for a plot
        y_val = 0.98

        ############## CAVEAT! #################
        # For plot :
        # This for loop should not be aggregated with the following for loop,
        # because this loop generates border in the timeseries plot.
        for q in np.arange(0, RPs_in_level[0]):

            # For plot : y_val
            y_val = y_val - q*0.005
            x_start = RPs_interval[q][0] # /float(SR)
            x_end = RPs_interval[q][0] + TR[-7] # /float(SR)

            # For plot : Set box size for each level
            x_list = [x_start, x_end, x_end, x_start, x_start]
            y_list = [-y_val, -y_val, y_val, y_val, -y_val]

        ########################################
        ## Put audio signal into "conved_sigs" list with pre-determined offset value.
        conved_sigs = audio_sig[(index*jump + offset):((index+1)*jump + offset)]


        rp_count = 0

        # gs_mat = form_gs_mat(img_orsz, 0, 8)
        r = 0

        NFFT = img_sz[0]*4*2
        HOP_LENGTH = NFFT/4

        seg_sig = conved_sigs[0:(HOP_LENGTH*img_sz[1])]

        audio_sig_show = audio_sig
        D = np.abs(librosa.stft(seg_sig, n_fft = NFFT, hop_length = HOP_LENGTH))

        for d in range(0, depth):
        ### Channel level loop
            for q in range(0, num_s):

                if ss_mode ==1 :
                    H, P = librosa.decompose.hpss(D, kernel_size=31, power=2.0, mask=False)
                    if q == 0:
                        Ds = H
                    elif q == 1:
                        Ds = P
                    elif q == 2:
                        Ds = D
                elif ss_mode == 0 :
                    Ds = D
                Dw = Ds[(0+img_sz[0]*d):((d+1)*img_sz[0]), 0:img_sz[1]]
                pic_bin[q, :, :] = Dw
            rp_count += 1

            # Plot per a line
            PL = int(ceil(FN/2))

            if img_plot in (1, 2, 3):
                ax1 = subplot(fig[depth-d-1, 0])
                img = librosa.display.specshow(Dw, sr=SR, y_axis='linear', x_axis='time',norm=LogNorm(vmin=0.01, vmax=10))
                img.set_cmap('jet')
                axis('off')

        ### END for loop
        if img_plot == 1:

            draw()
            show(block=False)

        elif img_plot == 2:

            draw()
            savefig(img_path + file_name + '_' + str(index+1)+ '.png', dpi=100)
            print "Image has been saved : " + img_path + file_name + '_' + str(index+1)+ '.png'

        elif img_plot == 3:

            draw()
            show(block=False)

        print str(index+1), "-th jump image...",
        if (index+1) != 1 and remainder(index+1,5) == 0 : print '\n',

        total_bin[depth*(index-1)+d, :, :, :] = pic_bin


    print "\n"

    if img_plot in (0,2):
        close("all")


    return total_bin

def RQA_eval(x_in, TR, RPpow, img_process):

    RQA_mat_L = zeros((TR, TR))

    # Set L,R matrix to compute Recurrence matrix
    RQA_mat_L[0:TR, :] = x_in
    RQA_mat_R = RQA_mat_L.transpose()

    if img_process == -1:

	# Subtract two matirices
	RQA_value = RQA_mat_L - RQA_mat_R

    if img_process == 0:

        # RQA_value = RQA_value/amax(abs(RQA_value))
        # Subtract two matirices
    	RQA_value = abs(RQA_mat_L - RQA_mat_R)
	RQA_value = RQA_value**RPpow

    if img_process == 1:

        # RQA_value = RQA_value/amax(abs(RQA_value))
        RQA_value = img_as_uint(RQA_value)
        RQA_value = skimage.exposure.equalize_hist(RQA_value)

    if img_process == 2:

        # RQA_value = RQA_value/amax(abs(RQA_value))
        RQA_value = img_as_uint(RQA_value)
        RQA_value = skimage.exposure.equalize_adapthist(RQA_value, clip_limit=0.01)

    RQA_value -= mean(RQA_value)

    return RQA_value

def preprocessing():
    return 0

def logistic(z,z0,a):
    return 1.0 / (1.0 + np.exp(-a*(z-z0)))
