__author__ = 'mac'

from pylab import *
import scipy.ndimage.interpolation as scindim
import numpy as np
import matplotlib.gridspec as gridspec
import librosa
from matplotlib.colors import LogNorm
import numpy
from numpy import *
from skimage import img_as_uint
import skimage
import skimage.exposure
import numpy.random


## Max pooling function
def search_max(search_fsout):
    max_arg_ts = int(argmax(search_fsout))
    return max_arg_ts

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



def MR_RQA(file_name, audio_sig, SR, offset, jump_num, TR, FR, jump, FN, RPpow, img_process, th, th_ts, gain, clim, I_mode, img_sz, img_plot, img_path, RPs_in_level, RPs_interval):

    num_level = len(RPs_in_level)

    ## List for output variable
    total_bin = np.zeros((jump_num, FN, img_sz, img_sz), dtype="float32")

    img_orsz = img_sz

    # Set NFFT, HOP_LENGTH for spectrogram
    NFFT = 128
    HOP_LENGTH = NFFT/16

    # Plot per a line
    PL = int(ceil(FN))
    PL =16

    # Open a main window
    if img_plot in (1, 2, 3):

        fig_sz = 15
        fig = plt.figure(figsize=(fig_sz,fig_sz ))
        fig = gridspec.GridSpec(num_level + 2, PL, height_ratios=[0.4,  0.4, 0.4,    0.4, 0.4, 0.4, 0.4,0.4, 0.4,    0.4, 0.4, 0.4, 0.4, 0.8, 0.8])
        # fig = gridspec.GridSpec(num_level + 2, PL, height_ratios=[0.4,   0.4, 0.4, 0.4, 0.4,    0.4, 0.4, 0.4, 0.4,    0.4, 0.4, 0.4, 0.4, 0.8, 0.8])

        ax2 = subplot(fig[13, :])
        SL=SR*19
        audio_sig_show = audio_sig[0:SL]
        D = np.abs(librosa.stft(audio_sig_show, n_fft = NFFT, hop_length = HOP_LENGTH))
        # img = librosa.display.specshow(D, sr=SR, y_axis='linear', x_axis='time',norm=LogNorm(vmin=0.01, vmax=10))
        img = librosa.display.specshow(D, sr=SR, y_axis='linear', x_axis='time',norm=LogNorm(vmin=0.01, vmax=10))
        img.set_cmap('jet')
        # show(block=False)

    # specgram(audio_sig, NFFT = NFFT, scale_by_freq=True,sides='default', noverlap = 0)
    # xticks([])

    pic_bin = np.zeros((FN, img_sz, img_sz), dtype="float32")

    print "\n"

    for index in range(0, jump_num):

        if img_plot in (1, 2, 3):
            audio_sig_show = audio_sig[0:SL]
            ax1 = subplot(fig[14, :])
            len_y= audio_sig_show.size
            t = linspace(0, len_y/float(SR), num = len_y)
            t = linspace(0, len_y, num = len_y)
            RGB=(0.2,0.2,1)

            pic = plot(t, audio_sig_show, color = RGB)
            # xticks([])
            draw()

        # Parameter for a plot
        y_val = 0.98

        ############## CAVEAT! #################
        # For plot :
        # This for loop should not be aggregated with the following for loop,
        # because this loop generates border in the timeseries plot.
        for q in np.arange(0, RPs_in_level[10]):

            # For plot : y_val
            # y_val = y_val - q*0.005
            y_val = 0.95
            x_start = RPs_interval[q][0] # /float(SR)
            x_end = RPs_interval[q][0] + TR[-5] # /float(SR)

            # For plot : Set box size for each level
            # x_list = [x_start, x_end, x_end, x_start, x_start]
            # y_list = [-y_val, -y_val, y_val, y_val, -y_val]
            x_list = [x_start, x_start, x_start, x_start, x_start]
            y_list = [-y_val, -y_val, y_val, y_val, -y_val]

            # For plot : Decrease box size in the time series plot
            RGB=(0.8 ,0.1,0.1)

            if img_plot in (1, 2, 3):

                # pic = plot(x_list, y_list, 'r-', linewidth=1)
                pic = plot(x_list, y_list, color = RGB, linewidth=1)
                draw()
        ########################################

        ## Put audio signal into "conved_sigs" list with pre-determined offset value.
        conved_sigs = audio_sig[(index*jump + offset):((index+1)*jump + offset + TR[0])]
        # print 'conved_sig', shape(conved_sigs)
        # print 'Now traveling :' + str(index*jump + offset) + 'th sample'
        rp_count = 0

        # gs_mat = form_gs_mat(img_orsz, 0, 8)
        floor = 1
        for q in range(0, num_level):

            for r in range(0, RPs_in_level[q]):

                # Get q-th signal from list "conved_sigs"

                # print "check out the interval:", TR[q]*r ,TR[q]*(r+1)
                # print 'RPs_interval[r][q]:', RPs_interval[r][q]
                # print 'conved_sigs len :',len(conved_sigs)
                # print 'from', str(RPs_interval[r][q]*r), 'to', str((RPs_interval[r][q]*r + TR[q]))

                # ################# MAX ALIGN ######################################
                # search_fsout = conved_sigs[(RPs_interval[r][q]):(RPs_interval[r][q] + (jump - 1) )]
                # max_arg_ts = search_max(search_fsout)
                # fsout = conved_sigs[(RPs_interval[r][q] + max_arg_ts):(RPs_interval[r][q] + max_arg_ts + TR[q])]
                #####################################################################

                fsout = conved_sigs[(RPs_interval[r][q]):(RPs_interval[r][q] + TR[q])]
                # print 'r:',r,'q:',q
                # fsout = conved_sigs[(TR[q]*r ):(TR[q]*r + TR[q])]
                # print 'from', str(TR[q]*r ), 'to', str(TR[q]*r + TR[q])

                # Make array with "fsout"
                fsout = array(fsout)
                # print 'shape of fsout', shape(fsout)
                # print 'TR[',q,']', TR[q]
                # print 'RPs_interval[',r,'][',q,']', RPs_interval[r][q]

                # If the frame size is too big, we max-pool it before creating a RQA mat.

                # Set level for the 1-d max pooling
                # max1d = minimum 2d max pooling number ex) n of 2^n maxpooling
                max1d = 3
                max1d_high = max1d - 0
                if FR[q] < -6 :

                    dec_scale_0 = -1*FR[q] - max1d_high

                    inv_s = 2**dec_scale_0

                    fsout = max_pooling1d(fsout, inv_s)
                    # fsout = max_pooling_avg(fsout, inv_s)

                    new_TR = int(TR[q]/inv_s)

                    # Call RQA_eval function ####################
                    RQA_value = RQA_eval(fsout, new_TR, RPpow, img_process)
                    #############################################

                    # Variable for print image section
                    flr = q
                    seq = r

                    # Set repeat count of max pooling for this level
                    dec_scale = 2**(-1*max1d_high)

                    # Set max-pooling scale
                    inv_d = int(dec_scale **(-1))

                    RQA_value = max_pooling2d(RQA_value, inv_d)

                elif FR[q] < -1*max1d:

                    dec_scale_0 = -1*FR[q] - max1d

                    inv_s = 2**dec_scale_0

                    fsout = max_pooling1d(fsout, inv_s)
                    # fsout = max_pooling_avg(fsout, inv_s)

                    new_TR = int(TR[q]/inv_s)

                    # Call RQA_eval function ####################
                    RQA_value = RQA_eval(fsout, new_TR, RPpow, img_process)
                    #############################################

                    # Variable for print image section
                    flr = q
                    seq = r

                    # Set repeat count of max pooling for this level
                    dec_scale = 2**(-1*max1d)

                    # Set max-pooling scale
                    inv_d = int(dec_scale **(-1))

                    RQA_value = max_pooling2d(RQA_value, inv_d)


                else:


                    # Call RQA_eval function ####################
                    RQA_value = RQA_eval(fsout, TR[q], RPpow, img_process)
                    #############################################

                    # Variable for print image section
                    flr = q
                    seq = r

                    # Set repeat count of max pooling for this level
                    dec_scale = 2**(FR[q])

                    # Set maxpolling scale
                    inv_d = int(dec_scale **(-1))
                    RQA_value = max_pooling2d(RQA_value, inv_d)

                # Store RQA image to output matrix
                img_redc = 0

                # RQA_value = multiply(RQA_value, gs_mat)

                pic_bin[rp_count, :, :] = RQA_value[img_redc:(img_orsz-img_redc), img_redc:(img_orsz-img_redc)]

                if img_plot in (1, 2, 3) :

                    # For plot : plot third graph
                    ax3 = subplot(fig[flr, seq])

                    xticks([]), yticks([])
                    cc = (1-float(0.1)*q)

                    ax3.set_title('RP' + str(TR[q]), fontsize=fig_sz)

                    # RQA_value = scindim.zoom(RQA_value, dec_scale)

                    # imgplot = plt.imshow(RQA_value, interpolation='nearest')
                    imgplot = plt.imshow(pic_bin[rp_count, :, :], interpolation=None, norm = None)
                    imgplot.set_cmap('gray')
                    # ax3.set_aspect('equal')
                    # plt.show(block=False)

                rp_count += 1


        ### END for loop
        if img_plot == 1:

            draw()
            show(block=False)

        elif img_plot == 2:

            draw()
            savefig(img_path + file_name + '_' + str(index+1)+ '.png', dpi=100)
            # close()
            # figure()
            # imshow(pic_bin[0, :, :], interpolation=None, norm = None)
            # draw()
            # savefig(img_path + file_name +'_patch' + '_' + str(index+1)+ '.png', dpi=100)
            print "Image has been saved : " + img_path + file_name + '_' + str(index+1)+ '.png'

        elif img_plot == 3:

            draw()
            show(block=False)

        print str(index+1), "-th jump image...",
        if (index+1) != 1 and remainder(index+1,5) == 0 : print '\n',

        total_bin[index-1, :, :, :] = pic_bin


    print "\n"

    if img_plot in (0,2):
        close("all")


    return total_bin

def RQA_eval(x_in, TR, RPpow, img_process):

    RQA_mat_L = zeros((TR, TR))

    # Set L,R matrix to compute Recurrence matrix
    RQA_mat_L[0:TR, :] = x_in
    RQA_mat_R = RQA_mat_L.transpose()

    # RQA_value = abs(RQA_mat_L - RQA_mat_R)**2

    # RQA_value -= mean(RQA_value)
    # RQA_value /= std(RQA_value)

    if img_process == -1:

	# Subtract two matirices
	    RQA_value = RQA_mat_L - RQA_mat_R

    if img_process == 0:

        # RQA_value = RQA_value/amax(abs(RQA_value))
        # Subtract two matirices
    	RQA_value = abs(RQA_mat_L - RQA_mat_R)

        # RQA_value = RQA_value.clip(min=0.5, max=0.5)

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
