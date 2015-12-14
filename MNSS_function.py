__author__ = 'mac'


import librosa
import librosa.decompose
from matplotlib.colors import LogNorm
import scipy.io.wavfile
from pylab import *
from numpy import *
import numpy as np
import scipy.io.wavfile
import scipy.linalg
from scikits.samplerate import resample
import pickle
import math
from random import *

# from xp00_s import NN_judge
from NN_wrapper import NN_judge2
from random import random
'''
TF domain Music Noise Segmentation

'''

# Convert Sampling Rate
def wav_convert(data, SR, tar_freq):

    # If the input signal is stereo, make it mono.
    if ndim(data) == 2:

        # Mix stereo signal into a mono signal
        buff01 = 0.49 * (data[:, 0] + data[:, 1])
        wave_ts = array(buff01)

    else:
        wave_ts = array(data[:])

    wave_ts = array(wave_ts)

    # Set a sampling rate
    up_SR = 44100

    # Compute a ratio to feed into resample function
    ratio = float(float(tar_freq)/float(up_SR))

    # Resample the file.
    wave_ts = resample(wave_ts, ratio , 'linear')

    # Transpose the data list.
    wave_ts = transpose(wave_ts)

    # Return wave_ts signal
    return wave_ts




def MNSS_function(file_names):

    ##
    SN = 1

    file_total = file_names[0]
    file_total_n = file_names[1]
    file_total_m = file_names[2]

    # # Set files for input data.
    # file_total = wav_folder + sls + inst_name + sls + fn_for
    # file_total_n = wav_folder + sls + inst_name_n + sls + fn_for_n
    # file_total_m = wav_folder + sls + inst_name_m + sls + fn_for_m

    print "wav file path: ", file_total
    print "wav file path_n: ", file_total_n
    print "wav file path_m: ", file_total_m

    # Use scipy.io library to read wave signal.
    SR, data = scipy.io.wavfile.read(file_total)
    SR, data_n = scipy.io.wavfile.read(file_total_n)
    SR, data_m = scipy.io.wavfile.read(file_total_m)

    tar_freq = 44100

    data_ts = wav_convert(data, SR, tar_freq )
    data_ts_n = wav_convert(data_n, SR, tar_freq )
    data_ts_m = wav_convert(data_m, SR, tar_freq )


    ## The log2 of RQA plot image size
    RP = 6 ## 32 by 32
    img_W = 4

    ## Feature image size for ANALYZE
    # Unit window size
    img_sz_org = [(2**RP), img_W*(2**RP)]

    # Total window size
    img_sz = [4*(2**RP), img_W*(2**RP)]

    ## FFT specification
    NFFT = 2*img_sz[0]
    HOP_LENGTH = NFFT/4


    ## Set size of dimension
    depth = 8
    stride = img_sz_org[0]/4
    x_stride = img_sz_org[0]/4

    D_org = np.abs(librosa.stft(data_ts, n_fft = NFFT, hop_length = HOP_LENGTH))

    # Show original spectrogram
    figure()
    img = librosa.display.specshow(D_org, sr=SR, cmap = 'jet', y_axis='linear', x_axis='time', norm=LogNorm(vmin=0.01, vmax=10))

    subplot(depth ,1 ,1)

    seg_sig = data_ts
    D = np.abs(librosa.stft(seg_sig, n_fft = NFFT, hop_length = HOP_LENGTH))
    D_n = np.abs(librosa.stft(data_ts_n, n_fft = NFFT, hop_length = HOP_LENGTH))
    D_m = np.abs(librosa.stft(data_ts_m, n_fft = NFFT, hop_length = HOP_LENGTH))

    time_length = int( (shape(D)[1] - img_sz_org[1]) /float(x_stride))

    X_test = np.zeros((time_length*(depth+1), 1, img_sz_org[0], img_sz_org[1]), dtype="float32")
    y_test = np.zeros((time_length*(depth+1), 1, 1), dtype="float32")

    for r in range(0, time_length):
        x_offset = r * x_stride
        # print 'x_offset', x_offset
        for k in range(0, depth+1):

            offset = stride * k

            x_start = x_offset + 0
            x_end = x_offset + img_sz_org[1]
            y_start = (offset + 0)
            y_end = (offset + img_sz_org[0])# /float(SR)

            Ds = D[ y_start : y_end, x_start : x_end ]

            X_test[r*(depth+1)+k,0,:,:] = abs(Ds)

            subplot(depth+1, 1, (depth+1)-k)
            img = librosa.display.specshow(Ds, sr=SR, y_axis='linear', cmap = 'jet', x_axis='time', norm=LogNorm(vmin=0.01, vmax=10))

    ## Dicern whether the frames are noise or music
    ##################################
    pr_val = NN_judge2(X_test, y_test)
    ##################################

    pr_val_raw = (pr_val + 1e-10)
    pr_val = log10(pr_val + 1e-10)

    pr_val_box_music = pr_val[:,0].reshape((depth+1), time_length)
    pr_val_box_noise = pr_val[:,1].reshape((depth+1), time_length)

    pr_val_box_music_raw = pr_val_raw[:,0].reshape((depth+1), time_length)
    pr_val_box_noise_raw = pr_val_raw[:,1].reshape((depth+1), time_length)

    D_mask_m = 0*D[0:(img_sz_org[0]*3), 0:stride*time_length ]
    D_mask_n = 0*D[0:(img_sz_org[0]*3), 0:stride*time_length ]

    D_mask_mb = 0*D[0:(img_sz_org[0]*3), 0:stride*time_length ]
    D_mask_nb = 0*D[0:(img_sz_org[0]*3), 0:stride*time_length ]


    for r in range(0, time_length):
        x_offset = r * x_stride
        for k in range(0, depth+1):
            offset = stride * k

            x_start = x_offset + 0
            x_end = x_offset + img_sz_org[1]
            y_start = (offset + 0)
            y_end = (offset + img_sz_org[0])# /float(SR)

            D_mask_m[ y_start : y_end, x_start : x_end  ] = \
                D_mask_m[ y_start : y_end, x_start : x_end  ] + pr_val_box_music[k, r]
            D_mask_n[ y_start : y_end, x_start : x_end  ] = \
                D_mask_n[ y_start : y_end, x_start : x_end  ] + pr_val_box_noise[k, r]

            D_mask_mb[ y_start : y_end, x_start : x_end  ] = \
                D_mask_m[ y_start : y_end, x_start : x_end  ] + pr_val_box_music_raw[k, r]
            D_mask_nb[ y_start : y_end, x_start : x_end  ] = \
                D_mask_n[ y_start : y_end, x_start : x_end  ] + pr_val_box_noise_raw[k, r]

    for r in range(0, time_length):
        x_offset = r * x_stride
        for k in range(0, depth+4):
            offset = stride * k

            x_start = x_offset + 0
            x_end = x_offset + img_sz_org[0]/4
            y_start = (offset + 0)
            y_end = (offset + img_sz_org[0]/4)# /float(SR)

            k_com_L = 4/float(amin([k+1, 4]))
            k_com_H = 4/float(amin([(depth+4-k), 4]))

            D_mask_m[ y_start : y_end, x_start : x_end  ] = \
                D_mask_m[ y_start : y_end, x_start : x_end  ]*float(float(k_com_L)*float(k_com_H))

            D_mask_n[ y_start : y_end, x_start : x_end  ] = \
                D_mask_n[ y_start : y_end, x_start : x_end  ]*float(float(k_com_L)*float(k_com_H))


    D_show = D[0:(img_sz_org[0]*3), 0:stride*time_length ]
    D_show_n = D_n[0:(img_sz_org[0]*3), 0:stride*time_length ]
    D_show_m = D_m[0:(img_sz_org[0]*3), 0:stride*time_length ]

    D_show = flipud(D_show)
    D_show_n = flipud(D_show_n)
    D_show_m = flipud(D_show_m)

    D_mask_dn = flipud(D_mask_n)
    D_mask_dm = flipud(D_mask_m)

    D_music_m = (D_mask_dm - D_mask_dn)
    D_soft_m = D_music_m
    D_music_m[D_music_m>0]=1
    D_music_m[D_music_m<=0]=0

    D_noise_n = (D_mask_dn - D_mask_dm)
    D_soft_n = D_noise_n
    D_noise_n[D_noise_n>0]=1
    D_noise_n[D_noise_n<=0]=0

    D_noise_n = (D_mask_dn - D_mask_dm)


    total_plot = 7

    with open( 'pickle_folder/' + 'MNN_visual_data_'+ str(SN) + '.pickle', 'w') as f:
        pickle.dump([D_show, D_show_n, D_show_m, D_mask_dn, D_mask_dm, D_music_m, D_noise_n], f)


    subplot(total_plot, 1, 1)
    img = librosa.display.specshow(flipud(D_show_n), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))

    subplot(total_plot, 1, 2)
    img = librosa.display.specshow(flipud(D_show_m), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))

    subplot(total_plot, 1, 3)
    img = librosa.display.specshow(flipud(D_show), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))

    subplot(total_plot, 1, 4)
    imshow(D_mask_dn, interpolation = 'none', cmap = 'jet' )

    subplot(total_plot, 1, 5)
    imshow(D_mask_dm, interpolation = 'none', cmap = 'jet' )

    subplot(total_plot, 1, 6)
    imshow(log10(np.array(D_show)*np.array(D_noise_n)), interpolation = 'none', cmap = 'jet' )

    subplot(total_plot, 1, 7)
    imshow(log10(np.array(D_show)*np.array(D_music_m)), interpolation = 'none', cmap = 'jet' )

    show()

    # with open(pickle_path + open_data + '.pickle') as f:
    #     sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes = pickle.load(f)

    seg_sig = data_ts
    D_out = librosa.stft(seg_sig, n_fft = NFFT, hop_length = HOP_LENGTH)
    shape_Dout = shape(D_out)
    D_wr = 0 * D_music_m

    shape_D = shape(D_music_m)
    D_wr = zeros((shape_D[0],shape_D[1]), dtype=complex)
    D_wr[0:shape_D[0], 0:shape_D[1] ] = D_out[ 0:shape_D[0], 0:shape_D[1] ]

    D_n_out = np.array(D_wr)*np.array(D_noise_n)
    D_m_out = np.array(D_wr)*np.array(D_music_m)

    D_yn = zeros((shape_Dout[0],shape_D[1]), dtype=complex)
    D_ym = zeros((shape_Dout[0],shape_D[1]), dtype=complex)

    D_yn[0:shape_D[0], 0:shape_D[1] ]  = D_n_out
    D_ym[0:shape_D[0], 0:shape_D[1] ]  = D_m_out

    y_noise = librosa.istft(D_n_out)
    y_music = librosa.istft(D_m_out)

    file_total_wr = 'result_data'
    sls = '/'
    print 'SR', SR
    print 'tar_freq', tar_freq
    SR_w = 33000

    scipy.io.wavfile.write(file_total_wr + sls + 'y_music.wav', SR_w, y_music)
    scipy.io.wavfile.write(file_total_wr + sls + 'y_noise.wav', SR_w, y_noise)
    show()

    print 'max and min of pvb music', amax(pr_val_box_music), amin(pr_val_box_music)
    print 'max and min of pvb noise', amax(pr_val_box_noise), amin(pr_val_box_noise)
    print 'max and min of pr_val', amax(pr_val), amin(pr_val)
    print 'shape of pr_val_box_music ', shape(pr_val_box_music)
