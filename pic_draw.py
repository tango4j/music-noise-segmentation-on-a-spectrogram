
from __future__ import absolute_import
# from __future__ import print_function
import numpy as np

from pylab import *
from numpy import *
import scipy.io.wavfile
import scipy.linalg
import matplotlib.gridspec as gridspec
from time import *
import scipy.signal as sig
import RQA_func2
import pickle
from scikits.samplerate import resample
import finish_alarm
from tts import *
import adjspecies
import librosa
from matplotlib.colors import LogNorm

def wav_convert(data, SR, tar_freq):

    # If the input signal is stereo, make it mono.
    if ndim(data) == 2:
        buff01 = 0.49 * (data[:, 0] + data[:, 1])
        wave_ts = array(buff01)
    else:
        wave_ts = array(data[:])

    wave_ts = array(wave_ts)
    # print "Shape of wave_ts", shape(wave_ts)

    up_SR = tar_freq/SR

    wave_ts = resample(wave_ts, 1 , 'linear')
    #
    # SR_div = int(floor(up_SR/tar_freq))
    # wave_ts = sig.decimate(wave_ts, SR_div)

    # Transpose the data list.
    wave_ts = transpose(wave_ts)
    return wave_ts

def RQA_eval(x_in, TR, th):

    RQA_mat_L = zeros((TR, TR))

    # Set L,R matrix to compute Recurrence matrix
    RQA_mat_L[0:TR, :] = x_in
    RQA_mat_R = RQA_mat_L.transpose()

    # Subtract two matirices
    RQA_value = abs(RQA_mat_L - RQA_mat_R)

    return RQA_value

def RQA_eval_sq(x_in, TR, th):

    RQA_mat_L = zeros((TR, TR))

    # Set L,R matrix to compute Recurrence matrix
    RQA_mat_L[0:TR, :] = x_in
    RQA_mat_R = RQA_mat_L.transpose()

    # Subtract two matirices
    RQA_value = sqrt(RQA_mat_L**2 + RQA_mat_R**2)

    return RQA_value

def max_pooling1d(image, pool_size):

    dim_img = shape(image)

    pooled_img = zeros((int(dim_img[0]/pool_size),), dtype=float32)

    print 'received pool size', pool_size
    print 'trying to pool:', str(dim_img[0]/pool_size)

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
def max_pooling2d(image, pool_size):

    dim_img = shape(image)

    pooled_img = zeros(( int(dim_img[0]/pool_size), int(dim_img[0]/pool_size) ), dtype=float32)

    for p in range(0, dim_img[0]/pool_size):
        for q in range(0, dim_img[1]/pool_size):

            set = image[ p*pool_size: (p+1)*pool_size ,q*pool_size : (q+1)*pool_size  ]
            cal = amax( reshape(set, (pool_size**2, 1)) )

            pooled_img[p, q] = cal

        ## FL2 END
    ## FL1 END

    return pooled_img

def PCA_RP(RPrs_repos):

    X = RPrs_repos
    dim = shape(X)

    X_mean = mean(X, axis = 0)
    print dim
    X -= mean(X, axis = 0)

    print 'dim of X', shape(X)
    cov = dot(X.T, X) / X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot = U[:,1]
    # Xrot = np.dot(X, U)
    # Xrot_reduced = np.dot(X, U[:,:100])
    return Xrot, U,S,V, X_mean

def PCA_RP_stack(PCA_fsout, period, p_num):

    jump = 2 * period

    shift = 0

    print 'len of PCA_fsout', len(PCA_fsout)

    for k in range(0, p_num):


        shift = shift + jump

        max_window = PCA_fsout[shift + 0 : shift + period]

        print 'len of PCA_fsout', len(max_window)

        max_arg_ts = int(argmax(PCA_fsout))

        ts_in = PCA_fsout[shift + max_arg_ts: (shift + max_arg_ts + period)]

        ts_repos.append(PCA_fsout)

        print 'len of ts_in', len(ts_in)

        RP = (RQA_eval(ts_in, period, 0.0) )

        RP = max_pooling2d(RP, 2**(size_RP-5))

        RP_repos.append(RP)

        if k ==0 :
            RPrs_repos = reshape(RP, 32*32)
        else:
            RPrs_repos = vstack((RPrs_repos, reshape(RP, 32*32) ))

        idx2 = range(0,len(max_window))

        subplot(fig[0,k])
        imshow(RP, interpolation = 'None')
        set_cmap('gray')
        subplot(fig[1,k])
        plot(idx2, max_window)
        title(str(shift+max_arg_ts))

    X = RPrs_repos
    dim = shape(X)

    X_mean = mean(X, axis = 0)
    print dim
    X -= mean(X, axis = 0)

    print 'dim of X', shape(X)
    cov = dot(X.T, X) / X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot = U[:,1]
    # Xrot = np.dot(X, U)
    # Xrot_reduced = np.dot(X, U[:,:100])

    return Xrot, U, S, V, X_mean

close('all')

path = 'audio_sample_old/piano_c3_10.wav'
path = 'audio_sample_old/guitar_c3_06.wav'
path = 'audio_sample_old/Tuba.ff.Gb2.stereo.wav'

SR, data = scipy.io.wavfile.read(path)

data = data[:,0]
data = data / float(max(abs(data)))

tar_freq =44100/2

data = wav_convert(data, SR, tar_freq)
print shape(data)

size_RP = 11
pool = 1

pool2D = 3
period = 2**size_RP

jump = 2*period

off_set = 3940

ts_repos = []
RP_repos = []
RPrs_repos = array([])

p_num = 1

fig_sz = p_num
fig = plt.figure(figsize=(10, 10))
fig = gridspec.GridSpec(2, 2,height_ratios=[0.4,  0.1, 0.3])

idx = range(0,period)

ts = data[(off_set + 0):]
ts_in = data[off_set:off_set+period]

subplot(fig[0,0])
#
# RP = RQA_eval(ts_in, period, 0.0)
# imshow(RP, interpolation = 'None')
# set_cmap('gray')

subplot(fig[1,0])
idx2 = range(0,len(ts_in))
LW = 1.3
plot(idx2, ts_in, linewidth=LW)


# fig = plt.figure(figsize=(10, 10))
# fig = gridspec.GridSpec(2, fig_sz)



size_RP = 9
# pool = 1
#
# pool2D = 3
period = 2**size_RP
idx = range(0,period)
ts_in = data[off_set:off_set+period]

subplot(fig[0,1])
RP = RQA_eval(ts_in, period, 0.0)
imshow(RP, interpolation = 'None')
set_cmap('gray')


subplot(fig[1,1])
idx2 = range(0,len(ts_in))
LW = 3
plot(idx2, ts_in, linewidth=LW)

img_path = 'rqa_img/'
file_name = 'RP images construct '
savefig(img_path + file_name + '.png', dpi=100)

show()



# subplot(fig[2,0])
# n_fft = 1024
# D = np.abs(librosa.stft(ts_in, n_fft = n_fft, hop_length = n_fft/4))
# img = librosa.display.specshow(D, sr=SR, y_axis='log', x_axis='time',norm=LogNorm(vmin=0.01, vmax=10),cmap='gray_r')

# img_path = 'rqa_img/'
# file_name = 'Before 1d Pooling'
# savefig(img_path + file_name + '.png', dpi=100)
#
#
# ts_in = max_pooling1d(ts_in, 2**pool )
# dr = size_RP - pool
#
# new_period = 2**dr
#
#
#
# RP = RQA_eval(ts_in, new_period, 0.0)
#
# fig = plt.figure(figsize=(10, 10))
# fig = gridspec.GridSpec(2, fig_sz)
# subplot(fig[0,0])
# imshow(RP, interpolation = 'None')
# set_cmap('gray')
# subplot(fig[1,0])
# idx2 = range(0,len(ts_in))
# plot(idx2, ts_in, linewidth=LW)
# title('After pooling image ')
#
# img_path = 'rqa_img/'
# file_name = 'After pooling image'
# savefig(img_path + file_name + '.png', dpi=100)
#
# RP  = max_pooling2d(RP, 2**pool )
#
#
# fig = plt.figure(figsize=(10, 10))
# fig = gridspec.GridSpec(2, fig_sz)
# subplot(fig[0,0])
# imshow(RP, interpolation = 'None')
# set_cmap('gray')
# subplot(fig[1,0])
# idx2 = range(0,len(ts_in))
# plot(idx2, ts_in, linewidth=LW)
# title('After 2Dpooling image ')
#
# img_path = 'rqa_img/'
# file_name = 'After 2Dpooling image '
# savefig(img_path + file_name + '.png', dpi=100)




# print shape(ts)
#
# start = clock()
# Xrot, U, S, V, X_mean = PCA_RP_stack(ts, period, p_num )
# stop = clock()




# figure()
# S_cut = S[:10]
# idx2 = range(0,len(S_cut))
# plot(idx2, S_cut)
#
#
# reconst_img = zeros((32*32,))
#
# for q in range(0, 2):
#     reconst_img = reconst_img + U[:,q]*S[q]
#
# reconst_img = reconst_img + X_mean
#
# gg = figure(figsize=(5,3))
# NR = 4
# subplot(1,NR,1)
# X_redu = reshape(X_mean,[32,32])
# imshow(X_redu)
#
# subplot(1,NR,2)
# X_redu = reshape(-U[:,0],[32,32])
# imshow(X_redu)
#
# subplot(1,NR,3)
# X_redu = reshape(U[:,1],[32,32])
# imshow(X_redu)
#
# subplot(1,NR,4)
# X_redu = reshape(reconst_img[:],[32,32])
# imshow(X_redu)
# title('Reconstructed image ')
# show()


# img_L = log2(RQA_eval(p_data,period,0.0) )
#
# img_2 = (RQA_eval(p_data,period,0.0) )**2
# img_sq = (RQA_eval_sq(p_data,period,0.0) )
#
# img = (RQA_eval(p_data,period,0.0) )
# img_n = img
#
# img_n -= np.mean(img_n)
# img_n /= np.std(img_n)
# img = (RQA_eval(p_data,period,0.0) )
#

#
# print len(p_data)
# p_data = data[off_set+max_arg_ts: (off_set+max_arg_ts + period)]
#
# idx = range(0,period)
#
# img_L = log2(RQA_eval(p_data,period,0.0) )
#
# img_2 = (RQA_eval(p_data,period,0.0) )**2
# img_sq = (RQA_eval_sq(p_data,period,0.0) )
#
# img = (RQA_eval(p_data,period,0.0) )
# img_n = img
#
# img_n -= np.mean(img_n)
# img_n /= np.std(img_n)
# img = (RQA_eval(p_data,period,0.0) )
#
# print shape(p_data), shape(idx)
# figure()
# subplot(3,2,1)
# plot(idx,p_data)
# title(str(period)+'Sample location:'+str(off_set+max_arg_ts))
# subplot(3,2,2)
# imshow(img_n)
# title('Normalized')
# set_cmap('gray')
# subplot(3,2,3)
# imshow(img)
# title('Original')
# set_cmap('gray')
# subplot(3,2,4)
# imshow(img_2)
# title('Square')
# set_cmap('gray')
# subplot(3,2,5)
# imshow(sqrt(img))
# title('SQRT')
# set_cmap('gray')
# subplot(3,2,6)
#
# imshow(sqrt(img_sq))
# set_cmap('gray')
# title('L2 Norm')
# show()
