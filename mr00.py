 


from __future__ import absolute_import
# from __future__ import print_function
import numpy as np

from pylab import *
from numpy import *
import scipy.io.wavfile
import scipy.linalg
import scipy.signal as sig
import RQA_func2
import pickle
from scikits.samplerate import resample
import finish_alarm
from tts import *
import adjspecies
from time import *
from requests.exceptions import ConnectionError
import matplotlib as plt

'''
    Edited : 7:01, 7, Sep, 2015
    This program should be grouped with the followings :

    RQA_cnn.py (Feature Extractor Launcher)
    RQA_func2.py (Python file that includesFeature Extractor Function)
    RQA_cnn_test.py (Keras Launcher)
    test_set.py (Python file that includes Test Set Divider Function)
    set_file.py (Test Set Divider Launcher)

'''

## Random seed for reproducibility.
seed_n = 0
# np.random.seed(1337)
np.random.seed(seed_n)



## The class for equalizing test set.
## It's still random, but equally distributes test samples in terms of instrument kind.
class Hist_rand:

    def __init__(self):
        self.hist = array([0]*10)
        self.non_eq = array([0]*10)

        self.idx = array(range(len(self.hist )))


    def add_num(self, rand):

        self.temp = self.hist
        self.non_eq[rand-1] += 1

        self.min_hist_vals = amin(self.hist)
        min_L = self.idx[(self.hist == self.min_hist_vals)]

        idx_s = randint(0,len(min_L))
        dist_rn = min_L[idx_s] + 1

        self.hist[dist_rn-1] += 1

        if remainder(sum(self.hist),30) == 0:
            print "Balanced random sequence   :", self.hist
            print "Unbalanced random sequence :", self.non_eq
        return dist_rn

        # if self.hist[k_bool-1] in min_L:
        #     print '

## wav_process :
## 1. Normalize the sample.
## 2. Insert silence at the end of the sample if the sample is too short.
## 3. Cut out the silence at the starting point.

def num10(num):

    if remainder(num,10) == 0 :
        p_num = 10
    else :
        p_num = remainder(num,10)

    return p_num


def wav_process(wav_in, th, min_length, min_margin):

    # Frame size
    FR = 32

    ## Normalization
    wav_in = wav_in / float(max(abs(wav_in)))

    ## Cut the silence part out if it is below the threshold.
    qt1 = 1
    for s in range(0, len(wav_in)):
        # print abs(wav_in[1:min_length])


        # if abs(wav_in[s]) > th and s > 0 :
        if s > (FR-1):

            if abs( max(wav_in[(s-FR+1):s+1]) - min(wav_in[(s-FR+1):s+1]) ) > th:

                margin = min( min_margin,s )
                wav_in = wav_in[s - margin:]

                break

    ## In case of short sample, insert silence
    if len(wav_in) < min_length:

        ZM = zeros(max(0, (min_length - len(wav_in)),))
        wav_in = concatenate((wav_in, ZM ))
        # print "wave is too short, shaped as ", shape(wav_in)

    elif len(wav_in) >= min_length:
        wav_in = wav_in[0:min_length]


    ## Return wav_in
    return wav_in

def wav_convert(data, SR, tar_freq):

    # If the input signal is stereo, make it mono.
    if ndim(data) == 2:
        buff01 = 0.49 * (data[:, 0] + data[:, 1])
        wave_ts = array(buff01)
    else:
        wave_ts = array(data[:])

    wave_ts = array(wave_ts)
    # print "Shape of wave_ts", shape(wave_ts)

    up_SR = 44100
    ratio = float(float(tar_freq)/float(up_SR))
    print tar_freq
    print float(float(tar_freq)/float(up_SR))
    wave_ts = resample(wave_ts, ratio , 'linear')

    # SR_div = int(floor(up_SR/tar_freq))
    # wave_ts = sig.decimate(wave_ts, SR_div)

    # Transpose the data list.
    wave_ts = transpose(wave_ts)
    print 'length wave_ts', shape(wave_ts)

    return wave_ts


t_start = clock()
## Folder for output RP feature images
img_path = "rqa_img/"

FLS_txt = "file_list.txt"

# file_path = "audio_sample/"
file_path = "audio_sample_old/"

file_ex = ".wav"

sls = "/"
wav = "wav"


# folder_name = ["Bass", "Bassoon", "Bells", "Cello", "Clarinet", "Flute", "Guitar", "Horn", "Marimba",
#                "Oboe", "Piano", "Sax", "Trombone", "Trumpet", "", "Vibraphone", "Viola", "Violin", "Xylophone"]
# folder_name = ["Bass", "Bassoon", "Bells", "Cello", "Clarinet"]

# folder_name = ["Piano", "Tuba", "Trumpet", "Horn", "Ttrombone",
#                "Btrombone", "Violin", "Viola", "Bass", "Cello", "Sax",
#                "Altosax","Oboe", "Bassoon", "Flute", "Altoflute",
#                "Bflute", "Bclarinet", "Bbclarinet", "Ebclarinet"]

# folder_name = ["Violin"]

# folder_name = ["Piano", "Bass", "Clarinet"]
wav_folder = "music_det"
# wav_folder = "music_inst_wav_fake"

folder_name = [["Piano"],

               ["Tuba", "Trumpet", "Horn", "Ttrombone", "Btrombone"],

               ["Violin", "Viola", "Bass", "Cello"],

               ["Sax","Altosax","Oboe", "Bassoon", "Flute", "Altoflute",
               "Bflute", "Bclarinet", "Bbclarinet", "Ebclarinet"],

               ["Piano", "Tuba", "Trumpet", "Horn", "Ttrombone",
               "Btrombone", "Violin", "Viola", "Bass", "Cello", "Sax",
               "Altosax","Oboe", "Bassoon", "Flute", "Altoflute",
               "Bflute", "Bclarinet", "Bbclarinet", "Ebclarinet"],

               ["Piano", "Tuba", "Trumpet", "Horn", "Ttrombone",
               "Btrombone", "Violin", "Viola", "Bass", "Cello", "Sax",
               "Altosax","Oboe", "Bassoon", "Flute", "Altoflute",
               "Bflute", "Bclarinet", "Bbclarinet", "Ebclarinet"],

               ["music", "noise"]]
               # ["noise"]]
               # ["mix_test_p20"]]
family = 6
data_tag = '_music_noise_'
# Make two classes
force_class = 0


ss_mode = 0

#
# folder_name = [["piano_gen_ff", "piano_gia_ff", "piano_mav_ff", "piano_grd_ff"]]
# wav_folder = "piano_bin"
# family = 0

# folder_name = [["piano_gen_ff", "piano_gia_ff", "piano_mav_ff", "piano_grd_ff"]]
# wav_folder = "piano_bin"
# family = 0


## Sampling Rate for RP plot
tar_freq = 44100/4

## No image plot if 0, Image plot if 1,
## image saving only if 2, Do both if 3
img_plot = 0

## The log2 of RQA plot image size
RP = 6 ## 32 by 32

## Feature image size
img_sz = 2**RP

NFFT = img_sz*2
HOP_LENGTH = NFFT / 4
## jump : The length
# jump = TR[-3]
jump = HOP_LENGTH*img_sz + NFFT-HOP_LENGTH # standard
# jump = 10 # standard
# jump = 17


## Jump count
jump_num = int(20*(tar_freq)/jump)
print 'jump_num', jump_num


pickle_path = 'pickle_folder/'
# pickle_file = 'class13_' + 'data_j' + str(jump_num)+'_01010101_' + 'SR48kHz'

data_note = '### Mean Sub.  \n' + \
            '### No Align\n'


## RP image matrix : set the number of image for each level.
# RPs_in_level = [1, 1, 1, 2,   4, 8, 16, 32]
# RPs_in_level = [1, 1, 1, 1,   2, 4, 8, 16]
# RPs_in_level = [0, 1, 0, 2,   0, 8, 0, 32]
# RPs_in_level = [0, 1, 0, 1,   0, 1, 0, 1]
# RPs_in_level = [0, 1, 0, 6,   0, 6, 0, 6]
# RPs_in_level = [0, 3, 0, 3, 0,  3, 0, 3, 0, 3]
# RPs_in_level = [0 ,1, 0, 1, 0, 3, 0, 3, 0, 3]
# RPs_in_level = [0, 0, 2, 2, 2,  2, 2, 2, 2, 2]
# RPs_in_level = [0 ,1, 0, 5, 0, 5, 0, 5, 0, 5]
# RPs_in_level = [0 ,1, 0, 4, 0, 4, 0, 4, 0, 4]
# RPs_in_level = [0 ,1, 0, 4, 0, 4, 0, 4, 0, 4]
# RPs_in_level = [2, 0, 2, 0, 2, 0, 2, 0, 2, 2]

RR = 1
RPs_in_level = [RR,   0, 0, 0, 0,   0, RR, 0, RR,   0, RR, 0, RR]
RPs_in_level = [RR,   0, RR, 0, RR,   0, RR, 0, RR,   0, RR, 0, RR]
RPs_in_level = [0,   0, 0, 0, RR, 0, 0, 0, 0,   0, 0, 0, 0]
# RPs_in_level = [1]

# RPs_in_level = [0, 0, 1,   0, 10, 0, 10,   0, 10, 0, 10]


RPpow = 0.5
print 'RPpow: ' + str(RPpow)

# img_process
# -1 = no absolute value
# 0 = pow
# 1 = hist EQ
# 2 = AdaHist

img_process = 0

RP_list = ''
for k in RPs_in_level : RP_list = RP_list + str(k)

rand_animal = adjspecies.random_adjspecies()
# pickle_file = 'dt20_j' + str(jump_num)+'_' + RP_list + '_' + 'SR' + str(int(tar_freq/1000)) + 'kHz_' + rand_animal
pickle_file = 'NNM_cla_'+ 'ss_'+ str(ss_mode)+ '_' + data_tag +'_j' + str(jump_num) +'_' + RP_list + '_' + 'SR' + str(int(tar_freq/1000)) + 'kHz_' + rand_animal
# pickle_file = 'data_j' + str(jump_num)+'_01010101_' + 'SR12kHz' + '_NOA' +'_dm400'
# pickle_file = 'test_mat'
print 'pickle file name : ' + pickle_file

speak = 0

## Sample interval for each level
# RPs_interval = [8*2048, 4*2048,   2*2048, 2048, 1024, 512,     256, 128, 64, 32]
# RPs_interval = [8*2048, 4*2048, 2*2048, 24, 1024, 256, 256, 64, 64]
# itv = 64
# RPs_interval = [512, 512, 512, 512, 512, 512, 512, 512]
# RPs_interval = [itv, itv, itv, itv, itv, itv, itv, itv, itv, itv]
# itv_stride = [[32],[64],[128],[256],[512],[1024],[2048],[4096]]
# itv_stride = [[32],[64],[128],[512],[1024],[4096],[4096*2],[4096*4]]
itv = 64


## Stride offset
SO1 = 512
SO2 = SO1 + 2048
SO3 = SO2 + 2048

## Stride offset
SO1 = 128
SO2 = SO1+itv*5
SO3 = SO2+itv*5



# itv_stride = [[SO1],[SO1+itv],[SO1+itv*2],[SO1+itv*3],[SO1+itv*4],
#               [SO1+itv*5],[SO1+itv*6],[SO1+itv*7],[SO1+itv*8], [SO1+itv*9] ]
# itv_stride = [[SO1+itv*1],[SO1+itv*2],[SO1+itv*3],[SO1+itv*4],[SO1+itv*5],
#               [SO2+itv*1],[SO2+itv*2],[SO2+itv*3],[SO2+itv*4],[SO2+itv*5],
#               [SO3+itv*1],[SO3+itv*2],[SO3+itv*3],[SO3+itv*4],[SO3+itv*5] ]
itv_stride = [[0],[64],[128],[256],[512],    [1024],[2048],[2048*2],[2048*4],[2048*8],  [2048*16],[2048*32],[2048*64],[2048*128],[2048*256]]
# itv_stride = [[0],[64],[128],[256],[512],    [1024],[2048],[2048*2],[2048*4],[2048*8],  [2048*16],[2048*32],[2048*64],[2048*128],[2048*256]]
# itv_stride = [[0],[128],[128*2],[128*3],[128*4],    [512*2],[512*3],[512*4],[512*5],[512*6]]
# itv_stride = [[0],[16],[32],[64],[128],     [512], [1024],[2048],[2048*2]]
# itv_stride = [[0],[64],[128],[512],[1024],  [2048],[2048*2],[2048*4], [2048*8]]
# itv_stride = [[0], [256*1],[256*2],[256*3],[256*4],[256*5],[256*6],[256*7]]
# itv_stride = [[32],[2048],[4096],[512],[1024],[4096],[4096*2],[4096*4]]

# itv_stride = [[0],[128],[256],[512],[1024],[4096],[4096*2],[4096*4]]

len_RPs = len(RPs_in_level)


RPs_interval = [itv_stride[0]*len_RPs,itv_stride[1]*len_RPs,itv_stride[2]*len_RPs,itv_stride[3]*len_RPs,itv_stride[4]*len_RPs,
                itv_stride[5]*len_RPs,itv_stride[6]*len_RPs,itv_stride[7]*len_RPs,itv_stride[8]*len_RPs,itv_stride[9]*len_RPs,
                itv_stride[10]*len_RPs,itv_stride[11]*len_RPs,itv_stride[12]*len_RPs,itv_stride[13]*len_RPs, itv_stride[14]*len_RPs]

# Minimum margin for on-set character
div_margin = 300
min_margin = int( floor( tar_freq / div_margin) )


# ## inst_index : simple index (1,2,3, ... )
# inst_index = range(1,len(folder_name[family])+1)

## Count of classification class
nb_classes = len(folder_name[family])

## inst_index : Grouping
# inst_index = [0, 1, 2, 3, 4,    5, 6, 7, 8, 9,    10, 11, 12, 13, 14,    15, 16, 17, 18, 19 ]

if family == 5 :
    inst_index = [0,   1, 1, 1, 1, 1,    2, 2, 2, 2,   3, 3, 3, 3, 3,   3, 3, 3, 3, 3 ]
    nb_classes = 4
elif force_class == 1:
    inst_index = [0,   1]
    nb_classes = 2
else:
    inst_index = range(0, nb_classes)



## The number Of Time Series Level
NTL = len_RPs + 5 - RP

## The log2 length of Longest RQ Window (LW) (e.g. 12 means 2**12)
## Max should be always 12
LW = RP + NTL - 1
# LW = 12

## Folding Ratio
FR = []

## TR : The log2 length of the RQA plot.
TR = []

## Setup the TR and FR matrix
for tc in range(1, (NTL+1)):

    TR.append(2**(RP + tc - 1))
    FR.append((RP - LW + tc - 1))

TR.reverse()
print "FR, TR", FR, TR
print "nb_classes is :", nb_classes
print "inst_index", inst_index


## Threshold for RQA plot.
th = 0.05

## Threshold for time series.
th_ts = 0.25

## Draw a plot for the given audio sample.
# FN = sum(RPs_in_level)
if ss_mode ==1 :
    FN = 2
elif ss_mode == 0 :
    FN = 1
## Gain for time series.
gain = 1

## Interpolation Mode
I_mode = 2

fileN = []
file_rs = []



## Offset : the starting point in time series
offset = tar_freq * 0.0
# offset = 512

ct1 = 1

## First loop for the instruments in "folder_name[family]"
start_t = clock()
sample_count = 0
for inst_name in folder_name[family]:

    ## Path for file_list.txt, which contains a sample list.
    total_path = wav_folder + sls + inst_name + sls + FLS_txt
    print "file_list_path:", total_path

    fileN = []

    ## A loop for file list generation.
    open_ct = 0
    with open(total_path) as fp:
        for line in fp:
            open_ct += 1
            
            fileN.append(line[:-1])

            # #### Cut the for loop for test purpose
            # if open_ct > 20:
            #     break

    ## Initialize data matrices
    X_data = []
    y_data = []
    k_data = []
    # D_data = []

    ct2 = 1

    eq_rnd = Hist_rand()


    for fn_for in fileN:

        sample_count += 1

        file_name = fn_for
        file_total = wav_folder + sls + inst_name + sls + fn_for
        # print "wav file path: ", file_total

        # Use scipy.io library to read wave signal.
        SR, data = scipy.io.wavfile.read(file_total)

        data_ts = wav_convert(data, SR, tar_freq )

        data_ts = wav_process(data_ts, th_ts, tar_freq*30, min_margin )

        ## Write an audio output file.
        # scipy.io.wavfile.write('audio_sample/temp_wavout.wav', new_SR, data_ts)

        clim = 1

        ## Pack in all the varialbes into the tuple "var_in".
        var_in = (file_name, data_ts, tar_freq, offset, jump_num, TR, FR, jump, FN, RPpow, img_process,
        th, th_ts, gain, clim, I_mode, img_sz, img_plot, img_path, RPs_in_level, RPs_interval)

        ## ############### CALL MR_RQA FUNCTION ###############
        X_data_buff = RQA_func2.MR_RQA(*var_in)
        ## ############### CALL MR_RQA FUNCTION ###############

        ## Instrument Class : y_bool
        y_bool= int(inst_index[ct1-1])*ones((jump_num,1))

        ## Random seed numbers for divding train, valid, and test sets
        # k_bool = randint(1,11)*ones((jump_num,1))
        k_bool = eq_rnd.add_num(randint(1,11))*ones((jump_num,1))

        ## File name data
        # D_bool = array((jump_num, 0), dtype='string')
        # D_bool = array((10, 1), dtype="string")
        D_bool = ['None']

        for qs in range(0,jump_num):
            if qs == 0:
                D_bool = file_name[:-4] + "_J_" + str(qs)
            else:
                D_bool = vstack( (D_bool, file_name[:-4] + "_J_" + str(qs) ))


        # print y_bool, k_bool, D_bool
        ## Assign datasets to the output variables.
        if ct2 == 1:

            X_data = X_data_buff
            y_data = y_bool
            k_data = k_bool
            D_data = D_bool

        elif ct2 > 1:

            X_data = vstack((X_data, X_data_buff))
            y_data = vstack((y_data, y_bool))
            k_data = vstack((k_data, k_bool))
            D_data = vstack((D_data, D_bool))


        print '\n', fn_for, "has been saved to matrix"

        ct2 += 1
        stop_t = clock()
        # print "Instrument level shape :", shape(X_data), shape(y_data)
        elap_t = stop_t - start_t
        ETA = int(1742*float( elap_t / sample_count )/60)
        ETCA = int(elap_t / 60)
        print 'The Estimated Time : ', str(ETCA),'m','/',str(ETA),'m', 'Sample Count :', str(sample_count)

    ## Assign matrix for final output matrix
    if ct1 == 1:
        sum_mat_X_data = X_data
        sum_mat_y_data = y_data
        sum_mat_k_data = k_data
        sum_mat_D_data = D_data

    elif ct1 > 1:
        # sum_mat_X_data = concatenate((sum_mat_X_data, X_data))
        sum_mat_X_data = vstack((sum_mat_X_data, X_data))
        sum_mat_y_data = vstack((sum_mat_y_data, y_data))
        sum_mat_k_data = vstack((sum_mat_k_data, k_data))
        sum_mat_D_data = vstack((sum_mat_D_data, D_data))

    ct1 += 1

## Convert all the numbers in the list to integer.
sum_mat_y_data = [[int(i)] for i in sum_mat_y_data]
sum_mat_k_data = [[int(i)] for i in sum_mat_k_data]

## Check out the matrix size.
print shape(sum_mat_X_data), shape(sum_mat_y_data), shape(sum_mat_k_data), shape(sum_mat_D_data)


with open(pickle_path + pickle_file + '.pickle', 'w') as f:
    pickle.dump([sum_mat_X_data, sum_mat_y_data, sum_mat_k_data, sum_mat_D_data, jump_num, FN, nb_classes], f)


# with open(pickle_file + '_spec.pickle', 'w') as f:
#     pickle.dump([var_in], f)

file = open(pickle_path + "data_spec/" + pickle_file + ".txt", "w")

file.write(pickle_file +'\n\n')
file.write('================================================\n\n')

# var_in = (file_name, data_ts, tar_freq, offset, jump_num, TR, FR, jump, FN, RPpow, img_process,
#            th, th_ts, gain, clim, I_mode, img_sz, img_plot, img_path, RPs_in_level, RPs_interval)

vars_str = ["file_name", "data_ts", "tar_freq", "offset", "jump_num", "TR", "FR", "jump", "FN", "RPpow", "img_process",
            "th", "th_ts", "gain", "clim", "I_mode", "img_sz", "img_plot", "img_path", "RPs_in_level", "RPs_interval", "ss_mode"]

for k, var in enumerate(var_in):
    file.write( vars_str[k] + ":"+ str(var) +"\n")

file.write(  "random seed" + ":" + str(seed_n) + "\n")
file.write(  "div_margin" + ":" + str(div_margin) + "\n")
file.write(  "min_margin" + ":" + str(min_margin) + "\n")
file.write(  "nb_classes is :"+str(nb_classes)+"\n")
file.write(  "inst_index"+str(inst_index)+"\n")
file.write(  "itv_stride"+str(itv_stride)+"\n\n")
file.write(  "Instruments(folder_name) : " + str(folder_name[family]) + "\n" )
file.write( data_note +'\n')
file.close()


# print sum_mat_D_data
t_stop = clock()
elapsed_time = int(int(t_stop - t_start)/60)
print 'Feature extraction process for ' + rand_animal + ' is complete'
print 'Elapsed time is :'+ str(elapsed_time) +' minutes '
print pickle_file + ' has been finished.'


if speak == 1:
    try:
        finish_alarm.ring('bell01')
        speak_str('Feature extraction process for ' + rand_animal + ' is complete.')
        speak_str('Elapsed time is :'+ str(elapsed_time) +' minutes ')
    except ConnectionError as e:
        print ('ConnectionError! Check the Wifi Connection. \n')

print 'sample count:', sample_count
