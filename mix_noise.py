__author__ = 'mac'

from pylab import *
from numpy import *
import scipy.io.wavfile
import scipy.linalg
import scipy.signal as sig
import RQA_func_det
import pickle
from scikits.samplerate import resample
import finish_alarm
from tts import *
import adjspecies
from time import *
from requests.exceptions import ConnectionError


## Random seed for reproducibility.
seed_n = 0
# np.random.seed(1337)
np.random.seed(seed_n)

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

wav_folder = 'music_det'
sls = '/'
inst_name_M = 'music_test'
inst_name_N = 'noise_test'
inst_name_mix = 'mix_test_pminf'
FLS_txt = "file_list.txt"


fileN = []
file_total_N = wav_folder + sls + inst_name_N + sls + FLS_txt
## A loop for file list generation.
open_ct = 0
with open(file_total_N) as fp:
    for line in fp:
        open_ct += 1

        fileN.append(line[:-1])

fileM = []
file_total_M = wav_folder + sls + inst_name_M + sls + FLS_txt
## A loop for file list generation.
open_ct = 0
with open(file_total_M) as fp:
    for line in fp:
        open_ct += 1

        fileM.append(line[:-1])

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

sample_count = 0
for fn_for in fileN:

    file_name = fn_for
    file_total_M = wav_folder + sls + inst_name_M + sls + fileM[sample_count]
    file_total_N = wav_folder + sls + inst_name_N + sls + fn_for


    # Use scipy.io library to read wave signal.
    SR, dataM = scipy.io.wavfile.read(file_total_M)
    print "wav file path: ", file_total_M, 'SR', SR, 'shape', shape(dataM)
    dataM = dataM[:,0]/float(amax(dataM[:,0]))
    # Use scipy.io library to read wave signal.
    SR, dataN = scipy.io.wavfile.read(file_total_N)
    dataN = dataN[:,0]/float(amax(dataN[:,0]))
    # print "wav file path: ", file_total_N, 'SR', SR, 'shape', shape(dataN)
    # print 'data check N:', dataN
    # print 'data check power N', power(dataN,2)
    # print 'data check M:', dataM
    # print 'data check power M', power(dataM,2)
    sum_N = sum(power(dataN[0:SR*20],2))
    sum_M = sum(power(dataM[0:SR*20],2))

    print 'ratio:', (float(sum_M)/float(sum_N)), sum_M , sum_N
    mix_dB = -20
    gain =float(float(mix_dB)/20)
    print 'gain DB', float(float(mix_dB)/20)
    data_w = (dataM[0:SR*20]) + float(10)**(float(gain))*sqrt(float(sum_M/sum_N))*(dataN[0:SR*20])
    max_val = amax(data_w)
    data_w = data_w.dot(1/max_val)
    # print shape(dataN[0:SR*20]**2)mix
    # print shape(dataN[0:SR*20]**2)

    Fname = file_total_M = wav_folder + sls + inst_name_mix + sls + 'mix_dB_'+str(mix_dB)+fileM[sample_count]

    scipy.io.wavfile.write(Fname, SR, data_w)
    # data_ts = wav_convert(data, SR, tar_freq )
    #
    # data_ts = wav_process(data_ts, th_ts, tar_freq*20, min_margin )
    sample_count += 1