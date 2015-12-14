__author__ = 'inctrl'


from MNSS_function import *


file_mix = 'music_noise_data/mix_dB_0song#103.wav'
file_noise = 'music_noise_data/mix_dB_500song#103.wav'
file_music ='music_noise_data/mix_dB_-500song#103.wav'

file_names=[file_mix, file_noise, file_music]

MNSS_function(file_names)
