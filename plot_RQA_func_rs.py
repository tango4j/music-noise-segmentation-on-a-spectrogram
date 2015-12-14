__author__ = 'mac'


# import librosa
# import librosa.decompose
from matplotlib.colors import LogNorm
import scipy.io.wavfile
from pylab import *
from numpy import *
import numpy as np
import scipy.io.wavfile
import scipy.linalg
# from scikits.samplerate import resample
import pickle
import math
from random import *


# with open( 'pickle_folder/' + 'MNN_visual_data' + '.pickle', 'w') as f:
#     pickle.dump([D_show, D_mask_dn, D_mask_dm, D_music, D_noise], f)
SN = 1
with open('pickle_folder/' + 'MNN_visual_data_'+ str(SN) + '.pickle') as f:
    D_show, D_show_n, D_show_m, D_mask_dn, D_mask_dm, D_music, D_noise = pickle.load(f)

start_t = 1200
stop_t = shape(D_show)[1] /2

D_show = D_show[:,start_t:stop_t]
D_show_n = D_show_n[:,start_t:stop_t]
D_show_m = D_show_m[:,start_t:stop_t]

print 'All the shapes of data : ', shape(D_show), shape(D_show_n), shape(D_show_m), shape(D_noise)

D_mask_dn = D_mask_dn[:,start_t:stop_t]
D_mask_dm = D_mask_dm[:,start_t:stop_t]
D_noise_n = D_noise[:,start_t:stop_t]
D_music_m = D_music[:,start_t:stop_t]

print 'All the shapes of data : ', shape(D_mask_dn), shape(D_mask_dm), shape(D_noise_n), shape(D_music_m)

total_plot = 3
y_ticks = [0,45,90, 140 ]
labels = (['16kHz', '12kHz', '8kHz', '4kHz'])
aspect = 4.0
subplot(total_plot, 2, 1)
# img = librosa.display.specshow(flipud(D_show), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))
imshow(log10( (D_show_n) ), interpolation = 'none', cmap = 'jet', aspect = aspect )
xlabel('Time (Frame index)')
title('Spectrogram of noise signal')
yticks(y_ticks, labels)

subplot(total_plot, 2, 3)
# img = librosa.display.specshow(flipud(D_show), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))
imshow(log10( (D_show_m) ), interpolation = 'none', cmap = 'jet', aspect = aspect )
xlabel('Time (Frame index)')
title('Spectrogram of a music signal')
yticks(y_ticks, labels)

subplot(total_plot, 2, 5)
# img = librosa.display.specshow(flipud(D_show), cmap = 'jet', y_axis='linear', norm=LogNorm(vmin=0.01, vmax=10))
imshow(log10( (D_show) ), interpolation = 'none', cmap = 'jet', aspect = aspect )
# xlabel('.,Time (Frame index)')
title('Spectrogram of a mixed signal')
yticks(y_ticks, labels)

subplot(total_plot, 2, 2)
print 'check shape Dshow and D mask dn', shape(D_show), shape(D_mask_dn)
imshow(D_mask_dn, interpolation = 'none', cmap = 'jet', aspect = aspect )
# tight_layout()
xlabel('Time (Frame index)')
title('Probability map of noise existence')
yticks(y_ticks, labels)

subplot(total_plot, 2, 4)
imshow(D_mask_dm, interpolation = 'none', cmap = 'jet', aspect = aspect )
# tight_layout()
xlabel('Time (Frame index)')
title('Probability map of music existence')
yticks(y_ticks, labels)
# colorbar(ticks = [0,1])

subplot(total_plot, 2, 6)
print 'D_noise', shape(D_noise)
print 'D_show', shape(D_show)
#
imshow(log2(np.array(D_show)*np.array(D_noise_n)), interpolation = 'none', cmap = 'gray', aspect = aspect )
xlabel('Time (Frame index)')
title('Segmented spectrogram as a noise signal')
yticks(y_ticks, labels)
#
# hold(True)
# #

vmax = amax(log10( (D_show) ) )
vmin = amin(log10( (D_show) ) )
subplot(total_plot, 2, 6)
imshow(log10(np.array(D_show)*np.array(D_music_m)), interpolation = None, cmap = 'jet', aspect = aspect, vmin=vmin, vmax=vmax)
xlabel('Time (Frame index)')
title('Segmented spectrogram as a music signal')
yticks(y_ticks, (['', '12kHz', '8kHz', '4kHz']))

show()