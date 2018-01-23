#!/usr/bin/env python

"""
    File name:          song_preprocess.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       18/12/2017
    Date last modified: 18/12/2017
    Description:        Imlementing single song pre-process according to the article
"""

import os.path
import sys
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import time

#LOCAL
sys.path.insert(0, 'DL_finnal_project')
from dataset_consts import *

def song_preprocess(song_path,output_data_path,consts,chop_spec=False,save_file=True):

    t0 = time.time() #FOR DEBUG

    ##PROCESS##

    #exit if file exists
    if(os.path.isfile(output_data_path)):
        if(os.path.getsize(output_data_path) > 1e5 ): #100KB
            print "song_preprocess --- %0s ALREADY EXISTS. EXITING ---" %(output_data_path)
            return

    [original_sample_rate, signal]  = read_song(song_path)

    t05 = time.time()  # FOR DEBUG

    [sample_rate, signal]           = resample(signal, original_sample_rate, consts.SAMPLE_RATE)

    #FIXME - Pre-Emphasis - do not mentioned in the article - do we want to add this?
    [frames]                        = frame(signal , consts.FFT_WINDOW_SIZE , consts.FFT_HOP_SIZE)
    [frames]                        = hanning_window(frames)
    [spec]                          = spectogram(frames , consts.FFT_WINDOW_SIZE)

    [mel_spec]                      = mel_spectogram(spec , consts)

    t1 = time.time()  # FOR DEBUG

    if chop_spec:
        [mel_spec]                  = chop_spectogram(mel_spec , consts.CHOP_WINDOW_SIZE , consts.CHOP_HOP_SIZE) # DO NOT FORGET TO DO THIS BEFORE NETWORK INPUT!!

    t2 = time.time()  # FOR DEBUG

    ##SAVE RESULT##
    #quant_mel_spec = quantize(mel_spec,consts)
    quant_mel_spec = mel_spec

    if(save_file):
        data_save(quant_mel_spec,output_data_path)

    t3 = time.time()

    running_time = t3 - t0
    name = os.path.splitext(song_path)[0]
    print "song_preprocess --- DONE WITH %0s RUN TIME %.2f sec ---" %(name,running_time)


    #print("--- READ TIME %.2f seconds ---" % (t1 - t05))  # FOR DEBUG
    #print("--- CHOP TIME %.2f seconds ---" % (t2 - t1))  # FOR DEBUG
    #print("--- TOTAL TIME %.2f seconds ---" % (time.time() - t0)) #FOR DEBUG

    #display_spectogram(spec)
    #display_spectogram(mel_spec)
    #display_spectogram(chopped_spec[:, :, 20])
    #display_spectogram(chopped_spec[:, :, 21])
    #display_spectogram(chopped_spec[:, :, 22])
    #display_spectogram(chopped_spec[:, :, 23])

    return quant_mel_spec


def read_song(song_path):

    extension = os.path.splitext(song_path)[1]
    name = os.path.splitext(song_path)[0]

    if (extension == '.WAV' or extension == '.wav'):
        sample_rate, signal = scipy.io.wavfile.read(song_path)

    elif(extension == '.M4A' or extension == '.m4a'):
        #convert to wav file
        os.system("avconv -i %0s.m4a %0s.wav 2> tmp" % (name,name))

        #read
        sample_rate, signal = scipy.io.wavfile.read("%0s.wav"%name)

        # delete wav file
        os.system("rm -rf %0s.wav tmp" % (name))

    else:
        raise TypeError("not supporting %s file exretension" % extension)

    # if stereo - take only left channel
    if (len(signal.shape) > 1):
        signal = signal[:, 1]

    return [sample_rate, signal]

def cut(signal , sample_rate , time_len , time_offset):


    samples_len = sample_rate * time_len
    samples_offset = sample_rate * time_offset

    signal = signal[samples_offset : samples_offset + samples_len]

    return [signal]


def resample(signal, original_sample_rate, desired_sample_rate):

    if (original_sample_rate == desired_sample_rate): # no need to resample
        return [desired_sample_rate, signal]

    resample_ratio = desired_sample_rate * 1.0 / original_sample_rate
    new_signal_size = int(round(signal.shape[0] * resample_ratio))
    signal = scipy.signal.resample(signal,new_signal_size)

    return [desired_sample_rate, signal]

def frame(signal , frame_length , frame_step):

    #frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    #frame_length = int(round(frame_length))
    #frame_step = int(round(frame_step))
    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,
                              z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return [frames]

def hanning_window(frames):
    frames *= np.hanning(frames.shape[1]) #frames[NUM_OF_FRAMES , FRAME_SIZE]
    return [frames]

def spectogram(frames , NFFT):

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    return [pow_frames]

def display_spectogram(spectogram):

    f, t = np.meshgrid(np.linspace(0.0, 22.05, num=spectogram.shape[1]),
                       np.linspace(0.0, 162.0, num=spectogram.shape[0]))
    plt.pcolormesh(t, f, spectogram)
    plt.ylabel('Frequency [KHz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return

def mel_spectogram(spectogram , consts):
    NFFT = consts.FFT_WINDOW_SIZE
    sample_rate = consts.SAMPLE_RATE
    nfilt = consts.MEL_BINS

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(spectogram, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB - FIXME - should be multipled by 10 or 20

    # clip the spectogram to 60dB - according to the paper
    filter_banks -= consts.MEL_LOW_CUTOFF
    filter_banks[filter_banks <= 0] = 0

    #scale to 0-60 values - FIXME - the paper talking here about adding offset(?) in order to scale to 0-60
    filter_banks = filter_banks *consts.MEL_MAX_VAL / np.max(filter_banks)

    return [filter_banks]

def chop_spectogram(spectogram , window_size , hop_size):

    signal_length = spectogram.shape[0] #time domain
    signal_width = spectogram.shape[1] #freq domain
    frame_length = window_size
    frame_step = hop_size

    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length , signal_width))
    pad_signal = np.concatenate((spectogram,z),axis=0)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    #chopped_spectogram = pad_signal[(1,1),(2,2)]
    chopped_spectogram = np.transpose(pad_signal[indices.astype(np.int32, copy=False)], (1,2,0)).astype(np.float32)


    return [chopped_spectogram]

def data_save(data,output_data_path):

    with open(output_data_path,'w') as file:
        np.save(file,data)

    return

def data_load(data_path):

    with open(data_path , 'rb') as file:
        data = np.load(file)

    return data

def quantize(signal,consts):
    if (consts.QUANTIZATION_NUM_OF_BITS == 8):
        data_type = np.uint8
    elif (consts.QUANTIZATION_NUM_OF_BITS == 16):
        data_type = np.uint16
    else:
        raise ValueError("illegal value for QUANTIZATION_NUM_OF_BITS const (%0d), should be 8 or 16" % consts.QUANTIZATION_NUM_OF_BITS)
    return np.fix(signal * 2 ** consts.QUANTIZATION_NUM_OF_BITS / consts.MEL_MAX_VAL).astype(data_type)


def dequantaize(signal,consts):
    return np.ndarray.astype(signal * consts.MEL_MAX_VAL / 2 ** consts.QUANTIZATION_NUM_OF_BITS , dtype=np.float64)


#c = Dataset_consts()
#song_preprocess('360_21740786.m4a','360_21740786.data',c)
#t4 = time.time()  # FOR DEBUG
#d = data_load('360_21740786.data')
#t5 = time.time()  # FOR DEBUG
#print("--- LOAD TIME %.2f seconds ---" % (t5 - t4))  # FOR DEBUG