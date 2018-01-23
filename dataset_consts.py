#!/usr/bin/env python

"""
    File name:          dataset_consts.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        const class for preprocess configurations
"""

class Dataset_consts():

    def __init__(self):
        # CONSTS - according the paper
        self.SAMPLE_RATE = 22050  # Hz
        self.CHUNK_SIZE = 6  # sec
        self.FFT_WINDOW_SIZE = 2048
        self.FFT_HOP_SIZE = 512
        self.MEL_BINS = 96
        self.MEL_LOW_CUTOFF = 60.0  # dB
        self.MEL_MAX_VAL = 60.0  # dB
        self.CHOP_WINDOW_SIZE = 256  # which is 6 sec
        self.CHOP_HOP_SIZE = 50  # approx. 20% of window size
        self.QUANTIZATION_NUM_OF_BITS = 16  # choose 8 or 16 bits
        self.DATA_THREADS_BATCH_SIZE = 2

    def psdisplay(self):

        str = "Dataset_consts:\n"
        str = str + "SAMPLE_RATE = %0d\n" % self.SAMPLE_RATE
        str = str + "CHUNK_SIZE = %0d\n" % self.CHUNK_SIZE
        str = str + "FFT_WINDOW_SIZE = %0d\n" % self.FFT_WINDOW_SIZE
        str = str + "FFT_HOP_SIZE = %0d\n" % self.FFT_HOP_SIZE
        str = str + "MEL_BINS = %0d\n" % self.MEL_BINS
        str = str + "MEL_LOW_CUTOFF = %0.3f\n" % self.MEL_LOW_CUTOFF
        str = str + "MEL_MAX_VAL = %0.3f\n" % self.MEL_MAX_VAL
        str = str + "CHOP_WINDOW_SIZE = %0d\n" % self.CHOP_WINDOW_SIZE
        str = str + "CHOP_HOP_SIZE = %0d\n" % self.CHOP_HOP_SIZE
        str = str + "QUANTIZATION_NUM_OF_BITS = %0d\n" % self.QUANTIZATION_NUM_OF_BITS
        str = str + "DATA_THREADS_BATCH_SIZE = %0d\n" % self.DATA_THREADS_BATCH_SIZE
        return str

#g = Dataset_consts()
#print(g.psdisplay())
