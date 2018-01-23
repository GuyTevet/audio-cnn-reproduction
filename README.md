# gt-cp-audio-cnn-reproduction
This is a reproduction of the paper 'LEARNING AUDIO FEATURES FOR SINGER IDENTIFICATION AND EMBEDDING' submitted to ICLR2018

The Paper:
https://openreview.net/pdf?id=SJwZUHRTZ

Openreview Page:
https://openreview.net/forum?id=SJwZUHRTZ

This code was writtem in python2.7 & runs well on Ubuntu / macOS

Steps to run our code:
======================

0.1 - clone it

0.2 - install libav 
        brew install libav (for macOS)
        
        sudo apt-get install libav-tools (for ubuntu)
        
             
0.3 - download the dataset & csv file from https://ccrma.stanford.edu/damp/ (sing! karaoke -> vocal performance (multiple songs) and place them @ ./datasets/DAMP_audio & ./datasets/DAMP_audio_labels.csv

1 - run pre-process : dataset_generator.py


2.1 - configure [run_classification.py] to the desired network arch

2.2 - run classification network [run_classification.py]


3.1 - configure [run_embeddings.py] to the desired network arch

3.2 - run embeddings network [run_embeddings.py]


Chen Ponchek & Guy Tevet


