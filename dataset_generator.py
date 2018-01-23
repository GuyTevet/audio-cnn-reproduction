
#!/usr/bin/env python

"""
    File name:          dataset_generator.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        script for generating spectogram dataset from the raw audio dataset
"""


import sys
import numpy as np
import threading
import os

#LOCAL
sys.path.insert(0, 'DL_finnal_project')
from dataset_consts import *
from dataset_song_preprocess import *
from dataset_consts import *
from dataset_labels import *


def create_classification_subset(src_dataset_name,dst_subset_name,src_labels_file,dst_labels_file\
                                 ,performers_num=46,song_per_performer=10):
    """create subset for classification task according to the paper"""

    #create dirs
    src_dir = os.path.join('./datasets', src_dataset_name)
    dst_dir = os.path.join('./datasets', dst_subset_name)

    if( not os.path.isdir(dst_dir)):
        os.mkdir(dst_dir)

    #find_performers & create new csv file

    src_csv =  open(src_labels_file, 'rb')
    dst_csv =  open(dst_labels_file, 'w')

    cur_line = src_csv.readline()

    #copy titles
    dst_csv.write(cur_line)

    #init performers lists
    male_performers = []
    female_performers = []

    cur_line = src_csv.readline()

    while(len(male_performers) < performers_num/2 \
                  or len(female_performers) < performers_num/2):

        gender = cur_line.split(',')[3]
        performer = int(cur_line.split(',')[0])

        if(performer == 77130559 or performer == 77716090): #sorry for the hard coding , unknown problem with this one
            cur_line = src_csv.readline()
            continue

        if(gender == 'M' and male_performers.count(performer) == 0 and len(male_performers) < performers_num/2)\
                or (gender == 'F' and female_performers.count(performer) == 0 and len(female_performers) < performers_num/2):

            #insert performer to list
            if(gender == 'M'):
                male_performers.append(performer)
            else:
                female_performers.append(performer)

            #insert songs to dir & csv
            for i in range(song_per_performer):
                song_name = cur_line.split(',')[1] + '.m4a'
                src_song = os.path.join(src_dir, song_name)
                dst_song = os.path.join(dst_dir, song_name)

                #assert same performer as expected
                assert(performer == int(cur_line.split(',')[0]))

                #copy line
                dst_csv.write(cur_line)
                #copy song
                os.system("cp -fv %0s %0s"%(src_song,dst_song))

                cur_line = src_csv.readline()
        else:
            cur_line = src_csv.readline()

    src_csv.close()
    dst_csv.close()

    num_dest_files = len([name for name in os.listdir(dst_dir)])

    #assertions
    assert(num_dest_files == performers_num * song_per_performer)
    assert(len(male_performers) == performers_num/2)
    assert(len(female_performers) == performers_num/2)

    return

def gen_classification_kfold_dataset(raw_data_path , labels_file ,  output_name , k = 10 ,size = 460):
    """gets classification subset as '' and creates 10-fold dataset according to the paper"""

    #create dirs
    output_dir = os.path.join('./datasets/preprocessed/', output_name)
    output_path = []

    for i in range(k):
        output_path.append(os.path.join(output_dir,"%0s_%02d.data" % (output_name,i)))
    data_dir = os.path.join(output_dir,'data')

    if( not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    if( not os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    #import consts
    consts = Dataset_consts()

    #load singers lookup table
    singers_lookup = get_singers_lookup_table(labels_file)

    create_spectograms(raw_data_path , data_dir ,labels_file , consts , size , chop_spec = True)
    merge_kfold_data_labels(data_dir , output_path , labels_file , size , k , singers_lookup , remove_date_files=False)

    create_logfile(consts , output_dir , output_name , size)
    return

def gen_dataset(raw_data_path , labels_file ,  output_name , size = 1000):

    #create dirs
    output_dir = os.path.join('./datasets/preprocessed/', output_name)
    output_path = os.path.join(output_dir,"dataset_%0s.data" % output_name)
    data_dir = os.path.join(output_dir,'data')

    if( not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    if( not os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    #import consts
    consts = Dataset_consts()

    #load singers lookup table
    singers_lookup = get_singers_lookup_table(labels_file)

    create_spectograms(raw_data_path , data_dir ,labels_file , consts , size)
    merge_data_labels(data_dir , output_path , labels_file , size , singers_lookup , remove_date_files=False)

    create_logfile(consts , output_dir , output_name , size)

    return

def create_spectograms(raw_data_path , data_dir , labels_file , consts, size = 1000 , chop_spec = False):

    threads = []
    threads_batch_size = consts.DATA_THREADS_BATCH_SIZE
    thread_last_batch_size = size % threads_batch_size
    thread_batch_num = size / threads_batch_size

    t_init = time.time()

    with open(labels_file, 'rb') as csvfile:

        csvfile.readline()  # skip title

        for i in range(thread_batch_num):
            # prepare threads
            for j in range(threads_batch_size):
                name = csvfile.readline().split(',')[1]
                song_name = "%0s.m4a" % name
                data_name = "%0s.data" % name
                song_path = os.path.join(raw_data_path , song_name)
                data_path = os.path.join(data_dir , data_name)
                #song_preprocess(song_path, data_path, consts)
                thread = threading.Thread(target = song_preprocess, args=(song_path, data_path, consts, chop_spec))
                threads.append(thread)

            #run threads
            for thread in threads:
                thread.start()

            #wait for done
            for thread in threads:
                thread.join()

            #remove threads
            threads = []

        #handle last batch
        for j in range(thread_last_batch_size):
            name = csvfile.readline().split(',')[1]
            song_name = "%0s.m4a" % name
            data_name = "%0s.data" % name
            song_path = os.path.join(raw_data_path, song_name)
            data_path = os.path.join(data_dir, data_name)
            # song_preprocess(song_path, data_path, consts)
            thread = threading.Thread(target=song_preprocess, args=(song_path, data_path, consts,))
            threads.append(thread)

        # run threads
        for thread in threads:
            thread.start()

        # wait for done
        for thread in threads:
            thread.join()

        # remove threads
        threads = []


    t_done = time.time()

    print("create_spectograms --- DONE. RUN TIME %.2f sec ---" % (t_done - t_init))

    return

#def merge_data():

def create_logfile(consts , output_dir , output_name , size , k=1):

    log_path = os.path.join(output_dir,'dataset_log.txt')

    with open(log_path,'w') as file:
        file.write("###DATASET LOG\n")
        file.write("###NAME : %0s \n" % output_name)
        file.write("###PATH : %0s \n" % os.path.abspath(output_dir))
        file.write("###SIZE : %0d \n" % size)
        file.write("###K    : %0d \n" % k)
        file.write("###CREATED : %0s \n" % time.strftime("%d/%m/%Y %H:%M:%S"))
        file.write("###%0s\n" % consts.psdisplay())

    return

def merge_data_labels(data_dir , output_path ,labels_file , size , lookup_table , remove_date_files = True ):

    t_init = time.time()

    merged_dataset = []

    with open(labels_file, 'rb') as csvfile:

        csvfile.readline()  # skip title

        for i in range(size):
            line = csvfile.readline().split(',')
            singer = int(line[0])
            data_file = "%0s.data"%line[1]
            label = siger2label(singer , lookup_table )
            data = np.load(os.path.join(data_dir,data_file))
            merged_dataset.append([data , label])
            if (remove_date_files):
                os.system("rm -rf %0s" % os.path.join(data_dir,data_file))

    with open(output_path , 'w') as outfile:
        np.save(outfile,merged_dataset)

    t_done = time.time()

    print("merge_data_labels --- DONE. RUN TIME %.2f sec ---" % (t_done - t_init))

    return

def merge_kfold_data_labels(data_dir , output_path ,labels_file , size, k , lookup_table , remove_date_files = True ):

    t_init = time.time()

    for offset_i in range(k):

        _merged_data = []
        _merged_labels = []

        with open(labels_file, 'rb') as csvfile:

            csvfile.readline()  # skip title

            for song_i in range(size):
                if song_i % k == offset_i :
                    line = csvfile.readline().split(',')
                    singer = int(line[0])
                    data_file = "%0s.data" % line[1]
                    label = siger2label(singer, lookup_table)
                    data = np.load(os.path.join(data_dir, data_file), encoding='bytes')
                    if (len(data.shape) == 3):
                        _merged_data.append(data)
                        _merged_labels.append(label)
                    else:
                        print ('shape problem with' + data_file + ' skiping:')
                        print (data.shape)
                    if (remove_date_files):
                        os.system("rm -rf %0s" % os.path.join(data_dir, data_file))
                else:
                    csvfile.readline()

            merged_data = _merged_data[0]
            merged_labels = np.expand_dims(_merged_labels[0], axis=1)
            for chunks in range(1,_merged_data[0].shape[2]):
                merged_labels = np.concatenate((merged_labels, np.expand_dims(_merged_labels[0],axis=1)), axis=1)

            for j in range(1, len(_merged_data)):
                merged_data = np.concatenate((merged_data, _merged_data[j]), axis=2)
                for chunks in range(_merged_data[j].shape[2]):
                    merged_labels = np.concatenate((merged_labels, np.expand_dims(_merged_labels[j],axis=1)), axis=1)

            with open(output_path[offset_i], 'w') as outfile:
                np.save(outfile, [merged_data, merged_labels])



    t_done = time.time()

    print("merge_data_labels --- DONE. RUN TIME %.2f sec ---" % (t_done - t_init))

    return

def siger2label(singer,lookup_table): #singer is the id in the csv file. label is one hot vector

    one_hot = np.zeros((len(lookup_table)) , dtype=np.float32)
    index = lookup_table.index(singer)
    one_hot[index] = 1.0

    return one_hot




#gen_dataset('./datasets/DAMP_audio','./datasets/DAMP_audio_labels.csv','try01' , size = 1000)

#with open('./datasets/preprocessed/try01/dataset_try01.data' , 'r') as file:
#    data = np.load(file)
#
#d = data[0,0].shape
#l = data[0,1].shape
#a=1

create_classification_subset('DAMP_audio','DAMP_subset_for_classification','./datasets/DAMP_audio_labels.csv'\
                             ,'./datasets/DAMP_subset_for_classification_labels.csv')


gen_classification_kfold_dataset('./datasets/DAMP_subset_for_classification','./datasets/DAMP_subset_for_classification_labels.csv'\
                                 ,'classification_dataset01',k=10,size=460)



