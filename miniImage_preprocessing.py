import os
import numpy as np
import pandas as pd
from collections import defaultdict
import tarfile
import gc
import warnings
import six
import argparse
from PIL import Image
import scipy.misc
import pickle
import tensorflow as tf

input_dir = '/home/fenoms/master_thesis/miniImagenet/csv/'

data_dir = '/home/fenoms/master_thesis/miniImagenet/'

_IMAGE_SIZE = 224 #84

def _read_image_as_array(image, dtype='float32'):
    f = Image.open(image)
    k = np.random.randint(0, 4)
    f.rotate(k*90)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image


def download_imgs():
    train = pd.read_csv(input_dir + '/train.csv', sep=',')
    test = pd.read_csv(input_dir + '/test.csv', sep=',')
    val = pd.read_csv(input_dir + '/val.csv', sep=',')
    
    labels = train.label.unique().tolist() + test.label.unique().tolist() + val.label.unique().tolist()

    for label in labels:
        os.system('wget "http://image-net.org/download/synset?wnid=' + label + '&username=fenoms&accesskey=acdcf71fbeafbdbc2cb0036e89b1c614346c733e&release=latest&src=stanford" -O /home/fenoms/master_thesis/miniImagenet/' + label + '.tar')

def pre_process_data(data_path, out_to_file):
    dataset = pd.read_csv(data_path, sep=',')
    labels = dataset.label.unique().tolist()
    tmp_dict = defaultdict(list)
    # file_names = []
    # lable_names = []

    for label in labels:
        print(label)
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        np.random.shuffle(imgs)
        
        c = 0
        
        for img in imgs:
            
            f = tar.extractfile(img)
            
            try:
                img_array = _read_image_as_array(f)
                img_array = scipy.misc.imresize(img_array, (84, 84))
                img_array = img_array.astype('float32')
                img_array *= (1.0/255.0)
                # img_array = img_array.reshape((1,84,84,3))
                # images = np.concatenate((images, img_array), axis=0)
                tmp_dict[label].append(img_array.reshape((1, 84, 84, 3)))
                # file_names.append(img.name)
                # lable_names.append(label)
                c += 1
            except Exception as e:
                print("skipping image, because " + str(e))
            
            if c == 600:
                break
        print(c)

    results = {key: np.concatenate(value) for key, value in tmp_dict.items()}
    np.savez(data_dir + out_to_file + ".npz", **results)
    # sub = pd.DataFrame({'img_file': file_names, 'label':lable_names})
    # sub.to_csv(data_dir + out_to_file + '.csv', index=False)


def process_data(data_path, out_to_file):
    csv = pd.read_csv(data_path, sep = ',')
    labels = csv.label.unique().tolist()
    tra_data = np.zeros((33280, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
    val_data = np.zeros((5120, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
    tra_labels = np.zeros((33280,), dtype=np.int8)
    val_labels = np.zeros((5210,), dtype=np.int8)
    nb_train_images = 0
    nb_val_images = 0
    for k, label in enumerate(labels):
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        c = 0

        for img in imgs:
            f = tar.extractfile(img)
            if c < 80:
                try:
                    img_array = _read_image_as_array(f)
                    img_array = scipy.misc.imresize(img_array, (_IMAGE_SIZE, _IMAGE_SIZE))
                    img_array = np.reshape(img_array, (1,_IMAGE_SIZE, _IMAGE_SIZE, 3))
                    val_data[nb_val_images] = img_array
                    val_labels[nb_val_images] = k
                    c += 1
                    nb_val_images += 1
                except Exception as e:
                    print("skipping image, because " + str(e))
            elif c >= 80 and c < 600:
                try:
                    img_array = _read_image_as_array(f)
                    img_array = scipy.misc.imresize(img_array, (_IMAGE_SIZE, _IMAGE_SIZE))
                    img_array = np.reshape(img_array, (1,_IMAGE_SIZE, _IMAGE_SIZE, 3))
                    tra_data[nb_train_images] = img_array
                    tra_labels[nb_train_images] = k
                    c += 1
                    nb_train_images += 1
                except Exception as e:
                    print("skipping image, because " + str(e))

            else:
                print(c)
                break
        print(nb_train_images)
        print(nb_val_images)

    tra_data = tra_data[:nb_train_images]
    tra_labels = tra_labels[:nb_train_images]
    val_data = val_data[:nb_val_images]
    val_labels = val_labels[:nb_val_images]

    tra_data = {"training_data" : tra_data, "training_label" : tra_labels}
    val_data = {"val_data": val_data, "val_label" : val_labels}
    #the name of data is miniImagenet_data
    pickle_tra_in = open("pickle_tra", "wb")
    pickle_val_in = open("pickle_val", "wb")
    pickle.dump(tra_data, pickle_tra_in)
    pickle.dump(val_data, pickle_val_in)
    print("saved successfully")
 

def load_miniImagenet(data_dir, is_training):
    if is_training:
        pickle_out = open("pickle_tra", "rb")
        data = pickle.load(pickle_out)
        tr_data = data["training_data"]
        tr_label = data["training_label"]
        return tr_data, tr_label
    else:    
        pickle_out = open("pickle_val", "rb")
        data = pickle.load(pickle_out)
        val_data = data["val_data"]
        val_label = data["val_label"]
        return (val_data, val_label)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(data_dir):
    """
    used for generating tfrecords
    """    
    tr_data, tr_label = load_miniImagenet(data_dir, True)
    val_data, val_label = load_miniImagenet(data_dir, False)

    nb_train_images = tr_data.shape[0]
    nb_val_images = val_data.shape[0]

    tr_filename = 'train.tfrecords'
    val_test_filename = 'eval.tfrecords'
    #generating training data
    with tf.python_io.TFRecordWriter(tr_filename) as writer:
        for index in range(nb_train_images):
            image_raw = tr_data[index].tostring()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                'label': _int64_feature(int(tr_label[index])),
                'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())

    #generating validation data

    with tf.python_io.TFRecordWriter(val_test_filename) as writer:
        for index in range(nb_val_images):
            image_raw = val_data[index].tostring()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                'label': _int64_feature(int(val_label[index])),
                'image_raw': _bytes_feature(image_raw)
                }))

            writer.write(example.SerializeToString())


if __name__ == '__main__':
    #download_imgs()
    print("generating tfrecords")
    generate_tfrecords(data_dir)
    # print("trian...")
    # process_data(input_dir + 'train.csv', 'train')
    # print("val...")
    # pre_process_data(input_dir + 'val.csv', 'val')
    # print("test...")
    # pre_process_data(input_dir + 'test.csv', 'test')
