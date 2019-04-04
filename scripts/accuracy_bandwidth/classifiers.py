
# Copyright 2016 The FilterForward Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Train Classifiers for FilterForward """
import argparse
from concurrent import futures
import io
import json
import multiprocessing
import os
from os import path
import queue
import time
import traceback

import h5py
import numpy as np
import tensorflow as tf

import keras
from keras import Model
from keras.applications import MobileNet, mobilenet
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import Activation, Flatten, Dense, Input, GlobalMaxPooling3D, Reshape
from keras.preprocessing import image
from keras.callbacks import Callback
from keras.models import Sequential

IMAGENET_MEAN_VAL = np.asarray([123.68, 116.779, 103.939])
ARRAYQUEUE_TIMEOUT = 800

CROPS = {
    "jackson" : (0, 540, 1920, 1080),
    "roadway" : (0, 315, 2048, 819)
}
CROP = None

#Keras Mobilenet (v1) models and their equivalents to the caffe model
#input_1
#conv1 (conv1)
#conv_pw_1 (conv2_1)
#conv_pw_2 (conv2_2)
#conv_pw_3 (conv3_1)
#conv_pw_4 (conv3_2)
#conv_pw_5 (conv4_1)
#conv_pw_6 (conv4_2)
#conv_pw_7 (conv5_1)
#conv_pw_8 (conv5_2)
#conv_pw_9 (conv5_3)
#conv_pw_10 (conv5_4)
#conv_pw_11 (conv5_5)
#conv_pw_12 (conv5_6)
#conv_pw_13 (conv6)
#global_average_pooling2d_1 (pool6)
#reshape_2


def count_imgs_in_dir(dirpath):
    if path.exists(dirpath):
        last_dir_path = path.join(dirpath,
                                  str(sorted([int(path) for path in os.listdir(dirpath)])[-1]))
        # Remove the file extension
        last_img_path = (sorted(os.listdir(last_dir_path))[-1]).split(".")[0]
        # Remove leading zeros
        while(last_img_path[0] not in [str(i) for i in range(1, 10)]):
            last_img_path = last_img_path[1:]
    else:
        raise Exception("{} does not exist".format(dirpath))
    return int(last_img_path)

class MetadataManager():
    """ Manage metadata related to an experiment """
    def __init__(self, ckpt_dir):
        self.train_frames_dir_key = "train_frames_dir"
        self.test_frames_dir_key = "test_frames_dir"
        self.train_labels_file_key = "train_labels_file"
        self.test_labels_file_key = "test_labels_file"
        self.num_train_frames_key = "num_train_frames"
        self.num_test_frames_key = "num_test_frames"
        self.model_name_key = "model_name"
        self.num_epochs_key = "num_epochs"
        self.batch_size_key = "batch_size"
        self.img_height_key = "img_height"
        self.img_width_key = "img_width"
        self.completed_iters_key = "completed_iters"
        self.completed_epochs_key = "completed_epochs"
        self.ckpt_dir = ckpt_dir
        self.path = path.join(ckpt_dir, "metadata.json")
        self.batches_per_epoch_train = None
        self.data = {}
    def save(self):
        """ Save the contents of this MetadataManager to a json file """
        save_file_atomic(self.path, json.dumps(self.data))
    def load(self):
        with open(self.path) as file_ptr:
            self.data = json.load(file_ptr)
    def setval(self, key, value):
        self.data[key] = value
    def getval(self, key):
        if key in self.data:
            return self.data[key]
        return None

    def get_weights_path(self):
        """Return path where model weights should be saved"""
        return path.join(self.ckpt_dir, "model.h5")

    def get_mc_path(self):
        """Returns path where mc should be saved"""
        return path.join(self.ckpt_dir, "mc.h5")

    def get_confs_path(self):
        """Return path where testing predictions should be saved"""
        return path.join(self.ckpt_dir, "confs.npy")

    def set_train_frames_dir(self, val):
        """ Also calculates the number of train frames """
        if not path.exists(val):
            raise Exception("{} does not exist".format(val))
        self.setval(self.train_frames_dir_key, val)
        self.setval(self.num_train_frames_key, count_imgs_in_dir(val))
    def get_train_frames_dir(self):
        return self.getval(self.train_frames_dir_key)
    def get_num_train_frames(self):
        return self.getval(self.num_train_frames_key)

    def set_test_frames_dir(self, val):
        """ Also calculates the number of test frames """
        if not path.exists(val):
            raise Exception("{} does not exist".format(val))
        self.setval(self.test_frames_dir_key, val)
        self.setval(self.num_test_frames_key, count_imgs_in_dir(val))
    def get_test_frames_dir(self):
        return self.getval(self.test_frames_dir_key)
    def get_num_test_frames(self):
        return self.getval(self.num_test_frames_key)

    def set_train_labels_file(self, val):
        if not path.exists(val):
            raise Exception("{} does not exist".format(val))
        self.setval(self.train_labels_file_key, val)
    def get_train_labels_file(self):
        return self.getval(self.train_labels_file_key)
    def get_train_labels(self):
        return h5py.File(self.get_train_labels_file(), "r")["labels"][:self.get_num_train_frames()]

    def set_test_labels_file(self, val):
        if not path.exists(val):
            raise Exception("{} does not exist".format(val))
        self.setval(self.test_labels_file_key, val)
    def get_test_labels_file(self):
        return self.getval(self.test_labels_file_key)
    def get_test_labels(self):
        return h5py.File(self.get_test_labels_file(), "r")["labels"][:self.get_num_test_frames()]

    def set_model_name(self, val):
        self.setval(self.model_name_key, val)
    def get_model_name(self):
        return self.getval(self.model_name_key)

    def set_batch_size(self, val):
        self.setval(self.batch_size_key, val)
    def get_batch_size(self):
        return self.getval(self.batch_size_key)

    def set_img_height(self, val):
        self.setval(self.img_height_key, val)
    def get_img_height(self):
        return self.getval(self.img_height_key)

    def set_img_width(self, val):
        self.setval(self.img_width_key, val)
    def get_img_width(self):
        return self.getval(self.img_width_key)

    def get_batches_per_epoch_train(self):
        """ Get the number of batches per epoch for training """
        if self.batches_per_epoch_train is None:
            train_labels = self.get_train_labels()
            num_pos = len(np.where(np.asarray(train_labels) == 1)[0])
            num_neg = len(np.where(np.asarray(train_labels) == 0)[0])
            samples_per_epoch = 2 * min(num_pos, num_neg)
            batch_size = self.get_batch_size()
            if batch_size is not None:
                self.batches_per_epoch_train = samples_per_epoch // batch_size
            else:
                raise Exception("Cannot get batches per epoch without labels and images")
        return self.batches_per_epoch_train


    def set_completed_iters(self, val):
        """ Set the number of completed batches and epochs """
        self.setval(self.completed_iters_key, val)
        if self.get_batches_per_epoch_train() is None:
            self.setval(self.completed_epochs_key, 0.0)
        else:
            self.setval(self.completed_epochs_key, val / self.get_batches_per_epoch_train())
    def get_completed_iters(self):
        return self.getval(self.completed_iters_key)
    def get_completed_epochs(self):
        return self.getval(self.completed_epochs_key)

def get_img_path(img_id, img_root_dir):
    """ Turn an image id into a path to a png """
    return path.join(img_root_dir, "{}".format(int(img_id // 5000)),
                     "img_{}.png".format(str(img_id).zfill(9)))

def fetch_data(datagen, idx):
    """ Fetch one batch of data from a data generator """
    try:
        return datagen[idx]
    except Exception as exp:
        traceback.print_exc()
        print()
        raise exp

def prefetch_image_batches(output_q, datagen, max_prefetched_elements, max_workers, start_batch=0):
    """ Prefetch batches of images and queue them into output_queue """
    try:
        futures_q = queue.Queue(maxsize=max_prefetched_elements * 2)
        prefetcher_threads = futures.ThreadPoolExecutor(max_workers=max_workers)

        for idx in range(start_batch, len(datagen)):
            if futures_q.full():
                while not futures_q.empty():
                    output_q.put(futures_q.get().result())
            futures_q.put(prefetcher_threads.submit(fetch_data, datagen, idx))
        output_q.put(None)
    except Exception as exp:
        traceback.print_exc()
        print()
        raise exp

class DataGenerator():
    """ Generic data generator, used to build TestDataGenerator and TrainDataGenerator"""
    def __init__(self, indices, labels, img_root_dir, height, width, batch_size):
        self.batch_size = batch_size
        self.img_root_dir = img_root_dir
        self.indices = indices
        self.img_dims = [height, width, 3]
        self.labels = labels
    def shuffle(self):
        """ Shuffle indices (to be called on epoch end) """
        np.random.shuffle(self.indices)
    def __len__(self):
        """ Returns the number of batches in the epoch """
        return len(self.indices) // self.batch_size
    def __getitem__(self, idx):
        """ Returns one batch of data """
        img_ids = list(self.indices[idx * self.batch_size:(idx+1) * self.batch_size])
        imgs = np.empty((self.batch_size, *self.img_dims))
        labels = np.empty((self.batch_size))
        for i, img_id in enumerate(img_ids):
            img = image.load_img(get_img_path(img_id, self.img_root_dir))

            if CROP is not None:
                foo = img.crop(CROP)
                img = foo

            imgs[i,] = img
            labels[i] = self.labels[img_id - 1]
        return mobilenet.preprocess_input(imgs), labels

class TestDataGenerator(keras.utils.Sequence):
    """ This class generates batches of data to test on """
    def __init__(self, labels, img_root_dir, height, width, batch_size=32):
        self.generator = DataGenerator(range(1, len(labels) + 1), labels,
                                       img_root_dir, height, width, batch_size)
    def on_epoch_end(self):
        """ noop """
        return
    def __len__(self):
        """ Returns the number of batches in the epoch """
        return len(self.generator)
    def __getitem__(self, idx):
        """ Returns one batch of data """
        return self.generator[idx][0]

# This class generates balanced batches of data to train on
class TrainDataGenerator(keras.utils.Sequence):
    """ This class generates batches of data to train on """
    def __init__(self, labels, img_root_dir, height, width, batch_size=32, shuffle=True):
        keras.utils.Sequence.__init__(self)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_root_dir = img_root_dir
        self.pos_gen = DataGenerator(np.where(labels == 1)[0] + 1, labels,
                                     img_root_dir, height, width, batch_size // 2)
        self.neg_gen = DataGenerator(np.where(labels == 0)[0] + 1, labels,
                                     img_root_dir, height, width, batch_size // 2)
        self.len = min(len(self.pos_gen), len(self.neg_gen))
        # Shuffle indices if necessary using on_epoch_end()
        self.on_epoch_end()
    def on_epoch_end(self):
        """ At the end of every epoch, shuffle all of the indices if requested """
        if self.shuffle:
            self.pos_gen.shuffle()
            self.neg_gen.shuffle()
    def __len__(self):
        """ Returns the number of batches in the epoch """
        return self.len
    def __getitem__(self, idx):
        """ Returns one (balanced) batch of data """
        pos_img, pos_label = self.pos_gen[idx]
        neg_img, neg_label = self.neg_gen[idx]
        return np.concatenate((pos_img, neg_img), axis=0), \
               np.concatenate((pos_label, neg_label), axis=0)

def get_microclassifier(mc_model_fns, mc_intermediate_layers, mobilenet_input_shape):
    """ Build a microclassifier (mobilenet -> microclassifier) """
    # Load pre-trained mobilenet and set all layers to not be trainable
    print("Loading weights from 224x224 MobileNet...", end=" ", flush=True)
    mobilenet_base_model = MobileNet(
        input_shape=(224, 224, 3), include_top=False,
        weights='imagenet', input_tensor=None,
        pooling=None)
    print("Done.")
    print("Initializing {}x{} MobileNet...".format(*mobilenet_input_shape), end=" ", flush=True)
    mobilenet_reshaped_model = MobileNet(
        input_shape=mobilenet_input_shape, include_top=False,
        weights=None, input_tensor=None,
        pooling=None)
    print("Copying weights from 224x224 MobileNet...", end=" ", flush=True)
    for reshaped_layer, layer in zip(mobilenet_reshaped_model.layers, mobilenet_base_model.layers):
        reshaped_layer.set_weights(layer.get_weights())
    print("Setting all MobileNet layers to be non-trainable...", flush=True)
    for layer in mobilenet_reshaped_model.layers:
        layer.trainable = False
    print("Done", flush=True)
    # Infer the input shape from the reshaped model
    full_mc_models = []
    for mc_model_fn, mc_intermediate_layer in zip(mc_model_fns, mc_intermediate_layers):
        mc_input_shape = mobilenet_reshaped_model.get_layer(mc_intermediate_layer).output.shape[1:]
        mc_input_shape = tuple([int(dim) for dim in mc_input_shape])
        full_mc_models.append(
            mc_model_fn(mc_input_shape)(
                mobilenet_reshaped_model.get_layer(mc_intermediate_layer).output))
    full_model = Model(inputs=mobilenet_reshaped_model.input, outputs=full_mc_models[0])
    full_model.summary()
    for layer in full_model.layers:
        if layer.trainable:
            print("Training: {}".format(layer.name))
    return full_model

def spatial_crop_model_fn(mc_input_shape):
    """ Returns an instance of the spatial crop MC """
    # Define the microclassifier
    mc_model = Sequential()
    mc_model.add(SeparableConv2D(16, 3, strides=(1, 1), padding="same",
                                 activation="relu", input_shape=mc_input_shape))
    mc_model.add(SeparableConv2D(32, 3, strides=(1, 1), padding="same", activation="relu"))
    mc_model.add(Flatten())
    mc_model.add(Dense(200, activation="relu"))
    mc_model.add(Dense(1, activation="sigmoid"))
    return mc_model

def larger_cnn3_model_fn(input_shape):
    """ Returns an instance of the discrete classifier (NoScope)"""
    model = Sequential()
    model.add(Conv2D(16, 3, strides=(2, 2), padding="same",
                     input_shape=input_shape, activation="relu"))
    model.add(Conv2D(32, 3, strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model

def objdet_model_fn(mc_input_shape):
    """ Returns an instance of the 1x1 simulated tile MC """
    mc_input = Input(shape=mc_input_shape, name="objdet/input")
    mc_output = Conv2D(128, 1, strides=(1, 1), padding="same", activation="relu", name="objdet/conv1")(mc_input)
    mc_output = Conv2D(64, 1, strides=(1, 1), padding="same", activation="relu", name="objdet/conv2")(mc_output)
    mc_output = Conv2D(64, 1, strides=(1, 1), padding="same", name="objdet/conv3")(mc_output)
    cur_shape = keras.backend.int_shape(mc_output)
    mc_output = Reshape([*cur_shape[1:], 1], name="objdet/reshape")(mc_output)
    mc_output = GlobalMaxPooling3D(data_format="channels_last", name="objdet/maxpool")(mc_output)
    mc_output = Activation("sigmoid")(mc_output)
    mc_model = Model(inputs=mc_input, outputs=mc_output, name="objdet")
    return mc_model

# Global data structure that maps model names to parameters/model definitions
MODEL_SPECS_MODEL_FN_KEY = "model_fn"
MODEL_SPECS_LAYER_KEY = "input_layer"
IS_MICROCLASSIFIER_KEY = "is_microclassifier"
MODEL_SPECS = {
    "spatial_crop": {
        MODEL_SPECS_MODEL_FN_KEY : spatial_crop_model_fn,
        MODEL_SPECS_LAYER_KEY : "conv_pw_13",
        IS_MICROCLASSIFIER_KEY : True,
    },
    "discrete": {
        MODEL_SPECS_MODEL_FN_KEY : larger_cnn3_model_fn,
        MODEL_SPECS_LAYER_KEY : None,
        IS_MICROCLASSIFIER_KEY : False
    },
    "objdet": {
        MODEL_SPECS_MODEL_FN_KEY : objdet_model_fn,
        MODEL_SPECS_LAYER_KEY : "conv_pw_12",
        IS_MICROCLASSIFIER_KEY : True
    }
}

# Code taken (with modifications) from:
# https://stackoverflow.com/questions/38666078/fast-queue-of-read-only-numpy-arrays/38775513
class ArrayQueue(object):
    """ Queue backed by shared memory to avoid unnecessary copies """
    def __init__(self, template, maxsize=0):
        print("Allocating new ArrayQueue...")
        if not isinstance(template, np.ndarray):
            raise ValueError("ArrayQueue(template, maxsize) template must be numpy.ndarray")
        if maxsize == 0:
            # this queue cannot be infinite, because it will be backed by real objects
            raise ValueError("ArrayQueue(template, maxsize) must use a finite value for maxsize.")

        # find the size and data type for the arrays
        # note: every ndarray put on the queue must be this size
        self.dtype = template.dtype
        self.shape = template.shape
        self.nbytes = template.nbytes
        print("Allocating {} shared memory ndarrays of size {}B".format(maxsize, self.nbytes))
        # make a pool of numpy arrays, each backed by shared memory,
        # and create a queue to keep track of which ones are free
        self.array_pool = [None] * maxsize
        self.free_arrays = multiprocessing.Queue(maxsize)
        for i in range(maxsize):
            print("Allocating array {}".format(i))
            start = time.time()
            buf = multiprocessing.Array('c', self.nbytes, lock=False)
            print("Allocated array {} in {}s".format(i, time.time() - start))
            self.array_pool[i] = np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
            self.free_arrays.put(i)

        self.queue = multiprocessing.Queue(maxsize)

    def put(self, item, *args, **kwargs):
        """ Put an item on the ArrayQueue, same semantics as multiprocessing.Queue.put() """
        if isinstance(item, np.ndarray):
            if item.dtype == self.dtype and item.shape == self.shape and item.nbytes == self.nbytes:
                # get the ID of an available shared-memory array
                item_id = self.free_arrays.get()
                # copy item to the shared-memory array
                self.array_pool[item_id][:] = item
                # put the array's id (not the whole array) onto the queue
                new_item = item_id
            else:
                raise ValueError(
                    'ndarray does not match type or shape of template used to initialize ArrayQueue'
                )
        else:
            # not an ndarray
            # put the original item on the queue (as a tuple, so we know it's not an ID)
            new_item = (item,)
        self.queue.put(new_item, *args, **kwargs)

    def get(self, *args, **kwargs):
        """ Get an item from the ArrayQueue, same semantics as multiprocessing.Queue.get() """
        item = self.queue.get(*args, **kwargs)
        if isinstance(item, tuple):
            # unpack the original item
            return item[0]
        # item is the id of a shared-memory array
        # copy the array
        arr = self.array_pool[item].copy()
        # put the shared-memory array back into the pool
        self.free_arrays.put(item)
        return arr


class BatchCheckpointer(Callback):
    """ Callback to save a checkpoint at the end of every batch """
    def __init__(self, ckpt_interval, metadata):
        Callback.__init__(self)
        self.ckpt_interval = ckpt_interval
        self.metadata = metadata
        self.cur_batch = 0

    def on_batch_end(self, batch, logs={}):
        """ This function is called by Keras at the end of every batch """
        if self.cur_batch % self.ckpt_interval == 0:
            self.model.save(self.metadata.get_weights_path())
            self.metadata.set_completed_iters(self.metadata.get_completed_iters() + self.cur_batch)
            self.metadata.save()
            self.cur_batch = 0
        self.cur_batch += 1

def save_file_atomic(filepath, contents):
    """ Save a file without risking corrupting on overwrite """
    tmp_filepath = filepath + ".tmp"
    if isinstance(contents, bytes):
        file_ptr = open(tmp_filepath, "wb")
    elif isinstance(contents, str):
        file_ptr = open(tmp_filepath, "w")
    elif isinstance(contents, np.ndarray):
        file_ptr = open(tmp_filepath, "wb")
        np.save(file_ptr, contents)
        file_ptr.flush()
        os.fsync(file_ptr.fileno())
        file_ptr.close()
        os.rename(tmp_filepath, filepath)
        return
    else:
        raise Exception("Cannot handle type {}".format(type(contents)))
    file_ptr.write(contents)
    file_ptr.flush()
    os.fsync(file_ptr.fileno())
    file_ptr.close()
    os.rename(tmp_filepath, filepath)

def save_confs(confs, filepath):
    """ Function that atomically saves confs into confs.npy """
    start = time.time()
    save_file_atomic(filepath, np.asarray(confs))
    print("Saved confs to {} in {}s".format(filepath, time.time() - start))

def train_model(metadata, model, num_epochs, ckpt_interval, max_queue_len, max_threads):
    """ Train a model """
    datagen = TrainDataGenerator(metadata.get_train_labels(),
                                 metadata.get_train_frames_dir(),
                                 metadata.get_img_height(), metadata.get_img_width(),
                                 batch_size=metadata.get_batch_size(), shuffle=True)
    steps_per_epoch = len(datagen)
    cur_epochs = metadata.get_completed_epochs()
    if cur_epochs + 1.0 > num_epochs:
        epochs_to_do = num_epochs - cur_epochs
    else:
        epochs_to_do = 1.0

    steps = epochs_to_do * steps_per_epoch
    model.fit_generator(generator=datagen,
                        use_multiprocessing=False,
                        steps_per_epoch=steps,
                        epochs=1,
                        callbacks=[BatchCheckpointer(ckpt_interval, metadata)],
                        max_queue_size=max_queue_len,
                        workers=max_threads)
    model.summary()
    model.save(metadata.get_weights_path())
    DL_input = Input(model.layers[-1].input_shape[1:])
    DL_model = DL_input
    for layer in model.layers[-1:]:
        DL_model = layer(DL_model)
    DL_model = Model(inputs=DL_input, outputs=DL_model)
    DL_model.summary()
    DL_model.save(metadata.get_mc_path())

def test_model(metadata, model, ckpt_interval, max_queue_len, max_threads):
    """ Test a model, saves results in ckpt_dir/confs.npy """
    labels = metadata.get_test_labels()
    datagen = TestDataGenerator(labels, metadata.get_test_frames_dir(),
                                metadata.get_img_height(), metadata.get_img_width(),
                                batch_size=metadata.get_batch_size())
    testing_batch_queue = ArrayQueue(template=datagen[0], maxsize=max_queue_len)
    test_batch_prefetcher = futures.ThreadPoolExecutor(max_workers=max_threads)
    confs = []
    if path.exists(metadata.get_confs_path()):
        confs = list(np.load(metadata.get_confs_path()))
    batch_size = metadata.get_batch_size()
    start_idx = len(confs) // batch_size
    test_batch_prefetcher.submit(prefetch_image_batches, testing_batch_queue,
                                 datagen, max_queue_len, max_threads, start_batch=start_idx)
    num_batches = len(datagen)
    for idx in range(start_idx, num_batches):
        queue_start = time.time()
        try:
            batch = testing_batch_queue.get(ARRAYQUEUE_TIMEOUT)
        except multiprocessing.TimeoutError:
            break
        queue_delay = time.time() - queue_start
        if batch is None:
            break
        start = time.time()
        probabilities = np.copy(model.predict_on_batch(batch).flatten())
        elapsed = time.time() - start
        predictions = (probabilities > 0.5).astype(int)
        num_valid_samples = len(np.where(labels[batch_size * idx:batch_size * (idx+1)] >= 0)[0])
        if num_valid_samples == 0:
            acc = "N/A"
        else:
            acc = float(len([i for i in range(batch_size) \
                        if predictions[i] == labels[batch_size * idx + i]])) / \
                        len(np.where(labels[batch_size * idx:batch_size * (idx+1)] >= 0)[0])
        print("{}/{}. {}s queue, {}s inference. acc {}".format(idx, num_batches,
                                                               queue_delay, elapsed, acc))
        target_len = (idx + 1) * batch_size
        if len(confs) < target_len:
            confs.extend([0] * (target_len - len(confs)))
        confs[idx * batch_size:(idx + 1) * batch_size] = list(probabilities.flatten())
        if idx % ckpt_interval == 0:
            save_confs(confs, metadata.get_confs_path())
    save_confs(confs, metadata.get_confs_path())


def main():
    parser = argparse.ArgumentParser(
        description="Trains microclassifiers on 2048 x 850 images")
    parser.add_argument(
        "--train-frames", type=str,
        help="Frames to use for training.", required=True)
    parser.add_argument(
        "--test-frames", type=str,
        help="Frames to use for testing.", required=True)
    parser.add_argument(
        "--train-labels", type=str,
        help="Labels to use for training.", required=True)
    parser.add_argument(
        "--test-labels", type=str,
        help="Labels to use for testing.", required=True)
    parser.add_argument(
        "--checkpoint-dir", type=str,
        help=("Directory in which to save keras checkpoint files, "
              "amongst other things...")
        , required=True)
    parser.add_argument(
        "--skip-train", action="store_true")
    parser.add_argument(
        "--load-ckpt", type=str,
        help=("Checkpoint to load (path to hdf5 file)"), required=False)
    parser.add_argument(
        "--model-name", type=str,
        help="Name of the model to run training and/or testing on.",
        required=True, choices=MODEL_SPECS.keys())
    parser.add_argument(
        "--num-epochs", type=float,
        help="The total number of epochs to run training for.", required=True)
    parser.add_argument(
        "--batch", type=int,
        help="The batch size to use in Keras.", required=False, default=32)
    parser.add_argument(
        "--ckpt-interval", type=int, default=32,
        help="The number of batches to run before saving a checkpoint", required=False)
    parser.add_argument(
        "--max-threads", type=int, default=8,
        help="The number of worker processes to fetch batches of images.", required=False)
    parser.add_argument(
        "--max-queue-len", type=int, default=8,
        help="The maximum number of batches to prefetch", required=False)
    parser.add_argument(
        "--input-width", type=int,
        help="Width of input images.", required=True)
    parser.add_argument(
        "--input-height", type=int,
        help="Height of input images.", required=True)
    parser.add_argument(
        "--num-cpu", type=int,
        help="Number of CPU cores to use to prefetch",
        required=False, default=8)
    parser.add_argument(
        "--num-gpu", type=int,
        help="Number of GPUs to use",
        required=False, default=1)
    parser.add_argument(
        "--crop-for-dataset", type=str, help="Apply a pixel-level crop for this dataset.",
        choices=CROPS.keys(), required=False)

    args = parser.parse_args()
    ckpt_dir = args.checkpoint_dir
    load_ckpt_path = args.load_ckpt
    num_epochs = args.num_epochs
    ckpt_interval = args.ckpt_interval
    max_threads = args.max_threads
    skip_train = args.skip_train
    max_queue_len = args.max_queue_len
    num_gpus = args.num_gpu
    crop_for_dataset = args.crop_for_dataset

    # Configure the crop.
    global CROP
    CROP = CROPS[crop_for_dataset] if crop_for_dataset is not None else None

    # Set up tensorflow backend to use available resources
    config = tf.ConfigProto(device_count={'GPU': num_gpus, 'CPU': args.num_cpu})
    keras.backend.set_session(tf.Session(config=config))
    # Set up output directories
    if not path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # Create a new metadata manage
    metadata = MetadataManager(ckpt_dir)
    # If metadata doesn't already exist, make a new metadata
    if not path.exists(metadata.path):
        metadata.set_train_frames_dir(args.train_frames)
        metadata.set_test_frames_dir(args.test_frames)
        metadata.set_train_labels_file(args.train_labels)
        metadata.set_test_labels_file(args.test_labels)
        metadata.set_model_name(args.model_name)
        metadata.set_batch_size(args.batch)
        metadata.set_img_height(args.input_height)
        metadata.set_img_width(args.input_width)
        metadata.set_completed_iters(0.0)
        metadata.save()
    else:
        metadata.load()
        #TODO: Make sure params are the same
    # If the metadata file describes that training has already occurred for the
    # number of epochs requested, skip the training step
    num_completed_epochs = metadata.get_completed_epochs()
    if abs(num_completed_epochs - num_epochs) <= 0.01:
        skip_train = True
        load_ckpt_path = metadata.get_weights_path()
    elif num_completed_epochs > num_epochs:
        print("Model previously trained for {} epochs, aborting...".format(num_completed_epochs))
    else:
        print("Training for {} additional epochs".format(num_epochs - num_completed_epochs))

    if path.exists(metadata.get_confs_path()):
        if len(list(np.load(metadata.get_confs_path()))) >= metadata.get_num_test_frames():
            print("Found confidences for all test frames in {}, exiting...".format(metadata.get_confs_path()))
            return

    # Build the model to be trained
    model_input_shape = [metadata.get_img_height(), metadata.get_img_width(), 3]
    if MODEL_SPECS[metadata.get_model_name()][IS_MICROCLASSIFIER_KEY]:
        # Build microclassifier
        model = \
            get_microclassifier([MODEL_SPECS[metadata.get_model_name()][MODEL_SPECS_MODEL_FN_KEY]],
                                [MODEL_SPECS[metadata.get_model_name()][MODEL_SPECS_LAYER_KEY]],
                                model_input_shape)
    else:
        # Build non-microclassifier
        model = MODEL_SPECS[metadata.get_model_name()][MODEL_SPECS_MODEL_FN_KEY](model_input_shape)

    # Load checkpoint if requested
    if not load_ckpt_path is None:
        # TODO: how do you load weights for a multi-headed model?
        print("Loading weights")
        model.load_weights(load_ckpt_path)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    if not skip_train:
        train_model(metadata, model, num_epochs,
                    ckpt_interval, max_queue_len, max_threads)
        print("Completed training.")

    # Execute testing
    test_model(metadata, model, ckpt_interval, max_queue_len, max_threads)


if __name__ == "__main__":
    main()
