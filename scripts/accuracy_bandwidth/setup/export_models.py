
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


import argparse
import tensorflow as tf
from networks_exp_utils import model_util
import os
from os import path

from keras import backend as K
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential

MCS = {
    "1x1_objdet" : {
        "feature_map_layer" : "conv5_6/sep",
        "input_shape" : [33, 60, 1024],
        "type": "TensorFlow"
    },
    "spatial_crop" : {
        "feature_map_layer" : "conv4_2/sep",
        "input_shape" : [67, 120, 512],
        "type": "TensorFlow"
    },
    "discrete" : {
        "input_shape" : [1080, 1920, 3],
        "type": "Keras"
    },
    "windowed" : {
        "feature_map_layer" : "conv4_2/sep",
        "input_shape" : [67, 120, 512],
        "type": "TensorFlow"
    }
}

def larger_cnn3(input_shape):
    model = Sequential()
    model.add(Conv2D(16, 3, strides=(2, 2), padding="same",
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, 3, strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation='relu'))
    model.add(Conv2D(64, 3, strides=(2, 2), padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid', name="probabilities"))
    return model

def mc_spatial_crop_model_fn():
    return model_util.simple_cnn_classifier(
        filter_layers=[tf.layers.separable_conv2d] * 2, filter_size=[3, 3],
        filter_strides=[(1, 1), (2, 2)], filter_number=[16, 32],
        filter_activations=[tf.nn.relu] * 2,
        filter_padding=["same"] * 2)

def mc_windowed_model_fn():
    return model_util.simple_cnn_classifier(
        filter_layers=[tf.layers.conv2d] * 3, filter_size=[1, 3, 3],
        filter_strides=[(1, 1), (1, 1), (2, 2)], filter_number=[32, 32, 32],
        filter_activations=[tf.nn.relu] * 3,
        filter_padding=["same"] * 3)

def larger_cnn3_model_fn(inputs, labels, secondary_inputs):
    conv1 = tf.layers.conv2d(
        inputs, 32, 3, strides=(2, 2), padding="same",
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        conv1, 64, 3, strides=(2, 2), padding="same",
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        conv2, 64, 3, strides=(2, 2), padding="same",
        activation=tf.nn.relu)
    flatten1 = tf.layers.Flatten()(conv3)
    fc1 = tf.layers.Dense(flatten1, 128, activation=tf.nn.relu)
    logits = tf.layers.Dense(fc1, 1)
    # Loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, tf.float32))
    loss = tf.reduce_mean(loss)
    # Outputs
    classes = tf.cast(logits >= 0, tf.int64)
    probabilities = tf.nn.sigmoid(logits, name="probabilities")
    accuracy = tf.contrib.metrics.accuracy(
        labels=labels, predictions=classes)
    return loss, classes, probabilities, accuracy

def mc_1x1_objdet_model_fn(inputs, labels, secondary_inputs):
    conv1 = tf.layers.conv2d(
        inputs, 128, 1, strides=(1, 1), padding="same",
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        conv1, 64, 1, padding="same",
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 1, 1, padding="same", name="conv3")
    logits = tf.reduce_max(conv3, axis=(1, 2, 3))
    # Loss.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, tf.float32))
    loss = tf.reduce_mean(loss)
    # Outputs.
    classes = tf.cast(logits >= 0, tf.int64)
    probabilities = tf.nn.sigmoid(logits, name="probabilities")
    accuracy = tf.contrib.metrics.accuracy(
        labels=labels, predictions=classes)
    return loss, classes, probabilities, accuracy

MCS["spatial_crop"]["model_fn"] = mc_spatial_crop_model_fn()
MCS["1x1_objdet"]["model_fn"] = mc_1x1_objdet_model_fn
MCS["discrete"]["model_fn"] = larger_cnn3
MCS["windowed"]["model_fn"] = mc_windowed_model_fn()

def freeze_graph(model_name, ckpt_dir, output_dir):
    load_ckpt = True
    if ckpt_dir is None:
        load_ckpt = False
        ckpt_dir = output_dir
    input_shape = MCS[model_name]["input_shape"]
    if MCS[model_name]["type"] == "TensorFlow":
        sess = tf.Session()
        input_tensor = tf.placeholder(
            tf.float32, [None] + list(input_shape), name="inputs")
        label_tensor = tf.placeholder(tf.int64, [None])
        secondary_input_tensor = tf.placeholder(tf.float32, [None])
        loss, classes, probabilities, accuracy = MCS[model_name]["model_fn"](
            input_tensor, label_tensor, secondary_input_tensor)
        tf.train.write_graph(
            sess.graph_def, output_dir, "{}.pbtxt".format(model_name), as_text=True)
        if not load_ckpt:
          saver = tf.train.Saver()
          sess.run(tf.global_variables_initializer())
          saver.save(sess, os.path.join(ckpt_dir, "model"))
        model_graph_file = path.join("out", "{}.pbtxt".format(model_name))
        output_prefix = path.join(output_dir, model_name)
        output_layer = "probabilities"
    elif MCS[model_name]["type"] == "Keras":
        model = MCS[model_name]["model_fn"](input_shape)
        saver = tf.train.Saver()
        saver.save(K.get_session(), os.path.join(ckpt_dir, "model"))
        tf.train.write_graph(
            K.get_session().graph_def, output_dir, "{}.pbtxt".format(model_name), as_text=True)
        if not path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        output_prefix = path.join(output_dir, model_name)
        output_layer = "probabilities/Sigmoid"
    else:
        print("Invalid model type {}.".format(MCS[model_name]["type"]))
        exit(-1)
    os.system("python freeze_graph.py "
              "--input_checkpoint {}/model "
              "--output_node_names \"{}\" "
              "--output_graph {}.pb "
              "--input_graph {}.pbtxt".format(ckpt_dir, output_layer, output_prefix, output_prefix))
    print("Frozen model saved in {}".format(output_prefix + ".pb"))
    print("Output layer name: {}".format(output_layer))



def main():
    parser = argparse.ArgumentParser(
        description="Exports frozen TF models that run on "
                    "feature vectors generated by 2048 x 850 input")
    parser.add_argument(
        "--load-checkpoint", type=str,
        help=("Directory containing a Tensorflow checkpoint, "
              "if you wish to load the weights from a checkpoint"),
        required=False)
    parser.add_argument("--model-name", type=str,
        help="The name of the MC to export.",
        required=True, choices=["1x1_objdet", "spatial_crop", "discrete", "windowed"])
    parser.add_argument("--output-dir", type=str,
        help="The directory in which to output the frozen graph.",
        required=True)
    args = parser.parse_args()
    load_ckpt = args.load_checkpoint
    model_name = args.model_name
    output_dir = args.output_dir
    if not load_ckpt is None:
        print("WARNING: Do NOT run this script on real checkpoints, only on temporary copies.")
        print("WARNING: freeze_graph.py can possibly corrupt your checkpoints.")
        raw_input("Press Enter to continue...")
        raw_input("Press Enter again to continue...")
        while load_ckpt[-1] == '/':
            load_ckpt = load_ckpt[:-1]

    freeze_graph(model_name, load_ckpt, output_dir)

if __name__ == "__main__":
    main()
