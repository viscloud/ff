
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

import keras
from keras import Model
from keras.applications import MobileNet, mobilenet
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import Activation, Flatten, Dense, Input, GlobalMaxPooling3D, Reshape
from keras.preprocessing import image
from keras.callbacks import Callback
from keras.models import Sequential

mobilenet_input_shape=(1080, 1920, 3)

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
mobilenet_reshaped_model.save("mobilenet.h5")
