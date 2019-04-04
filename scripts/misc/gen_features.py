
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

import numpy as np
import h5py
import json
import os
import tensorflow as tf
import skvideo.io
from run_nn_models.model_runners import CaffeModelRunner
from run_nn_models.caffe_models import input_pre_process_fn
import threading
import sys
import shutil
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
import time
import skimage.io
import argparse

def mobilenet(layer, batch_size=8, gpu_id=0, width=2048, height=850):
    with tf.device('/device:GPU:{}'.format(gpu_id)):
        input_imgs = tf.placeholder('uint8', [None, None, None, 3], name='imgs')
        resized_imgs = tf.cast(input_imgs, tf.float32)
        normalized_inputs = \
            0.017 * (resized_imgs - [123.68, 116.78, 103.94])
    def inference_fn(model, inputs):
        model.blobs['data'].data[...] = np.transpose(inputs, (0, 3, 1, 2))
        model.forward()
        outputs = model.blobs[layer].data
        outputs = np.transpose(outputs, (0, 2, 3, 1))
        return outputs

    def post_process_fn(input_columns, outputs, tf_sess=None):
        num_outputs = len(input_columns[0])
        return outputs[:num_outputs]

    return {
        'model_prototxt_path': 'run_nn_models/caffe_nets/' + \
                               'mobilenet_deploy.prototxt',
        'model_weights_path': 'run_nn_models/caffe_nets/mobilenet.caffemodel',
        'input_dims': [width, height],
        'input_preprocess_fn': lambda sess, cols: sess.run(
            normalized_inputs, 
            feed_dict={input_imgs: input_pre_process_fn(cols, batch_size)}),
        'inference_fn': inference_fn,
        'post_processing_fn': post_process_fn,
    }

image_queue = Queue(maxsize=256)
features_queue = Queue(maxsize=256)

def prefetch_images(video_path, num_frames):
    vreader = skvideo.io.vreader(video_path)
    for i in range(0, num_frames):
        img = next(vreader)
        image_queue.put(img)
    image_queue.put([])

def gen_features(layer, batch_size=1, num_gpus=2):
    futures = []
    features_threads = ThreadPoolExecutor(max_workers=num_gpus)
    model_runner_0 = CaffeModelRunner('', batch_size, mobilenet(layer=layer, batch_size=batch_size, height=850, width=2048, gpu_id=0), gpu_id=0)
    model_runner_1 = CaffeModelRunner('', batch_size, mobilenet(layer=layer, batch_size=batch_size, height=850, width=2048, gpu_id=1), gpu_id=1)
    done = False
    while not done:
        for i in range(0, num_gpus):
            inputs = []
            for batch_idx in range(batch_size):
                img = image_queue.get()
                if len(img) == 0:
                  done = True
                  break
                inputs.append(img)
            if i % 2 == 0:
                futures.append(features_threads.submit(model_runner_0.execute, [inputs]))
            else:
                futures.append(features_threads.submit(model_runner_1.execute, [inputs]))
        for future in futures:
            result = future.result()
            features_queue.put(np.copy(result))
            futures = []
    features_queue.put([])

def make_features(video_path, layer, output_file):
    batch_size = 1
    if os.path.exists(output_file):
        print("Error: {} already exists".format(output_file))
        return
    output_hdf5_file = h5py.File(output_file, mode='w')

    num_frames = int(skvideo.io.ffprobe(video_path)['video']["@nb_frames"])
    threads = ThreadPoolExecutor(max_workers=2)
    gen_features_future = threads.submit(gen_features, layer, batch_size=batch_size)
    image_prefetcher_future = threads.submit(prefetch_images, video_path, num_frames)
    crop_fn = lambda x: x

    for i in range(0, num_frames, batch_size):
        start, end = i, i + batch_size
        outputs = crop_fn(features_queue.get())
        print("{}/{}".format(i, num_frames))
        if "features" not in output_hdf5_file:
            output_hdf5_file.create_dataset('features', [end - start] + list(outputs.shape[1:]), maxshape=[None] + list(outputs.shape[1:]), dtype=np.float32)
        else:
            output_hdf5_file['features'].resize(end, axis=0)
        output_hdf5_file['features'][start:end] = outputs
    output_hdf5_file.close()
    os._exit(0)

def encode_video(input_file, crf, bitrate=None, framerate=15):
    output_dir = os.path.dirname(input_file)
    if bitrate is None:
      bitrate_str = "full_quality"
    else:
      bitrate_str = "{}kbps".format(bitrate)
    output_filename = "{}-{}-{}fps.mp4".format(os.path.splitext(os.path.basename(input_file))[0], bitrate_str, framerate)
    output_path = os.path.join(output_dir, output_filename)
    if bitrate is None:
        cmd = "ffmpeg -n -i {} -r 15 -crf {} -an -codec:v libx264 {}".format(input_file, crf, output_path)
    else:
        cmd = "ffmpeg -n -i {} -r 15 -b:v {}k -an -codec:v libx264 {}".format(input_file, bitrate, output_path)
    os.system(cmd)
    return output_path

def main():
    parser = argparse.ArgumentParser("Generate features for a video to disk")
    parser.add_argument("--video-path", type=str, help="Path to the video to extract features from.")
    parser.add_argument("--reencode-video", action="store_true", help="Pass this flag to reencode the video to 15fps")
    parser.add_argument("--output-file", type=str, help="Output file in which to store the hdf5 archive holding the features")
    parser.add_argument("--layer", type=str, help="Layer to extract from mobilenet")
    parser.add_argument("--bitrate", type=int, help="Bitrate in kbps to encode the video to before extracting features", required=False)
    args = parser.parse_args()
    video_path = args.video_path
    reencode_video = args.reencode_video
    output_file = args.output_file
    layer = args.layer
    bitrate = args.bitrate
    if reencode_video:
      video_path = encode_video(video_path, bitrate=bitrate, crf=18, framerate=15)
    make_features(video_path, layer, output_file)

if __name__ == "__main__":
    main()
