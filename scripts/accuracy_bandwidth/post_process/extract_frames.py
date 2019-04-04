
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
import math
import multiprocessing
import os
from os import path

import skvideo.io


Q_MAX_LEN = 32


def decoder(in_filepath, total_frames, encoders):
    # Only print the progress every 10%.
    print_interval = math.floor(total_frames / 10)

    reader = skvideo.io.vreader(
        in_filepath)
    for frame_id in range(total_frames):
        if frame_id % print_interval == 0:
            print("Frames: {}/{}".format(frame_id + 1, total_frames))

        frame = next(reader)
        # For every encoder, forward this frame if the encoder's config list
        # include the frame id.
        for conf, queue, _ in encoders:
            if frame_id in conf:
                queue.put(frame)

    # Sending an int signals the encoders to stop.
    for _, queue, _ in encoders:
        queue.put(0)


def encoder(queue, out_filepath, out_bitrate, num_threads):
    writer = skvideo.io.FFmpegWriter(
        out_filepath, outputdict={
            '-vcodec': 'libx264',
            '-b': str(out_bitrate),
            '-threads': str(num_threads)})

    while True:
        val = queue.get()
        # If we receive an int, that is the signal to stop.
        if isinstance(val, int):
            break
        else:
            writer.writeFrame(val)

    writer.close()


def extract_frames(in_filepath, confs, total_frames, out_bitrate, num_threads):
    print("in_filepath: {}".format(in_filepath))
    print("out_bitrate: {}".format(out_bitrate))
    """
    confs should be a list of tuples: (list of frames to select, out filepath)
    """
    # List of tuples: (frame list, queue, out filepath)
    encoders = []
    # List of all of the Process objects.
    processes = []

    for frames, out_filepath in confs:
        queue = multiprocessing.Queue(Q_MAX_LEN)
        encoders.append((frames, queue, out_filepath))
        processes.append(multiprocessing.Process(
            target=encoder,
            args=(queue, out_filepath, out_bitrate, num_threads)))

    processes.append(multiprocessing.Process(
        target=decoder,
        args=(in_filepath, total_frames, encoders)))

    for proc in processes:
        proc.start()

    # Wait for the processes to finish. They coordinate completion on their own.
    for proc in processes:
        proc.join()


def main():
    # Parse args.
    parser = argparse.ArgumentParser(
        ("Selects frames for all of the config files in a given dir. One process"
         "per config file."))
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--out-bitrate", type=int, required=True)
    parser.add_argument(
        "--num-threads", type=int,
        help="The number of FFMPEG decoder and encoder threads to use (each).",
        required=True)
    args = parser.parse_args()
    in_filepath = args.in_file
    config_dirpath = args.config_dir
    out_dirpath = args.out_dir
    total_frames = args.total_frames
    out_bitrate = args.out_bitrate
    num_threads = args.num_threads

    confs = []
    # Parse the config dir. Each config file is given an encoder thread.
    for config_filename in os.listdir(config_dirpath):
        config_filepath = path.join(config_dirpath, config_filename)

        # Parse the config file. Each line is a single int frame number,
        # starting at zero.
        frames = set()
        with open(config_filepath, "r") as config_file:
            for line in config_file:
                frames.add(int(line))

        out_filepath = path.join(
            out_dirpath,
            "{}_selected.mp4".format(config_filename.split(".")[0]))
        confs.append((frames, out_filepath))

    extract_frames(in_filepath, confs, total_frames, out_bitrate, num_threads)


if __name__ == "__main__":
    main()
