
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


import subprocess
import multiprocessing
from skimage.io import imread
import time
import argparse
import math
from concurrent.futures import ThreadPoolExecutor
import os
from os import path
import shutil


def test_decode_time():
    num_trials = 500

    def time_img(image):
        def get_png():
            foo = imread(image)
            return foo[0][0][0]

        print(get_png())

        start_time = time.time()
        v = []

        for _ in range(num_trials):
            v.append(get_png())

        elapsed_time = time.time() - start_time
        print(sum(v))
        print("{}: {}".format(image, elapsed_time / num_trials))

    time_img("~/scratch/out/output_small.png")
    time_img("~/scratch/out/output_large.png")


def mv_img(img_number, out_dirpath, formt, num_per_dir):
    src_filename = "img_{}.{}".format(str(img_number).zfill(9), formt)
    src_filepath = path.join(out_dirpath, src_filename)
    dst_dirpath = path.join(out_dirpath, "{}".format(img_number // num_per_dir))
    dst_filepath = path.join(dst_dirpath, src_filename)

    if not path.exists(dst_dirpath):
        os.makedirs(dst_dirpath)
    if path.exists(dst_filepath):
        os.remove(dst_filepath)
    shutil.move(src_filepath, dst_filepath)


def mv(out_dirpath, num_per_dir, formt, total_frames):
    # WARNING: This will not move the final image!
    for i in range(1, total_frames):
        # Wait until the next image exists to be sure that the current image
        # has been completely written.
        next_path = path.join(out_dirpath, "img_{}.{}".format(str(i + 1).zfill(9), formt))
        while not path.exists(next_path):
            time.sleep(0.01)
        mv_img(i, out_dirpath, formt, num_per_dir)


def gen(vid_filepath, total_frames, formt_str, out_dirpath, formt):
    subprocess.run(
        "ffmpeg -i {} -vframes {} {} {}/img_%09d.{}".format(
            vid_filepath, total_frames, formt_str, out_dirpath, formt),
        shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-per-dir", type=int, required=True)
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--format-str", type=str, default="", required=False)
    parser.add_argument("--total-frames", type=int, required=True, help="If you do not know the total number of frames, run: ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 <video>")
    parser.add_argument("--fps", type=int, required=True)
    args = parser.parse_args()

    vid_filepath = args.vid
    out_dirpath = args.out_dir
    num_per_dir = args.num_per_dir
    formt = args.format
    formt_str = args.format_str
    total_frames = args.total_frames
    fps = args.fps
    if not path.exists(out_dirpath):
        os.makedirs(out_dirpath)

    mv_proc = multiprocessing.Process(
        target=mv, args=(out_dirpath, num_per_dir, formt, total_frames))
    mv_proc.start()

    gen_proc = multiprocessing.Process(
        target=gen,
        args=(vid_filepath, total_frames, formt_str, out_dirpath, formt))
    gen_proc.start()
    gen_proc.join()

    mv_proc.join()

    # Now that the gen process has been joined and we know all frames have been
    # extracted, we can move the final image.
    mv_img(total_frames, out_dirpath, formt, num_per_dir)


if __name__ == "__main__":
    main()
