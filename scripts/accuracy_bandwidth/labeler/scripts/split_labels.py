
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
from os import path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=("Splits a single labels HDF5 file into two after a "
                     "particular number of frames."))
    parser.add_argument(
        "--in-file", type=str, help="The labels file to split.", required=True)
    parser.add_argument(
        "--num-frames", type=int,
        help="The number of frames after which to split.", required=True)
    parser.add_argument(
        "--out-dir", type=str, help="The output dir.", required=True)
    args = parser.parse_args()
    in_filepath = args.in_file
    num_frames = args.num_frames
    out_dirpath = args.out_dir

    out_basename = ".".join(path.basename(in_filepath).split(".")[:-1])
    out_train_filepath = path.join(
        out_dirpath, "{}_train.h5".format(out_basename))
    out_test_filepath = path.join(
        out_dirpath, "{}_test.h5".format(out_basename))

    base_labels = h5py.File(in_filepath, "r")
    train = h5py.File(out_train_filepath, "w")
    test = h5py.File(out_test_filepath, "w")

    num_labels = len(base_labels["labels"])
    train.create_dataset("labels", [num_frames], dtype=np.int32)
    test.create_dataset("labels", [num_labels - num_frames], dtype=np.int32)

    for i in range(0, num_labels):
        if i < num_frames:
            train["labels"][i] = base_labels["labels"][i]
        else:
            test["labels"][i - num_frames] = base_labels["labels"][i]

    train.close()
    test.close()


if __name__ == "__main__":
    main()
