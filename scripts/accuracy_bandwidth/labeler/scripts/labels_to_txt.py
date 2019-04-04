
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

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=("Converts labels HDF5 file into a text file where each "
                     "line is frame number of a positive example."))
    parser.add_argument(
        "--in-file", type=str, help="The labels HDF5 file to parse.",
        required=True)
    parser.add_argument(
        "--out-file", type=str, help="The text file to create.", required=True)
    args = parser.parse_args()

    in_filepath = args.in_file
    out_filepath = args.out_file

    in_file = h5py.File(in_filepath, "r")["labels"][:]

    print("Processing {} frames".format(len(in_file)))
    with open(out_filepath, "w") as out_file:
        for frame_id in np.where(in_file == 1)[0]:
            out_file.write("{}\n".format(frame_id))
    return 0


if __name__ == "__main__":
    main()
