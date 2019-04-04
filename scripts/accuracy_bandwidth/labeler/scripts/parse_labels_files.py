
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
import os
from os import path
import sys

import h5py
import numpy as np


LABELS_FILE_EXTENSION = "dat"
EVENT_LABEL = "Event"
UNCERTAIN_LABEL = "Uncertain"
EVENT_VAL = 1
NO_EVENT_VAL = 0
UNCERTAIN_VAL = -1


def read_file(filepath, frame_to_event):
    print("Reading file: {}".format(filepath))
    with open(filepath, "r") as labels_file:
        for line in labels_file:
            # (19, 41) - Event 0
            # (64, 124) - Uncertain
            range_str, label_str = line.strip().split("-")
            start_str, end_str = range_str.strip().lstrip("(").rstrip(")").split(",")
            start_frame = int(start_str.strip())
            end_frame = int(end_str.strip())

            label_str = label_str.strip()
            is_event = label_str.startswith(EVENT_LABEL)
            if is_event:
                label = EVENT_VAL
            elif label_str == UNCERTAIN_LABEL:
                label = UNCERTAIN_VAL
            else:
                print("Invalid label: {}".format(label_str))
                sys.exit(1)

            for frame_id in range(start_frame, end_frame + 1):
                if frame_id not in frame_to_event or is_event:
                    frame_to_event[frame_id] = label


def main():
    parser = argparse.ArgumentParser(
        description="Convert label files into frame vector HDF5 files.")
    parser.add_argument(
        "--labels-dir", type=str,
        help="Path to a directory containing labels files.", required=True)
    parser.add_argument(
        "--num-frames", type=int,
        help="The total number of frames in the video.", required=True)
    parser.add_argument(
        "--out", type=str, help="Path to the output HDF5 file.", required=True)
    args = parser.parse_args()
    labels_dirpath = args.labels_dir
    num_frames = args.num_frames
    out_filepath = args.out

    # Map from frame to a label
    frame_to_event = dict()
    for labels_filename in os.listdir(labels_dirpath):
        if labels_filename.endswith(LABELS_FILE_EXTENSION):
            read_file(path.join(labels_dirpath, labels_filename), frame_to_event)

    out_file = h5py.File(out_filepath, "w")
    out_file.create_dataset("labels", [num_frames], dtype=np.int32)
    for frame_id in range(num_frames):
        out_file["labels"][frame_id] = frame_to_event.get(
            frame_id, NO_EVENT_VAL)


if __name__ == '__main__':
    main()
