
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

import glob
import json
import numpy as np
import pandas as pd
import scipy.io
import sys

FRAME_RATE = 30.00 # frames/second
OBJECT_CLASS = 'person'

DATA_PREFIX = 'data'
LABELS_NAME = 'labels'
TEST_DATA_PREFIX = 'test'
TEST_FRAMES_PREFIX = 'frames_test'
TEST_LABELS_NAME = 'labels_test'
TRAIN_DATA_PREFIX = 'train'
TRAIN_FRAMES_PREFIX = 'frames_train'
TRAIN_LABELS_NAME = 'labels_train'

DEFAULT_SHARD_SIZE = 100000

class Time(object):
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second
    def __repr__(self):
        return "%d:%d:%d" % (self.hour, self.minute, self.second)

def convert_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def write_numpy_array_to_shards(npy_path, data_prefix, output_dir,
                                SHARD_SIZE=DEFAULT_SHARD_SIZE):
    npy_data = np.load(npy_path)
    for i in range(0, len(npy_data) // SHARD_SIZE):
        start_offset = i * SHARD_SIZE
        data_chunk = npy_data[start_offset:start_offset + SHARD_SIZE]
        np.save('%s/%s_%d.npy' % (output_dir, data_prefix, i), data_chunk)
    if len(npy_data) % SHARD_SIZE != 0:
        start_offset = (len(npy_data) // SHARD_SIZE) * SHARD_SIZE
        data_chunk = npy_data[start_offset:]
        np.save('%s/%s_%d.npy' % (output_dir, data_prefix, i + 1), data_chunk)

# Label making utilities
def find_num_continuous(frames, label_min_interval):
    sorted_frames = sorted(frames)
    start = 0
    ranges = []
    i = 0
    for i, elem in enumerate(sorted_frames[:-1]):
      if sorted_frames[i + 1] - sorted_frames[i] > label_min_interval:
        ranges.append([(start, i), (sorted_frames[start], sorted_frames[i])])
        start = i + 1
    start = min(start, len(sorted_frames)-1)
    i -= 1
    ranges.append([(start, i), (sorted_frames[start], sorted_frames[i])])
    return ranges

def smooth_detections(frame_numbers, label_min_interval):
    label_ranges = find_num_continuous(frame_numbers, label_min_interval)
    smoothed_frames = []
    for (start, end_inclusive), (frame_start, frame_end_inclusive) in label_ranges:
        smoothed_frames += [i for i in range(frame_start, frame_end_inclusive + 1)]
    # Quantify impact of the smoothing.
    print(find_num_continuous(frame_numbers, 1))
    print(find_num_continuous(smoothed_frames, 1))
    print("Smoothed %d separate detection regions into %d regions" % \
            (len(find_num_continuous(frame_numbers, 1)),
             len(find_num_continuous(smoothed_frames, 1))))
    return smoothed_frames

def smooth_existing_labels(labels, label_min_interval):
    detection_frames = np.where(labels == 1)[0]
    smoothed_detection_frames = \
        smooth_detections(detection_frames, label_min_interval)
    frame_set = np.zeros(len(labels), dtype=np.bool)
    frame_set[smoothed_detection_frames] = True
    frame_numbers = np.arange(len(labels))
    detections_idx = [i for i, frame_no in enumerate(frame_numbers) if frame_set[frame_no]]
    new_labels = np.zeros(len(labels))
    new_labels[detections_idx] = 1
    return new_labels

def generate_labels(csv_path, conf_threshold, frame_numbers, label_min_interval=500):
    frames_set = set(frame_numbers)
    df = pd.read_csv(csv_path)
    df = df[(df['object_name'] == OBJECT_CLASS) & (df['frame'].isin(frames_set)) & \
            (df['confidence'] >= conf_threshold)]
    detection_frames = np.unique(df['frame'])
    # Smooth out the detections.
    # smoothed_detection_frames = detection_frames
    smoothed_detection_frames = smooth_detections(detection_frames, label_min_interval)
    frame_set = np.zeros(max(frame_numbers) + 1, dtype=np.bool)
    frame_set[smoothed_detection_frames] = True
    detections_idx = [i for i, frame_no in enumerate(frame_numbers) if frame_set[frame_no]]
    labels = np.zeros(len(frame_numbers))
    labels[detections_idx] = 1
    return labels

def make_data_labels(
        data_dir, csv_path, conf_threshold, label_min_interval, output_dir):
    num_frames = get_data_len(data_dir)
    labels = generate_labels(
        csv_path, conf_threshold, np.arange(num_frames), label_min_interval)
    np.save('%s/%s.npy' % (output_dir, LABELS_NAME), labels)

# Sharded File reading utilities
def read_data_range(data_dir, start, end_exclusive,
                    SHARD_SIZE=DEFAULT_SHARD_SIZE):
    # Find the files corresponding to the start and end indices.
    start_file_no = start // SHARD_SIZE
    end_file_no = (end_exclusive - 1) // SHARD_SIZE
    num_files = len(glob.glob('%s/%s_*.npy' % (data_dir, DATA_PREFIX)))
    assert start_file_no < num_files and end_file_no < num_files
    data = []
    for i in range(start_file_no, end_file_no + 1):
        data_array = np.load('%s/%s_%d.npy' % (data_dir, DATA_PREFIX, i))
        start_index = (start % SHARD_SIZE) if i == start_file_no else 0
        end_index = ((end_exclusive - 1) % SHARD_SIZE) + 1 if i == end_file_no else len(data_array)
        data.append(data_array[start_index:end_index])
    return np.vstack(data)

def read_data_time_range(data_dir, start_time, end_time):
    interval_seconds = convert_to_seconds(interval)
    start_frame = int(convert_to_seconds(start_time) * FRAME_RATE)
    end_frame = int(convert_to_seconds(end_time) * FRAME_RATE)
    return read_data_range(data_dir, start_frame, end_frame + 1)

def get_data_len(data_dir, read_all=False, SHARD_SIZE=DEFAULT_SHARD_SIZE):
    data_len = 0
    if read_all:
        filenames = glob.glob('%s/%s_*.npy' % (data_dir, DATA_PREFIX))
        for fname in filenames:
            data = np.load(fname)
            data_len += len(data)
    else:
        num_files = len(glob.glob('%s/%s_*.npy' % (data_dir, DATA_PREFIX)))
        last_file = '%s/%s_%d.npy' % (data_dir, DATA_PREFIX, num_files - 1)
        data_len = (num_files - 1) * SHARD_SIZE + len(np.load(last_file))
    return data_len

# Sharded file writing utilities
def write_indices_to_new_dir(data_dir, output_dir, indices,
                             SHARD_SIZE=DEFAULT_SHARD_SIZE):
    f_iter = ShardedFileIterator(data_dir)
    output_buffer = []
    num_output_shards, curr_idx = 0, 0
    for i in indices:
        if i > curr_idx:
            f_iter.get_next_entries(i - curr_idx)
            curr_idx = i
        curr_idx += 1
        output_buffer.append(f_iter.get_next_entries(1))
        if len(output_buffer) == SHARD_SIZE:
            np.save('%s/data_%d.npy' % (output_dir, num_output_shards),
                    np.array(output_buffer).squeeze())
            output_buffer = []
            num_output_shards += 1
    if len(output_buffer) > 0:
        np.save('%s/data_%d.npy' % (output_dir, num_output_shards),
                np.array(output_buffer).squeeze())

class ShardedFileIterator(object):
    def __init__(self, data_dir, stride=DEFAULT_SHARD_SIZE,
                 data_preprocessing_fn=lambda x: x):
        self.data_dir = data_dir
        self.file_index = 0
        self.item_index = 0
        self.global_offset = 0
        self.stride = stride
        self.data_preprocessing_fn = data_preprocessing_fn
        self.curr_data = self.load_data(0)

    def load_data(self, index):
        return np.load('%s/%s_%d.npy' % (self.data_dir, DATA_PREFIX, index))

    def get_data_len(self, read_all=False):
        return get_data_len(self.data_dir, read_all=read_all)

    def get_next_entries(self, num_items):
        stride = self.stride
        num_strides = num_items // self.stride 
        remainder = num_items % self.stride
        items = []
        for _ in range(num_strides):
            items.append(self.get_next_entries_helper(stride))
        if remainder > 0:
            items.append(self.get_next_entries_helper(remainder))
        return np.vstack(items)

    def get_next_entries_helper(self, num_items):
        if self.item_index + num_items > len(self.curr_data):
            entries = self.curr_data[self.item_index:]
            self.item_index = num_items - (len(self.curr_data) - self.item_index)
            self.file_index += 1
            self.curr_data = self.load_data(self.file_index)
            entries = np.vstack((entries, self.curr_data[:self.item_index]))
        else:
            entries = self.curr_data[self.item_index:self.item_index+num_items]
            self.item_index += num_items
        self.global_offset += num_items
        # Apply preprocessing to data if applicable.
        entries = self.data_preprocessing_fn(entries)
        return entries

    def seek(self, index):
        if index < self.global_offset:
            self.__init__(self.data_dir)

        stride = self.stride
        num_strides = (index - self.global_offset) // stride
        remainder = (index - self.global_offset) % stride
        for i in range(num_strides):
            self.get_next_entries(stride)
        if remainder > 0:
            self.get_next_entries(remainder)
        self.global_offset = index

class LabelIterator(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.item_index = 0
        self.labels = self.load_labels()

    def load_labels(self):
        return np.load('%s/%s.npy' % (self.data_dir, LABELS_NAME))

    def get_data_len(self):
        return len(self.labels)

    def get_next_entries(self, num_items):
        entries = self.labels[self.item_index:self.item_index + num_items]
        self.item_index += num_items
        return entries

    def seek(self, index):
        self.item_index = index
