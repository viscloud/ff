
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

from __future__ import print_function
import h5py
import numpy as np
import sys
import argparse
import math
import os
import event_metrics
import datetime
import uuid
import time
import skvideo.io
import skimage
import math


def iff(mc_preds, iff_preds, selectivity, buflen):
  ff_max_frames_per_buffer = int(math.ceil(selectivity * float(buflen)))
  extra_count = 0
  ff_confs = np.array(list(map(lambda x: 1 if x > 0.5 else 0, mc_preds)))
  for i in range(0, len(ff_confs), buflen):
    # The last buffer may have fewer than buflen frames in it
    true_len = buflen  # if i + buflen < len(ff_confs) else len(ff_confs) - i
    # Skip this many frames in the current buffer
    cur_buf_num_skipped_frames = true_len - ff_max_frames_per_buffer
    iff_slice = iff_preds[i:i+true_len]
    while(np.count_nonzero(ff_confs[i:i+true_len]) > ff_max_frames_per_buffer):
      zero_idxs = np.argwhere(iff_slice == 0).flatten()
      zero_idxs = [k for k in zero_idxs if ff_confs[i + k] > 0]
      if len(zero_idxs) > 0:
        ff_confs[i + np.random.choice(zero_idxs)] = 0
      else:
        extra_count += 1
        break
  print("EXTRA: {}".format(extra_count))
  print("num selected frames: {}".format(np.count_nonzero(ff_confs)))
  return ff_confs

def k_voting(mc_confs, k=5, pessimistic=False):
  if k % 2 == 0:
    print("Cannot k-vote on an even-length interval because Thomas was too lazy to implement this case")
  preds = np.asarray(np.asarray(mc_confs) >= 0.5).astype(int)
  for i in range(k // 2 + 1, len(preds)):
    interval_start = i - (k // 2)
    interval_end = i + (k // 2) + 1
    if np.count_nonzero(preds[interval_start:interval_end]) > k // 2:
      preds[i] = 1
    else:
      preds[i] = 0
  return preds

fps = 15.0
def cut_video(source_video, start, end, event_number, output_dir):
    duration = str(datetime.timedelta(seconds = float(end - start) / fps))
    start_time = str(datetime.timedelta(seconds = float(start) / fps))
    output_path = os.path.join(output_dir, "{}.mp4".format(event_number))
    cmd = "ffmpeg -ss {} -i {} -c copy -t {} {}".format(start, source_video, duration, output_path)
    os.system(cmd)

def smooth_confs(confs, k=10, pessimistic=False):
    return k_voting(confs, k=k, pessimistic=pessimistic)

def cut_events(confs, labels, source_video, output_dir, iff_labels = None):
    out_bitrate = 2248
    reader = skvideo.io.vreader(source_video)
    writer = skvideo.io.FFmpegWriter(os.path.join(output_dir, "vid.mp4"), outputdict={
        '-vcodec': 'libx264', '-b': str(out_bitrate)
    })
    for i in range(len(confs)):
        if i % 10000 == 0:
          print(i)
        frame = next(reader)
        if(confs[i] > 0.5):
          writer.writeFrame(frame)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FilterForward')
    parser.add_argument('--confidences', type=str, required=False, help='Path to the confidences.npy file.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels.h5 file.')
    parser.add_argument('--video', type=str, required=False, help='Path to the video file.')
    parser.add_argument('--outdir', type=str, required=False, help='output dir')
    args = parser.parse_args(sys.argv[1:])

    mc_confs = np.load(args.confidences)
    confs = get_confs(mc_confs, k=10)
    labels = h5py.File(args.labels)['labels'][0:]
    orig_event_recall = event_metrics.compute_event_detection_metric(mc_confs, labels, existence_coeff=0.9, overlap_coeff=0.1)
    new_event_recall = event_metrics.compute_event_detection_metric(confs, labels, existence_coeff=0.9, overlap_coeff=0.1)
    #source_video_path = args.video
    # Smooth the detections
    # encode a video
    
