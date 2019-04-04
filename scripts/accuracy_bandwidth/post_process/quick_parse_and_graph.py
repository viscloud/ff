#!/usr/bin/python3

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import csv
from os import path

import json
import event_metrics
import h5py

def do_graph(data, outfile):
    plt.plot(range(len(data)), data)
    plt.savefig(outfile)
    plt.close()
    print("Saved graph in {}.".format(outfile))

def trial_complete(trial_dir):
    return True
    metric_files = []
    metric_files.append(path.join(trial_dir, "confs.npy"))
    metric_files.append(path.join(trial_dir, "train_metrics", "loss.npy"))
    metric_files.append(path.join(trial_dir, "train_metrics", "acc.npy"))
    metric_files.append(path.join(trial_dir, "metadata.json"))
    for f in metric_files:
        if not path.exists(f):
            return False
    return True

metrics = {}

trial_dirs = sys.argv[1:]

for trial_dir in trial_dirs:
    if not trial_complete(trial_dir):
        print("Skipping {}".format(trial_dir))
        continue
    metrics[trial_dir] = {}

    with open(path.join(trial_dir, "metadata.json"), "r") as f:
        metadata_json = json.load(f)
    metrics[trial_dir]["bitrate"] = metadata_json["test_frames_dir"].split("-")[-2]
    if metrics[trial_dir]["bitrate"] == "baseline":
        metrics[trial_dir]["bitrate"] = 6197
    else:
        metrics[trial_dir]["bitrate"] = metrics[trial_dir]["bitrate"][:-1]
    
    metrics[trial_dir]["task"] = path.basename(path.dirname(metadata_json["test_labels_file"]))

    #confs = np.load(path.join(trial_dir, "confs.npy"))
    confs = np.array([1] * 160000)
    labels = h5py.File(metadata_json["test_labels_file"])["labels"][:len(confs)]
    vals = event_metrics.get_all_metrics(confs, labels)
    metrics[trial_dir].update(vals)

    loss = np.load(path.join(trial_dir, "train_metrics", "loss.npy"))
    acc = np.load(path.join(trial_dir, "train_metrics", "acc.npy"))
    metrics[trial_dir]["loss"] = loss
    metrics[trial_dir]["acc"] = acc
    total_positives = metrics[trial_dir]["per_frame_fp"] + metrics[trial_dir]["per_frame_tp"]
    ff_bandwidth = 0.0#TODO

    print(metrics[trial_dir].keys())

xkey = "bitrate"
per_trial_keys = ["loss", "acc"]
cross_trial_keys = ["event_recall", "event_coverage", "event_overlap",
                    "per_frame_fp", "per_frame_fn", "per_frame_recall",
                    "per_frame_precision", "f1", "event_f1"]

cross_trial_data_points = {}
for trial_dir, trial in metrics.items():
    for key in per_trial_keys:
        do_graph(trial[key], path.join(trial_dir, "{}.pdf".format(key)))
    xval = metrics[trial_dir][xkey]
    for key in cross_trial_keys:
        if key not in cross_trial_data_points:
            cross_trial_data_points[key] = []
        cross_trial_data_points[key].append((xval, trial[key]))

for metric, data in cross_trial_data_points.items():
    xs, ys = zip(*data)
    plt.scatter(xs, ys)
    plt.title(metric)
    plt.savefig("{}.pdf".format(metric))
    plt.close()
    print("Saved graph in {}.pdf".format(metric))


sys.exit()
data_map = {}

for key in keys:
  data_points = {}
  for i in range(len(mycsv[key])):
      fps = mycsv['fps'][i]

      bitrate = mycsv['bitrate'][i]
      is_ff = mycsv['filterforward'][i] == "True"
      microclassifier = mycsv['microclassifier'][i]
      metric = mycsv[key][i]

      if(is_ff):
        print("ff bitrate: {}".format(bitrate))
        fps = "FilterForward"
      if fps not in ["15", "FilterForward"]:
        continue
      if(microclassifier not in data_points):
          data_points[microclassifier] = {}
      if(bitrate not in data_points[microclassifier]):
          data_points[microclassifier][bitrate] = {}
      if(fps not in data_points[microclassifier][bitrate]):
          data_points[microclassifier][bitrate][fps] = []
      data_points[microclassifier][bitrate][fps].append(float(metric))
  data_map[key] = data_points

MC_NAMES = {
    "1x1_objdet" : "Full-Frame Object Detector MC",
    "spatial_crop" : "Localized Binary Classifier MC"
}

for mc in data_points.keys():
    plt.subplots(figsize=(13,7))
    count = 1

    ys = []
    xs = []
    colors = []
    # Sort bitrates by numerical value by removing "k" at the end.
    # Otherwise they will be randomly ordered.
    data_points = data_map["event_f1"]
    bitrates = data_points[mc].keys()
    for bitrate_str in sorted(bitrates, key=lambda x: float(x[:-1])):
        bitrate = float(bitrate_str[:-1])
        for fps in data_points[mc][bitrate_str].keys():
            for point in data_points[mc][bitrate_str][fps]:
                ys.append(point)
                xs.append(bitrate)
                colors.append(colormap[str(fps)])

    plt.subplot(1, 1, count)
    plt.xscale('log', basex=10)
    plt.xlim((9, 10000))
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel("Average Bandwidth, Log Scale (kilobits per second)", fontsize=35)
    plt.ylabel("Event-Level F1 Score", fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    count += 1

    plt.scatter(xs, ys, c=colors, s=250)

    rr = plt.Line2D((0,1),(0,0), color='r', marker='o', linestyle='', markersize=16)
    bl = plt.Line2D((0,1),(0,0), color='b', marker='o', linestyle='', markersize=16)
    plt.legend((rr, bl), ("Compress Everything", "FilterForward"), loc=4, fontsize=30)
    plt.tight_layout()

    plt.savefig("{}-{}-event_f1.pdf".format(mc, mycsv['task'][0]))
    plt.close()
