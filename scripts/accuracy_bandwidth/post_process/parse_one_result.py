
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

import argparse
import os
from os import path
import pickle
import shutil
import sys
import uuid

import numpy as np
import h5py

import filterforward
import extract_frames
import event_metrics


# Metadata about the various datasets.
DATASET_MAP = {
    "jackson": {
        "alias" : "new_jh",
        "video_len_s" : 5 * 60 * 60 + 59 * 60 + 42.40,
        "num_frames" : 300000,
        "original_bitrate" : "2288k",
        "resolution" : "1920-1080",
        # Relative to fawnstore2 root.
        "sufficient_quality_path" : "ff/serv/new_jh_3-15fps-{}.mp4",
        "labels_filename" : "new_jh_3-{}-15fps.h5"
    },
    "roadway" : {
        "alias" : "iii3",
        "video_len_s" : 3 * 60 * 60 + 0.67,
        "num_frames" : 161984,
        "original_bitrate" : "6916k",
        "resolution" : "2048-850",
        # Relative to fawnstore2 root.
        "sufficient_quality_path" : "ff/serv/vimba_iii_3_2018-3-21_16-15fps-{}.mp4",
        "labels_filename" : "labels-15fps.h5"
    }
}
SUFFICIENT_QUALITY = {
    "pedestrian" : {
        "spatial_crop" : "1000k",
        "1x1_objdet" : "1000k",
        "1x1_simulated_tile" : "500k",
        "windowed_spatial_crop-window5" : "250k"
    },
    "red_truck" : {
        "spatial_crop" : None,
        "1x1_objdet" : None,
        "1x1_simulated_tile" : "250k",
        "windowed_spatial_crop-window5" : "250k"
    },
    "people_red" : {
        "spatial_crop" : "1000k",
        "1x1_objdet" : "250k",
        "1x1_simulated_tile" : None,
        "windowed_spatial_crop-window5" : None
    }
}
CHECKPOINT_LEN = 20
CURRENT_TRIAL_DIR_LISTING = {
    "jackson" : {
        "pedestrian" : {
            "spatial_crop" : {
                "100k" : [],
                "175k" : [],
                "250k" : [],
                "500k" : [],
                "1000k" : [],
                "baseline" : []
            },
            "1x1_objdet" : {
                "100k" : [],
                "175k" : [],
                "250k" : [],
                "500k" : [5, 10, 12, 15],
                "1000k" : [50, 55, 57, 60],
                "baseline" : [3, 8, 11, 14]
            }
        }
    },
    "roadway": {
        "people_red" : {
            "spatial_crop" : {
                "100k" : [7],
                "175k" : [],
                "250k" : [7, 9],
                "500k" : [31, 33],
                "1000k" : [3],
                "baseline" : [4]
            },
            "1x1_objdet" : {
                "100k" : [0, 1],
                "175k" : [],
                "250k" : [1, 2],
                "500k" : [1, 2],
                "1000k" : [0, 1],
                "baseline" : [3]
            }
        }
    }
}


def get_one_exps(ckpt_root, labels_root, dataset, task, mc_to_test, num_trials=None):
    print("Looking for checkpoints in: {}".format(ckpt_root))
    paths = []
    for task in CURRENT_TRIAL_DIR_LISTING[dataset].keys():
        for microclassifier in [mc for mc in CURRENT_TRIAL_DIR_LISTING[dataset][task].keys() \
                                if mc == mc_to_test]:
            for bitrate in CURRENT_TRIAL_DIR_LISTING[dataset][task][microclassifier].keys():
                most_current_trials = CURRENT_TRIAL_DIR_LISTING[
                    dataset][task][microclassifier][bitrate]
                print("Looking for trails {} for {}:{}:{}".format(
                    most_current_trials, task, microclassifier, bitrate))

                if most_current_trials is None:
                    continue
                for idx, trial in enumerate(most_current_trials):
                    if num_trials is not None and idx >= num_trials:
                        continue

                    trial_filepath = path.join(
                        ckpt_root, DATASET_MAP[dataset]["alias"], task, microclassifier, bitrate,
                        "{}-{}-{}-trial{}/confs.npy".format(task, microclassifier, bitrate, trial))
                    if path.exists(trial_filepath):
                        labels_filepath = path.join(
                            labels_root, task,
                            DATASET_MAP[dataset]["labels_filename"].format(task))
                        paths.append(
                            (trial_filepath, labels_filepath, bitrate, microclassifier))
                    else:
                        old_format_trial_filepath = path.join(
                            ckpt_root, task, microclassifier, bitrate,
                            "{}-{}-trial{}/confs.npy".format(task, microclassifier, trial))
                        if path.exists(old_format_trial_filepath):
                            print("ERROR: THIS CHECKPOINT IS IN THE OLD FORMAT: {}".format(
                                old_format_trial_filepath))
                            sys.exit(1)
                        else:
                            print("Error: Unable to find trial: {}".format(trial_filepath))
                            sys.exit(1)
    return paths


def process_confs(confs):
    if len(confs.shape) > 1:
        return [x[1] for x in confs]
    else:
        return confs


def print_results(results, outfile):
    keys = set()
    for result_dict in results:
        keys.update(result_dict.keys())
    if "one_exp" in keys:
        keys.remove("one_exp")

    header = ""
    for key in sorted(keys):
        header += key + ","
    # Strip trailing comma
    header = header[:-1]
    with open(outfile, "w") as outfd:
        print(header, file=outfd)
        for result in results:
            output_line = ""
            for key in sorted(keys):
                if key in result:
                    if key == "frame_indices":
                        val = -1
                    else:
                        val = result[key]
                        try:
                            # Convert ndarrays to strings. This hack is required
                            # to make sure that all ndarray values are printed
                            # and that they are printed on a single line. We
                            # delete the commas so that we do not mess up the
                            # syntax of the final CSV file.
                            val = str(val.tolist()).replace(",", "")
                        except AttributeError:
                            # If the value is not an ndarray...
                            pass
                    output_line += str(val) + ","
                else:
                    output_line += "-1" + ","
            # Strip trailing comma and print to file
            print(output_line[:-1], file=outfd)
    print("Wrote: {}".format(outfile))


def save_ckpt(data, dataset, task, out_dir):
    """data is list """
    ckpt_basename = "{}-{}-checkpoint".format(dataset, task)
    ckpt_dirpath = path.join(out_dir, ckpt_basename)

    # Always delete the old checkpoint first, in case the new one contains
    # fewer files.
    if path.exists(ckpt_dirpath):
        shutil.rmtree(ckpt_dirpath)
    os.makedirs(ckpt_dirpath)

    ckpt_buckets = range(0, len(data), CHECKPOINT_LEN)
    for i in ckpt_buckets:
        ckpt_filepath = path.join(
            ckpt_dirpath, "{}-{}.pkl".format(ckpt_basename, i / CHECKPOINT_LEN))
        with open(ckpt_filepath, "wb") as ckpt_file:
            pickle.dump(data[i:i + CHECKPOINT_LEN], ckpt_file)
    print("Saved {} checkpoint files in: {}".format(
        len(ckpt_buckets), ckpt_dirpath))


def load_ckpt(dirpath):
    print("Loading checkpoint dir: {}".format(dirpath))
    data = []
    ckpt_filenames = os.listdir(dirpath)
    print("Found {} checkpoint files in: {}".format(
        len(ckpt_filenames), dirpath))
    for ckpt in ckpt_filenames:
        with open(path.join(dirpath, ckpt), "r") as ckpt_file:
            data.extend(pickle.load(ckpt_file))
    return data


def rebuild(data, dataset):
    print("Rebuilding...")
    for idx, result in enumerate(data):
        if "video_output_path" in result:
            vid_path = result["video_output_path"]
            if result["bitrate"] != "-1k" and path.exists(vid_path):
                data[idx]["bitrate"] = "{}k".format(
                    get_bitrate_of_file(vid_path, dataset))
        if result["bitrate"] == "{}k".format(sys.maxsize):
            data[idx]["bitrate"] = "-1k"
    return data


def get_one_pt(one_exp, dataset, scratch_dir, do_ff, do_rebuild, task, mc_to_test, num_frames):
    all_results = []
    # Only process a single experiment
    confs_filepath = [fn for fn in one_exp if fn.endswith("npy")][0]
    labels_filepath = [fn for fn in one_exp if fn.endswith("h5")][0]
    labels = h5py.File(labels_filepath, 'r')['labels'][:]
    confs = process_confs(np.load(confs_filepath))
    confs = (np.asarray(confs) >= 0.5).astype(int)
    max_len = min(len(confs), len(labels))
    labels = labels[:max_len]
    confs = confs[:max_len]

    bitrate_str = one_exp[2]
    bitrate = DATASET_MAP[dataset]["original_bitrate"] \
              if bitrate_str == "baseline" else bitrate_str

    mc_name = one_exp[3]

    if num_frames > 0:
      labels = labels[:num_frames]
      confs = confs[:num_frames]
    else:
      labels = labels[:len(confs)]
      num_frames = len(confs)
    parameters = {
        "bitrate": bitrate,
        "resolution": DATASET_MAP[dataset]["resolution"],
        "fps": 15,
        "microclassifier": mc_name,
        "task": task,
        "one_exp": one_exp
    }

    print("Getting metrics for: No FF")
    metrics = event_metrics.get_all_metrics(confs, labels)
    metrics.update(parameters)
    all_results.append(metrics)

    if do_ff and bitrate_str == SUFFICIENT_QUALITY[task][mc_to_test]:
        print("Getting metrics for: FF")
        # Make the scratch dir and make sure it's empty
        if path.exists(scratch_dir) and not do_rebuild:
            shutil.rmtree(scratch_dir)
        os.makedirs(scratch_dir)

        # Now apply filterforward to the baseline output
        ff_confs = filterforward.smooth_confs(
            confs, k=5, pessimistic=False)
        frame_indices = np.transpose(np.nonzero(ff_confs)).flatten()
        video_output_path = path.join(
            scratch_dir, "{}.mp4".format(uuid.uuid4()))
        parameters = {
            "frame_indices": frame_indices,
            "video_output_path": video_output_path,
            "bitrate": "-1k",
            "resolution": DATASET_MAP[dataset]["resolution"],
            "fps": 15,
            "task": task,
            "filterforward": True,
            "one_exp": one_exp,
            "microclassifier": mc_name,
            "num_frames": num_frames
        }
        metrics = event_metrics.get_all_metrics(confs, labels)
        metrics.update(parameters)
        all_results.append(metrics)

    return all_results


def get_bitrate_of_file(filepath, dataset, num_frames=0):
    ratio = 1.0
    if num_frames > 0:
        ratio = num_frames / float(DATASET_MAP[dataset]["num_frames"])
    bandwidth = path.getsize(filepath) / (ratio * float(DATASET_MAP[dataset]["video_len_s"]))
    # Convert to bits -> divide by 1000^2 to go from bits to megabits
    bandwidth = float(bandwidth) * 8. / 1000.
    return bandwidth


def main():
    parser = argparse.ArgumentParser(
        description="Parses bitrate experiment results.")
    parser.add_argument(
        "--parse-checkpoint", type=str,
        help="Path to a checkpoint file to parse.", required=False)
    parser.add_argument(
        "--load-checkpoint", type=str, help=
        "Path to a checkpoint file to resume.", required=False)
    parser.add_argument(
        "--task", type=str,
        help=("\"pedestrian\", \"redtruck\", "
              "\"people_red\", \"police_cars\", \"plastic_bag\""),
        required=True)
    parser.add_argument(
        "--scratch", type=str, help="Scratch dir. Will be deleted.",
        required=True)
    parser.add_argument(
        "--do-ff", action="store_true",
        help=("Whether to run FilterForward to smooth the detections and drop "
              "unwanted frames."),
        required=False)
    parser.add_argument(
        "--fake-encode", action="store_true",
        help=("Skip the extract/encode and use a dummy bandwidth. FOR DEBUGGING "
              "ONLY."))
    parser.add_argument(
        "--rebuild", action="store_true",
        help=("Recalculate bitrates from encoded files in scratch dir, and "
              "discard bitrates set by \"--fake-encode\"."))
    parser.add_argument(
        "--dataset", type=str, help="The name of the dataset.", required=True,
        choices=["jackson", "roadway"])
    parser.add_argument(
        "--out-dir", type=str,
        help="The dir in which to write checkpoint and output files.",
        required=True)
    parser.add_argument(
        "--vid-batch", type=int, help="Encode batch size", required=True)
    parser.add_argument(
        "--do-bitrate", action="store_true",
        help="Calculate the bitrate for FilterForward.", required=False)
    parser.add_argument(
        "--fs2-dir", type=str,
        help=("The path to the fawnstore2, where all of the large data files "
              "are stored."),
        required=False)
    parser.add_argument(
        "--one-exp", type=str,
        help=("Process only a single experiment. First argument is the "
              "confs.npy file and the second is the labels HDF5 file. Does "
              "support bitrate calculation and checkpointing."),
        nargs=4, required=False)
    parser.add_argument(
        "--ckpt-root", type=str,
        help="Directory holding run_classifiers checkpoints.")
    parser.add_argument(
        "--labels-root", type=str,
        help="Directory holding labels for your particular video.")
    parser.add_argument(
        "--num-frames", type=int, default=0,
        help="Number of frames to process. 0 to use length of confs w/ full video")
    parser.add_argument("--mc-to-test", type=str)
    args = parser.parse_args()
    ckpt_to_parse = args.parse_checkpoint
    ckpt_to_load = args.load_checkpoint
    task = args.task
    fs2_dir = args.fs2_dir
    scratch_dir = args.scratch
    do_ff = args.do_ff
    fake_encode = args.fake_encode
    do_rebuild = args.rebuild
    dataset = args.dataset
    out_dir = args.out_dir
    vid_batch_size = args.vid_batch
    one_exp = args.one_exp
    ckpt_root = args.ckpt_root
    labels_root = args.labels_root
    mc_to_test = args.mc_to_test
    num_frames = args.num_frames
    if one_exp is not None and ckpt_root is not None:
        print("--one-exp and --ckpt-root are incompatible with each other.")
    if one_exp is not None and task is None:
        print("Must provide --task if --one-exp is specified.")
    if not fake_encode and fs2_dir is None:
        print("Must provide --fs2-dir if --fake-encode is not specified.")

    if ckpt_to_parse is not None:
        # Parse checkpoint
        print("Parsing checkpoint: {}".format(ckpt_to_parse))
        data = load_ckpt(ckpt_to_parse)
        if do_rebuild:
            data = rebuild(data, dataset)
        print_results(data, "{}-{}-checkpoint.csv".format(dataset, task))
        return 0

    outfile = path.join(out_dir, "{}-{}-{}-results.csv".format(
        dataset, task, mc_to_test))
    all_results = []
    # If we are resuming from checkpoint,
    exps_to_run = []
    if ckpt_to_load is not None:
        all_results = load_ckpt(ckpt_to_load)
        if do_rebuild:
            all_results = rebuild(all_results, dataset)
    if one_exp is not None:
        exps_to_run = [one_exp]
        all_results = []
    else:
        one_exps = get_one_exps(ckpt_root, labels_root, dataset, task, mc_to_test, num_trials=None)

        for exp, label_path, bitrate, mc_name in one_exps:
            exp_clean = exp.replace(ckpt_root, "")
            found = False
            for idx, result in enumerate(all_results):
                if exp_clean in result["one_exp"][0]:
                    result["one_exp"] = (exp, label_path, bitrate, mc_name)
                    found = True
            if not found:
                exps_to_run.append((exp, label_path, bitrate, mc_name))

    for exp, labels_path, bitrate, mc_name in exps_to_run:
        print("RUN POINT: {} {}".format(exp, labels_path))
        new_results = get_one_pt([exp, labels_path, bitrate, mc_name], dataset, scratch_dir, do_ff,
                                 do_rebuild, task, mc_to_test, num_frames)
        all_results += new_results

    # Look through all_results and select the experiments that don't have
    # bandwidth metrics yet, adding them to the todo list.
    print("Finding experiments with no bitrate metric")
    bitrate_calculation_input = []
    max_num_frames = 0
    for idx, result in enumerate(all_results):
        if result["microclassifier"] == mc_to_test:  # TODO: TERRIBLE HACK
            if result["bitrate"] == "-1k":
                bitrate_calculation_input.append(
                    (result["frame_indices"], result["video_output_path"], idx, result["num_frames"]))
                if max_num_frames < result["num_frames"]:
                    max_num_frames = result["num_frames"]

    print("Starting extract/encode stage")
    save_ckpt(all_results, dataset, task, out_dir)

    print_results(all_results, outfile)

    for i in range(0, len(bitrate_calculation_input), vid_batch_size):
        print("Working on tasks {} through {} of {}".format(
            i, i + vid_batch_size - 1, len(bitrate_calculation_input) - 1))
        batch = bitrate_calculation_input[i:i + vid_batch_size]
        confs = [(frame_indices, output_path)
                 for frame_indices, output_path, _, _ in batch]
        if not fake_encode:
            extract_frames.extract_frames(
                in_filepath=path.join(
                    fs2_dir, DATASET_MAP[dataset]["sufficient_quality_path"].format(
                        SUFFICIENT_QUALITY[task][mc_to_test])),
                confs=confs,
                total_frames=max_num_frames,
                out_bitrate=SUFFICIENT_QUALITY[task][mc_to_test],
                num_threads=0)
        for _, _, results_idx, num_frames in batch:
            if fake_encode:
                # Debug mode
                print("DEBUG MODE: Skipped encode for: {}".format(
                    all_results[results_idx]["video_output_path"]))
                bandwidth = sys.maxsize
            else:
                bandwidth = get_bitrate_of_file(
                    all_results[results_idx]["video_output_path"], dataset, num_frames)

            all_results[results_idx]["bitrate"] = "{}k".format(int(bandwidth))

        # Update the checkpoint so this step wasn't worthless.
        save_ckpt(all_results, dataset, task, out_dir)

    # Print results to csv
    print_results(all_results, outfile)
    return 0


if __name__ == "__main__":
    main()
