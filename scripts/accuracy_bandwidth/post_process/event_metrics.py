
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

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


LABEL_START_MARKER = 42
LABEL_END_MARKER = 43
LABEL_STARTEND_MARKER = 44


def get_all_metrics(confs, labels):
  if not len(confs) == len(labels):
    print("conf label mismatch")
    return {}
  labels = np.asarray(labels, dtype=int)
  confs = np.asarray(confs, dtype=int)

  # Event metrics
  event_recall = compute_event_recall(
    confs, labels, existence_coeff=0.9, overlap_coeff=0.1)
  event_coverage = compute_event_recall(
    confs, labels, existence_coeff=1.0, overlap_coeff=0.0)
  event_overlap = compute_event_recall(
    confs, labels, existence_coeff=0.0, overlap_coeff=1.0)
  print("Event - Recall: {} Coverage: {} Overlap: {}".format(
    event_recall, event_coverage, event_overlap))

  # Before calculating precision, filter out confs and labels if label is -1.
  # For the recall calculations above, this is handled in
  # compute_event_recall().
  confs, labels = zip(*[(c, l) for c, l in zip(confs, labels) if l != -1])
  labels = np.asarray(labels, dtype=int)
  confs = np.asarray(confs, dtype=int)

  # per-frame metrics
  num_ones = np.sum(labels)
  num_zeros = len(labels) - num_ones

  fp = len([p for p, l in zip(confs, labels) if p >= 0.5 and l == 0])
  fn = len([p for p, l in zip(confs, labels) if p < 0.5 and l == 1])
  # Total positives - false positives
  tp = len([p for p in confs if p >= 0.5]) - fp

  recall = float(tp) / float(tp + fn)
  precision = get_precision(confs, labels)

  f1 = 2.0 * float(precision * recall) / float(precision + recall)
  event_f1 = 2.0 * float(precision * event_recall) / float(precision + event_recall)
  print("Event f1: {}".format(event_f1))

  print("fp: {} fn: {} tp: {} f1: {}".format(fp, fn, tp, f1))

  results = {
    "event_recall": event_recall,
    "event_coverage": event_coverage,
    "event_overlap": event_overlap,
    "per_frame_fp": fp,
    "per_frame_tp": tp,
    "per_frame_fn": fn,
    "per_frame_recall": recall,
    "per_frame_precision": precision,
    "f1": f1,
    "event_f1": event_f1
  }
  return results

def compute_event_start_end(labels):
    """
    Returns a copy of labels where the first and last frame in each event have
    replaced with a special marker.
    """
    marked_labels = np.copy(labels)
    prev_val = 0
    first = True
    for i, val in enumerate(labels):
        if first:
            first = False
            if val == 1:
                marked_labels[i] = LABEL_START_MARKER
        else:
            if prev_val == 1:
                # Previous frame was in an event.
                if val != 1:
                    # This is the first non-event frame, so mark the previous
                    # frame.
                    if marked_labels[i - 1] == LABEL_START_MARKER:
                        marked_labels[i - 1] = LABEL_STARTEND_MARKER
                    else:
                        marked_labels[i - 1] = LABEL_END_MARKER
            else:
                # prev_val was a 0 or -1.
                if val == 1:
                    # This is the start of an event.
                    marked_labels[i] = LABEL_START_MARKER
        prev_val = val
    if marked_labels[-1] == 1:
        marked_labels[-1] = LABEL_END_MARKER
    if marked_labels[-1] == LABEL_START_MARKER:
        marked_labels[-1] = LABEL_STARTEND_MARKER
    return marked_labels

def compute_iou(window1, window2):
    start1, end1 = window1
    start2, end2 = window2
    start, end = min(start1, start2), max(end1, end2)
    union = float(end - start)
    # Create buffer then take the element-wise AND.
    elem_buffer = np.zeros(end - start)
    elem_buffer[start1-start:end1-start] += 1
    elem_buffer[start2-start:end2-start] += 1
    intersection = np.sum(elem_buffer == 2)
    return intersection / union

def compute_event_recall(preds, labels, existence_coeff=0.5, overlap_coeff=0.5):
    # Mark the start and end frames in each event
    marked_labels = compute_event_start_end(labels)
    # Drop the uncertain frames. The marked frames still show where the original
    # events started and ended.
    filtered_preds, filtered_labels = zip(
        *[(c, l) for c, l in zip(preds, marked_labels) if l != -1])
    # Extract the positions of the start and end events.
    label_starts = [
        i for i, val in enumerate(filtered_labels) if val == LABEL_START_MARKER or val == LABEL_STARTEND_MARKER]
    label_ends = [
        i+1 for i, val in enumerate(filtered_labels) if val == LABEL_END_MARKER or val == LABEL_STARTEND_MARKER]
    assert(len(label_starts) == len(label_ends))
    print("Number of events in this subset of labels: {}".format(len(label_starts)))

    # (1) Check what fraction of events have at least one prediction.
    events_with_pred = 0
    for e_start, e_end in zip(label_starts, label_ends):
        events_with_pred += int(np.sum(filtered_preds[e_start:e_end]) > 0)
    events_with_pred_score = events_with_pred / float(len(label_starts))
    # (2) Compute the overlap between predicted events and ground truth
    #     events to obtain frame-level event recall.
    event_recalls = []
    for e_start, e_end in zip(label_starts, label_ends):
        assert(e_end > e_start)
        event_overlap = np.sum(filtered_preds[e_start:e_end]) / float(e_end - e_start)
        event_recalls.append(event_overlap)
    event_overlap_score = np.mean(event_recalls)
    return existence_coeff * events_with_pred_score + overlap_coeff * event_overlap_score

def get_precision(confs, labels, pred_thresh=0.5):
    preds = np.asarray(confs) >= pred_thresh
    labels = [l == 1 for l in labels]
    return precision_score(labels, preds)


classifier_to_multiply_adds = {
    "larger_cnn4" :     17698112,
    "larger_cnn5" :    118227072,
    "larger_cnn3" :    164626560,
    "larger_sep_cnn4" :  6650566,
    "larger_sep_cnn5" : 17768576,
    "larger_sep_cnn3" : 26723072,
    "1x1_simulated_tile" : 8659200,
    "spatial_crop_mc(larry)": 7456968,
    "windowed (elizabeth)": 33980616
}

classifier_to_nice_name = {
    "larger_cnn4" : "cnn1",
    "larger_cnn5" : "cnn2",
    "larger_cnn3" : "cnn3",
    "larger_sep_cnn4" : "sep cnn1",
    "larger_sep_cnn5" : "sep cnn2",
    "larger_sep_cnn3" : "sep cnn3",
    "1x1_simulated_tile" : "full-frame object detector"
}
