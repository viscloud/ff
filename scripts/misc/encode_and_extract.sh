#!/bin/bash

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


VID_DIR=~/videos
OUT_DIR=/data/frames
FRAMES_PER_DIR=5000
FORMAT="png"


########## Jackson ##########

# Baseline

# Training video.
# TRAINING_VID=$VID_DIR/new_jh_1-15fps-baseline.mp4
# BITRATE=1905k
# ffmpeg -y -i $VID_DIR/new_jh_1-30fps-baseline.mp4 -r 15 -b:v $BITRATE -an -codec:v libx264 $TRAINING_VID
# python3 vid_to_imgs.py --vid $TRAINING_VID --out-dir $OUT_DIR/new_jh_1-15fps-baseline-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323932 --fps 15

# Testing video.
# BITRATE=1883k
# TESTING_VID=$VID_DIR/new_jh_3-15fps-baseline.mp4
# ffmpeg -y -i $VID_DIR/new_jh_3-30fps-baseline.mp4 -r 15 -b:v $BITRATE -an -codec:v libx264 $TESTING_VID
# python3 vid_to_imgs.py --vid $TESTING_VID --out-dir $OUT_DIR/new_jh_3-15fps-baseline-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323730 --fps 15


# 1000 kbps
# BITRATE=1000k

# Training video.
# TRAINING_VID=$VID_DIR/new_jh_1-15fps-$BITRATE.mp4
# ffmpeg -y -i $VID_DIR/new_jh_1-30fps-baseline.mp4 -r 15 -b:v $BITRATE -an -codec:v libx264 $TRAINING_VID
# python3 vid_to_imgs.py --vid $TRAINING_VID --out-dir $OUT_DIR/new_jh_1-15fps-$BITRATE-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323932 --fps 15

# Testing video.
# TESTING_VID=$VID_DIR/new_jh_3-15fps-$BITRATE.mp4
# ffmpeg -y -i $VID_DIR/new_jh_3-30fps-baseline.mp4 -r 15 -b:v BITRATE -an -codec:v libx264 $TESTING_VID
# python3 vid_to_imgs.py --vid $TESTING_VID --out-dir $OUT_DIR/new_jh_3-15fps-$BITRATE-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323730 --fps 15


# 500 kbps
# BITRATE=500k

# Training video.
# TRAINING_VID=$VID_DIR/new_jh_1-15fps-$BITRATE.mp4
# ffmpeg -y -i $VID_DIR/new_jh_1-30fps-baseline.mp4 -r 15 -b:v $BITRATE -an -codec:v libx264 $TRAINING_VID
# python3 vid_to_imgs.py --vid $TRAINING_VID --out-dir $OUT_DIR/new_jh_1-15fps-$BITRATE-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323932 --fps 15

# Testing video.
# TESTING_VID=$VID_DIR/new_jh_3-15fps-$BITRATE.mp4
# ffmpeg -y -i $VID_DIR/new_jh_3-30fps-baseline.mp4 -r 15 -b:v $BITRATE -an -codec:v libx264 $TESTING_VID
# python3 vid_to_imgs.py --vid $TESTING_VID --out-dir $OUT_DIR/new_jh_3-15fps-$BITRATE-frames --num-per-dir $FRAMES_PER_DIR --format $FORMAT --total-frames 323730 --fps 15


########## III ##########

BASELINE_VID="$VID_DIR/vimba_iii_3_2018-3-21_15-30fps-baseline.mp4"
TRAINING_VID_PREFIX="vimba_iii_3_2018-3-21_15-15fps"
TESTING_VID_PREFIX="vimba_iii_3_2018-3-21_16-15fps"


# 1000 kbps
BITRATE="1000k"

# Training video.
TRAINING_VID="$VID_DIR/$TRAINING_VID_PREFIX-$BITRATE.mp4"
ffmpeg -y -i $BASELINE_VID -r 15 -b:v $BITRATE -an -codec:v libx264 $TRAINING_VID
python3 vid_to_imgs.py \
        --vid $TRAINING_VID \
        --out-dir "$OUT_DIR/$TRAINING_VID_PREFIX-$BITRATE-frames" \
        --num-per-dir $FRAMES_PER_DIR \
        --format $FORMAT \
        --total-frames 161997 \
        --fps 15

# Testing video.
TESTING_VID="$VID_DIR/$TESTING_VID_PREFIX-$BITRATE.mp4"
ffmpeg -y -i $BASELINE_VID -r 15 -b:v $BITRATE -an -codec:v libx264 $TESTING_VID
python3 vid_to_imgs.py \
        --vid $TESTING_VID \
        --out-dir "$OUT_DIR/$TESTING_VID_PREFIX-$BITRATE-frames" \
        --num-per-dir $FRAMES_PER_DIR \
        --format $FORMAT \
        --total-frames 162012 \
        --fps 15
