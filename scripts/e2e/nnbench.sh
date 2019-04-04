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

# Usage: ./nnbench.sh MODEL BATCH_SIZE DO_MEM EXP_LIMIT NUM_FRAMES DRY_RUN
# DO_MEM = 1: Run memory experiments
# DO_MEM != 1: Run CPU experminets
# Only trial values less than EXP_LIMIT will be performed. This allows the user
# to skip the experiments too intensive for the machine (e.g., mobilenet512
# beyond 5 copies)
# DRY_RUN = 1: Do not execute experiment commands.

MODEL=$1
BATCH_SIZE=$2
DO_MEM=$3
EXP_LIMIT=$4
NUM_FRAMES=$5
DRY_RUN=$6

CONFIG_DIR="/home/ccanel/src/filterforward-doc/papers/ff/scripts/e2e/configs/saf_config"
OUT_DIR="~/out/nnbench"
CAMERA="FILE"
LOG_LEVEL=0
NUMS_COPIES=( 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 )

if [ $DO_MEM = "1" ]; then
    echo "Running memory experiments"
    MEM_FLAG="--memory"
    OUT_SUFFIX="mem"
else
    echo "Running CPU experiments"
    MEM_FLAG=""
    OUT_SUFFIX="cpu"
fi

echo "Output dir: $OUT_DIR"
mkdir -p $OUT_DIR

make nnbench -j8
for NUM_COPIES in ${NUMS_COPIES[@]}; do
    if [ $NUM_COPIES -lt $EXP_LIMIT ]; then
        OUT_FILE=$OUT_DIR/$MODEL\_$NUM_COPIES\_$OUT_SUFFIX.csv
        echo "Generating: $OUT_FILE"
        echo "Running: ./src/nnbench -C $CONFIG_DIR -c $CAMERA -m $MODEL -n $NUM_COPIES -b $BATCH_SIZE -q $BATCH_SIZE -f $NUM_FRAMES -o $OUT_FILE $MEM_FLAG"
        if [ $DRY_RUN != "1" ]; then
            GLOG_minloglevel=$LOG_LEVEL ./src/nnbench \
                            -C $CONFIG_DIR \
                            -c $CAMERA \
                            -m $MODEL \
                            -n $NUM_COPIES \
                            -b $BATCH_SIZE \
                            -q $BATCH_SIZE \
                            -f $NUM_FRAMES \
                            -o $OUT_FILE \
                            $MEM_FLAG
        fi
    fi
done
