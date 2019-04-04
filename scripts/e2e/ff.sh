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

# Usage: ./filterforward_perf.sh MODEL DO_MEM MC_TYPE MC_BATCH_SIZE TOKENS EXP_LIMIT NUM_FRAMES DRY_RUN
# DO_MEM = 1: Run memory experiments
# DO_MEM != 1: Run CPU experminets
# MC_TYPE = {1x1_objdet, spatial_crop, windowed}
# Only trial values less than EXP_LIMIT will be performed. This allows the user
# to skip the experiments too intensive for the machine (e.g., mobilenet512
# beyond 5 copies)
# DRY_RUN = 1: Do not execute experiment commands.

MODEL=$1
DO_MEM=$2
MC_TYPE=$3
MC_BATCH_SIZE=$4
TOKENS=$5
EXP_LIMIT=$6
NUM_FRAMES=$7
DRY_RUN=$8

FF_SCRIPTS_PATH="set this"
CONFIG_DIR="$FF_SCRIPTS_PATH/e2e/configs/saf_config"
FF_CONF_DIR="$FF_SCRIPTS_PATH/e2e/configs/ff_confs/$MC_TYPE"
OUT_DIR="~/out/ff"
CAMERA="FILE"
NNE_BATCH_SIZE=8
QUEUE_SIZE=8
LOG_LEVEL=0
KV_WINDOW=5
FILE_FPS=15
NUMS_COPIES=( 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 )
FIELDS=("frame_id")
STACKED_FLAG=""

OUT_SUFFIX=$MC_TYPE"_"
if [ $DO_MEM = "1" ]; then
    echo "Running memory experiments"
    MEM_FLAG="--memory-usage"
    OUT_SUFFIX=$OUT_SUFFIX"mem"
else
    echo "Running CPU experiments"
    MEM_FLAG=""
    OUT_SUFFIX=$OUT_SUFFIX"cpu"
fi

echo "MC type: $MC_TYPE"
echo "Output dir: $OUT_DIR"
mkdir -p $OUT_DIR

make filterforward -j8
for NUM_COPIES in ${NUMS_COPIES[@]}; do
    if [ $NUM_COPIES -lt $EXP_LIMIT ]; then
        FF_CONF=$FF_CONF_DIR/config-$NUM_COPIES.csv
        OUT_FILE=$OUT_DIR/ff_$NUM_COPIES\_$OUT_SUFFIX.csv
        echo "Generating: $OUT_FILE"
        echo "Using conf: $FF_CONF"
        echo "Running: ./src/filterforward --config-dir $CONFIG_DIR --ff-conf $FF_CONF --camera $CAMERA --model $MODEL --fields ${FIELDS[@]} --tokens $TOKENS --queue-size $QUEUE_SIZE --num-frames $NUM_FRAMES --nne-batch-size $NNE_BATCH_SIZE --kv-window $KV_WINDOW --output-dir $OUT_DIR --log-file $OUT_FILE --mc-batch-size $MC_BATCH_SIZE --file-fps $FILE_FPS --block --mc-pass-all --force-kill $STACKED_FLAG $MEM_FLAG"
        if [ $DRY_RUN != "1" ]; then
            GLOG_minloglevel=$LOG_LEVEL ./src/filterforward \
                            --config-dir $CONFIG_DIR \
                            --ff-conf $FF_CONF \
                            --camera $CAMERA \
                            --model $MODEL \
                            --fields ${FIELDS[@]} \
                            --tokens $TOKENS \
                            --queue-size $QUEUE_SIZE \
                            --num-frames $NUM_FRAMES \
                            --nne-batch-size $NNE_BATCH_SIZE \
                            --kv-window $KV_WINDOW \
                            --output-dir $OUT_DIR \
                            --log-file $OUT_FILE \
                            --mc-batch-size $MC_BATCH_SIZE \
                            --file-fps $FILE_FPS \
                            --block \
                            --mc-pass-all \
                            --force-kill \
                            $STACKED_FLAG \
                            $MEM_FLAG
        fi
    fi
done
