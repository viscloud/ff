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


FF_SCRIPTS_PATH="set this"
EXP_LIMIT=51
# NUM_FRAMES=1000
# NUM_FRAMES=500
NUM_FRAMES=300
DRY_RUN="0"

# FilterForward.
# 1x1_objdet
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 0 "1x1_objdet" 16 16 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 1 "1x1_objdet" 16 24 $EXP_LIMIT $NUM_FRAMES $DRY_RUN

# spatial_crop
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 0 "spatial_crop" 8 8 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 1 "spatial_crop" 8 16 $EXP_LIMIT $NUM_FRAMES $DRY_RUN

# windowed
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 0 "windowed" 8 8 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/ff.sh $FF_SCRIPTS_PATH mobilenet 1 "windowed" 8 16 $EXP_LIMIT $NUM_FRAMES $DRY_RUN

# Discrete classifier (dc).
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/nnbench.sh $FF_SCRIPTS_PATH dc 8 0 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/nnbench.sh $FF_SCRIPTS_PATH dc 8 1 $EXP_LIMIT $NUM_FRAMES $DRY_RUN

EXP_LIMIT=31
NUM_FRAMES=80
# MobileNet. Batch size based on number of classifiers: 1-4: 8, 5-8: 4, 9-15: 2, 20-30: 1
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/nnbench.sh $FF_SCRIPTS_PATH mobilenet 1 0 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
# CUDA_VISIBLE_DEVICES="" $FF_SCRIPTS_PATH/e2e/nnbench.sh $FF_SCRIPTS_PATH mobilenet 1 1 $EXP_LIMIT $NUM_FRAMES $DRY_RUN
