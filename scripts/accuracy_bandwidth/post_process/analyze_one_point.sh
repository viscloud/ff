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


TASK="people_red"

python3 ./parse_one_result.py --task $TASK --scratch ~/scratch --dataset iii3 --out-dir ./results --vid-batch 16 --scripts-dir .. --fs2-dir /fawnstore2/ff/ckpts/iii3/people_red/ --one-exp $1 ~/src/labeler/labels/vimba_iii_3_2018-3-21_16/$TASK/labels-15fps.h5 a a
