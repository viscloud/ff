
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

python3 classifiers.py --train-frames /fawnstore2/ff/serv/vimba_iii_3_2018-3-21_15-15fps-1000k-frames/ --test-frames /fawnstore2/ff/serv/vimba_iii_3_2018-3-21_16-15fps-1000k-frames/ --train-labels labeler/labels/vimba_iii_3_2018-3-21_15/people_red/labels-15fps.h5 --checkpoint-dir ./exp-out/out-1000k/objdet/trial0 --model-name "objdet" --num-epochs 1.0 --batch 16 --ckpt-interval 10 --max-threads 8 --input-height 850 --input-width 2048 --max-queue-len 6 --test-labels labeler/labels/vimba_iii_3_2018-3-21_16/people_red/labels-15fps.h5 --num-gpu 2
