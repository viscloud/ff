
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


Parse "iii3/people_red" results for "1x1_objdet" on `mercury`:
```sh
cd /home/ccanel/src/filterforward-doc/papers/ff/scripts/accuracy_bandwidth/post_process
git checkout b45693072c9cbd395c12eb51bbde267e60fadc6e
python parse_one_result.py --task people_red --scratch ~/scratch --do-ff --dataset iii3 --out-dir ./results --vid-batch 2 --fs2-dir /fawnstore2/ff/scripts/bitrate --scripts-dir ../.. --ckpt-root /fawnstore2/ff/ckpts/iii3 --labels-root ~/src/labeler/labels/vimba_iii_3_2018-3-21_16 --mc-to-test 1x1_objdet --do-ff
```

Parse "iii3/people_red" results for "spatial_crop" on `mercury`:
```sh
cd /home/ccanel/src/filterforward-doc/papers/ff/scripts/accuracy_bandwidth/post_process
git checkout b45693072c9cbd395c12eb51bbde267e60fadc6e
python parse_one_result.py --task people_red --scratch ~/scratch --do-ff --dataset iii3 --out-dir ./results --vid-batch 2 --fs2-dir /fawnstore2/ff/scripts/bitrate --scripts-dir ../.. --ckpt-root /fawnstore2/ff/ckpts/iii3 --labels-root ~/src/labeler/labels/vimba_iii_3_2018-3-21_16 --mc-to-test spatial_crop --do-ff
```
