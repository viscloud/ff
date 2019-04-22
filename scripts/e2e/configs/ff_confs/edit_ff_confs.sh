
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


FILES=(./1x1_objdet/config-1.csv \
       ./1x1_objdet/config-10.csv \
       ./1x1_objdet/config-15.csv \
       ./1x1_objdet/config-2.csv \
       ./1x1_objdet/config-20.csv \
       ./1x1_objdet/config-25.csv \
       ./1x1_objdet/config-3.csv \
       ./1x1_objdet/config-30.csv \
       ./1x1_objdet/config-35.csv \
       ./1x1_objdet/config-4.csv \
       ./1x1_objdet/config-40.csv \
       ./1x1_objdet/config-45.csv \
       ./1x1_objdet/config-5.csv \
       ./1x1_objdet/config-50.csv \
       ./1x1_objdet/config-55.csv \
       ./1x1_objdet/config-6.csv \
       ./1x1_objdet/config-60.csv \
       ./1x1_objdet/config-7.csv \
       ./1x1_objdet/config-8.csv \
       ./1x1_objdet/config-9.csv \
       ./spatial_crop/config-1.csv \
       ./spatial_crop/config-10.csv \
       ./spatial_crop/config-15.csv \
       ./spatial_crop/config-2.csv \
       ./spatial_crop/config-20.csv \
       ./spatial_crop/config-25.csv \
       ./spatial_crop/config-3.csv \
       ./spatial_crop/config-30.csv \
       ./spatial_crop/config-35.csv \
       ./spatial_crop/config-4.csv \
       ./spatial_crop/config-40.csv \
       ./spatial_crop/config-45.csv \
       ./spatial_crop/config-5.csv \
       ./spatial_crop/config-50.csv \
       ./spatial_crop/config-55.csv \
       ./spatial_crop/config-6.csv \
       ./spatial_crop/config-60.csv \
       ./spatial_crop/config-7.csv \
       ./spatial_crop/config-8.csv \
       ./spatial_crop/config-9.csv \
       ./windowed/config-1.csv \
       ./windowed/config-10.csv \
       ./windowed/config-15.csv \
       ./windowed/config-2.csv \
       ./windowed/config-20.csv \
       ./windowed/config-25.csv \
       ./windowed/config-3.csv \
       ./windowed/config-30.csv \
       ./windowed/config-35.csv \
       ./windowed/config-4.csv \
       ./windowed/config-40.csv \
       ./windowed/config-45.csv \
       ./windowed/config-5.csv \
       ./windowed/config-50.csv \
       ./windowed/config-55.csv \
       ./windowed/config-6.csv \
       ./windowed/config-60.csv \
       ./windowed/config-7.csv \
       ./windowed/config-8.csv \
       ./windowed/config-9.csv)

for FILE in ${FILES[@]}; do
    echo "Editing: $FILE"
    sed -i 's+/home/ccanel/src/filterforward-doc/papers/ff+~/src/filterforward+g' $FILE
done
