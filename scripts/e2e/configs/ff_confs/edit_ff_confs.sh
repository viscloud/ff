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


       # ./window5_pedestrian_uc/config-60.csv \
       # ./window5_pedestrian_uc/config-20.csv \
       # ./window5_pedestrian_uc/config-5.csv \
       # ./window5_pedestrian_uc/config-10.csv \
       # ./window5_pedestrian_uc/config-50.csv \
       # ./window5_pedestrian_uc/config-40.csv \
       # ./window5_pedestrian_uc/config-30.csv \
       # ./window5_pedestrian_uc/config-1.csv \
# FILES=(./person_classifier_1x1_objdet/config-60.csv \
#        ./person_classifier_1x1_objdet/config-35.csv \
#        ./person_classifier_1x1_objdet/config-20.csv \
#        ./person_classifier_1x1_objdet/config-25.csv \
#        ./person_classifier_1x1_objdet/config-5.csv \
#        ./person_classifier_1x1_objdet/config-10.csv \
#        ./person_classifier_1x1_objdet/config-50.csv \
#        ./person_classifier_1x1_objdet/config-40.csv \
#        ./person_classifier_1x1_objdet/config-15.csv \
#        ./person_classifier_1x1_objdet/config-55.csv \
#        ./person_classifier_1x1_objdet/config-30.csv \
#        ./person_classifier_1x1_objdet/config-45.csv \
#        ./person_classifier_1x1_objdet/config-1.csv)
FILES=(./windowed1/config-60.csv \
       ./windowed1/config-35.csv \
       ./windowed1/config-20.csv \
       ./windowed1/config-25.csv \
       ./windowed1/config-5.csv \
       ./windowed1/config-10.csv \
       ./windowed1/config-50.csv \
       ./windowed1/config-40.csv \
       ./windowed1/config-15.csv \
       ./windowed1/config-55.csv \
       ./windowed1/config-30.csv \
       ./windowed1/config-45.csv \
       ./windowed1/config-1.csv)
# FILES=(./person_gr8_classifier/config-60.csv \
#        ./person_gr8_classifier/config-35.csv \
#        ./person_gr8_classifier/config-20.csv \
#        ./person_gr8_classifier/config-25.csv \
#        ./person_gr8_classifier/config-5.csv \
#        ./person_gr8_classifier/config-10.csv \
#        ./person_gr8_classifier/config-50.csv \
#        ./person_gr8_classifier/config-40.csv \
#        ./person_gr8_classifier/config-15.csv \
#        ./person_gr8_classifier/config-55.csv \
#        ./person_gr8_classifier/config-30.csv \
#        ./person_gr8_classifier/config-45.csv \
#        ./person_gr8_classifier/config-1.csv)

for FILE in ${FILES[@]}; do
    echo "Editing: $FILE"
    sed -i 's/conv2d_1_input/inputs/g' $FILE
done
