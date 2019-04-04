
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

import os 
import sys

partial_start_str = "Partial Start"
partial_end_str = "Partial Finish"
full_start_str = "Full Start"
full_end_str = "Full Finish"

def read_file(filename, data_points):
    print(filename);
    f = open(filename);
    elems = []
    for line in f:
        split_line = line.rstrip().split(":");
        if (int(split_line[1]) in data_points):
            print(line, data_points[int(split_line[1])])
        data_points[int(split_line[1])] = split_line[0];
        elems.append((int(split_line[1]), split_line[0]))
    elems = sorted(elems, key=lambda x: x[0])
    stack = []
    for elem in elems:
        if elem[1] in [x[1] for x in stack]:
            sys.exit(1)
        if elem[1] == full_end_str:
            if stack[-1][1] != full_start_str:
                print(stack, elem)
                sys.exit(1)
            stack.pop()
        elif elem[1] == partial_end_str:
            print(stack, elem)
            if stack[-1][1] != partial_start_str:
                print(stack, elem)
                sys.exit(1)
            stack.pop()
        else:
            stack.append(elem)

data_dir = 'new_jh_1_pedestrians'
label_files = os.listdir(data_dir);
data_points = dict()
for file in label_files:
    if(file != "readme" and 'swp' not in file):
        read_file(data_dir + '/' + file, data_points);

print(data_dir)
