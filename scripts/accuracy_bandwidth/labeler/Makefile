
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


labeler : labeler.cpp
	$(CXX) labeler.cpp -o labeler `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_program_options --std=c++14 -Wextra -Wall -Wpedantic

clean:
	rm -f labeler

clangformat:
	clang-format --style=Google -i labeler.cpp
