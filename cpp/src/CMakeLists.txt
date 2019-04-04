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

macro(build APP)
  target_link_libraries(${APP}
    saf
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
  add_build_reqs(${APP})
endmacro()

# filterforward
add_executable(filterforward
  ff/deduplicator.cpp
  ff/deduplicator.h
  ff/ed.cpp
  ff/ed.h
  ff/filter.cpp
  ff/filter.h
  ff/filterforward.cpp
  ff/imagematch.cpp
  ff/imagematch.h
  ff/fv_gen.h
  ff/fv_gen.cpp)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIR})
target_link_libraries(filterforward ${CURL_LIBRARIES})
build(filterforward)

# nnbench
add_executable(nnbench
  nnbench/nnbench_app.cpp
  nnbench/nnbench.h
  nnbench/nnbench.cpp)
build(nnbench)
