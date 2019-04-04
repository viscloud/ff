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

cmake_minimum_required(VERSION 3.0)
project(filterforward)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/clangformat.cmake)
include(cmake/build.cmake)

# Custom Find*.cmake files.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(TensorFlow REQUIRED)

include_directories(SYSTEM
  ${GSTREAMER_INCLUDE_DIRS}
  ${TensorFlow_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  src
  third_party/saf/src
  third_party/saf/3rdparty
  third_party/saf/3rdparty/nsync/public)

add_subdirectory(third_party)
add_subdirectory(src)
