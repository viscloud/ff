
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

def sepconv(filters, size, stride, ih, iw, ic):
  madds = ih // stride * iw // stride * ic * (size * size + filters)
  return madds, ih // stride, iw // stride, filters

def conv(filters, size, stride, ih, iw, ic):
  madds = ih // stride * iw // stride * ic * (size * size) * filters
  return madds, ih // stride, iw // stride, filters

def fc(filters, ih, iw, ic):
  madds = filters * ih * iw * ic
  return madds, 1, 1, filters

def dc_mod(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = conv(16, 3, 2, h, w, c)
  total_madds[1], h, w, c = conv(32, 3, 2, h, w, c)
  total_madds[2], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[3], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[4], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[5], h, w, c = fc(128, h, w, c)
  total_madds[6], h, w, c = fc(1, h, w, c)
  return sum(total_madds)

def dc(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = conv(32, 3, 2, h, w, c)
  total_madds[1], h, w, c = conv(32, 3, 2, h, w, c)
  total_madds[2], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[3], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[4], h, w, c = conv(64, 3, 2, h, w, c)
  total_madds[5], h, w, c = fc(128, h, w, c)
  total_madds[6], h, w, c = fc(1, h, w, c)
  return sum(total_madds)

def objdet(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = conv(128, 1, 1, h, w, c)
  total_madds[1], h, w, c = conv(64, 1, 1, h, w, c)
  total_madds[2], h, w, c = conv(64, 1, 1, h, w, c)
  total_madds[3], h, w, c = fc(1, h, w, c)
  return sum(total_madds)

def spatial_crop(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = sepconv(16, 3, 1, h, w, c)
  total_madds[1], h, w, c = sepconv(32, 3, 1, h, w, c)
  total_madds[2], h, w, c = fc(200, h, w, c)
  total_madds[3], h, w, c = fc(1, h, w, c)
  return sum(total_madds)

def windowed(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = conv(32, 3, 1, h, w, c)
  c = c * 5
  total_madds[1], h, w, c = conv(32, 3, 1, h, w, c)
  total_madds[2], h, w, c = conv(32, 3, 2, h, w, c)
  total_madds[3], h, w, c = fc(200, h, w, c)
  total_madds[4], h, w, c = fc(1, h, w, c)
  return sum(total_madds)

def mobilenet(h, w, c):
  total_madds = [0] * 1000
  total_madds[0], h, w, c = conv(32, 3, 2, h, w, c)
  total_madds[1], h, w, c = sepconv(64, 3, 1, h, w, c)
  total_madds[2], h, w, c = sepconv(128, 3, 2, h, w, c)
  total_madds[3], h, w, c = sepconv(128, 3, 1, h, w, c)
  total_madds[4], h, w, c = sepconv(256, 3, 2, h, w, c)
  total_madds[5], h, w, c = sepconv(256, 3, 1, h, w, c)
  total_madds[6], h, w, c = sepconv(512, 3, 1, h, w, c)
  total_madds[7], h, w, c = sepconv(512, 3, 1, h, w, c)
  total_madds[8], h, w, c = sepconv(512, 3, 1, h, w, c)
  total_madds[9], h, w, c = sepconv(512, 3, 1, h, w, c)
  total_madds[10], h, w, c = sepconv(512, 3, 1, h, w, c)
  total_madds[11], h, w, c = sepconv(512, 3, 2, h, w, c)
  total_madds[12], h, w, c = sepconv(1024, 3, 2, h, w, c)
  total_madds[13], h, w, c = sepconv(1024, 3, 1, h, w, c)
  total_madds[14], h, w, c = fc(1000, h, w, c)
  return sum(total_madds)

  

print("Objdet 1080p: {}".format(objdet(34, 60, 1024)))
print("Objdet 2048x850: {}".format(objdet(27, 64, 1024)))
print("Spatial crop 1080p: {}".format(spatial_crop(33, 120, 512)))
print("Spatial crop 2048x850: {}".format(spatial_crop(27, 128, 512)))
print("Discrete 1080p: {}".format(dc_mod(1080, 1920, 3)))
print("Discrete 2048x850: {}".format(dc_mod(425, 2048, 3)))
print("Mobilenet 1080p: {}".format(mobilenet(1080, 1920, 3)))
print("Mobilenet 2048x850: {}".format(mobilenet(850, 2048, 3)))
print("Mobilenet 540x1920: {}".format(mobilenet(540, 1920, 3)))
print("Mobilenet 425x2048: {}".format(mobilenet(425, 2048, 3)))
print("Windowed 2048x850: {}".format(windowed(27, 128, 512)))
