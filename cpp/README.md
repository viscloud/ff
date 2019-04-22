
## Building

FilterForward requires [Caffe](https://github.com/BVLC/caffe) (we use [Intel Caffe](https://github.com/intel/caffe) for our evaluation) and [Tensorflow](https://github.com/tensorflow/tensorflow). Install them using the instructions in their respective documentation.

```sh
mkdir build
cd build
cmake -DCAFFE_HOME=<path to Caffe>/distribute -DTENSORFLOW_HOME=<path to TensorFlow> ..
make
```
The resulting binaries will be located in `build/src`. `filterforward` is, as the name suggests, the FilterForward system. `nnbench` is used for benchmarking straightforward classifier architectures that only consist of one level of network, e.g., multiple copies of a discrete classifier or MobileNet.

## Run clang-format
```sh
make ff-clangformat
```
***Note:*** `make clangformat` will not work, since that target pertains to SAF.
