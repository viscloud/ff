
## Building

```sh
mkdir build
cd build
cmake -DCAFFE_HOME=<path to Caffe> -DTENSORFLOW_HOME=<path to TensorFlow> ..
make
```
The resulting binaries are located in `build/src`.

## Run clang-format
```sh
make ff-clangformat
```
***Note:*** `make clangformat` will not work, since that target pertains to SAF.
